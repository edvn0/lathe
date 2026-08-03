#include "screenshot.hxx"

#include "context.hxx"
#include "error_describe.hxx"
#include "logger.hxx"

#include <stb_image_write.h>

#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <span>
#include <thread>
#include <vector>

namespace {

    auto bytes_per_pixel(VkFormat format) -> std::uint32_t {
        switch (format) {
            case VK_FORMAT_B8G8R8A8_SRGB:
            case VK_FORMAT_B8G8R8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
            case VK_FORMAT_R8G8B8A8_UNORM:
                return 4;

            default:
                return 0;
        }
    }

    auto is_bgra(VkFormat format) -> bool {
        return format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_B8G8R8A8_UNORM;
    }

    auto make_timestamp() -> std::string {
        auto const now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm_buf{};

#if defined(_WIN32)
        localtime_s(&tm_buf, &now);
#else
        localtime_r(&now, &tm_buf);
#endif

        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm_buf);

        return buffer;
    }

    // Runs off the render thread -- PNG encoding a full frame costs tens of
    // milliseconds, which would show up as a frame-time spike if done inline.
    auto write_screenshot_png(std::vector<std::byte> pixels, VkExtent2D extent, VkFormat format) -> void {
        if (is_bgra(format)) {
            for (std::size_t i = 0; i + 4 <= pixels.size(); i += 4) {
                std::swap(pixels[i], pixels[i + 2]);
            }
        }

        std::error_code ec;
        std::filesystem::create_directories("screenshots", ec);

        auto const path = std::format("screenshots/screenshot_{}.png", make_timestamp());
        auto const row_pitch = static_cast<int>(extent.width) * 4;

        if (stbi_write_png(path.c_str(), static_cast<int>(extent.width), static_cast<int>(extent.height), 4,
                           pixels.data(), row_pitch) == 0) {
            error("Screenshot: stbi_write_png failed for '{}'", path);
            return;
        }

        info("Screenshot saved to '{}'", path);
    }

} // namespace

auto ScreenshotCapture::record(VulkanContext &ctx, VkCommandBuffer command_buffer, VkImage image, VkFormat format,
                               VkExtent2D extent, std::uint32_t frame_index) -> bool {
    if (pending_ || !requested_.exchange(false, std::memory_order_relaxed)) {
        return false;
    }

    auto const pixel_size = bytes_per_pixel(format);

    if (pixel_size == 0) {
        error("Screenshot: unsupported swapchain format {}", static_cast<std::uint32_t>(format));
        return false;
    }

    auto const byte_size =
            static_cast<VkDeviceSize>(extent.width) * static_cast<VkDeviceSize>(extent.height) * pixel_size;

    if (!readback_buffer_.valid() || readback_buffer_.size() < byte_size) {
        auto created = Buffer::create(ctx, BufferCreateInfo{
                                                    .size = byte_size,
                                                    .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                    .memory = BufferMemory::readback,
                                                    .debug_name = "screenshot_readback",
                                            });

        if (!created) {
            error("Screenshot: failed to create readback buffer: {}", describe(created.error()));
            return false;
        }

        readback_buffer_ = std::move(*created);
    }

    VkImageMemoryBarrier2 const to_transfer_src{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange =
                    VkImageSubresourceRange{
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                    },
    };

    VkDependencyInfo const to_transfer_src_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &to_transfer_src,
    };

    vkCmdPipelineBarrier2(command_buffer, &to_transfer_src_info);

    VkBufferImageCopy2 const region{
            .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .pNext = nullptr,
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
                    VkImageSubresourceLayers{
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .mipLevel = 0,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                    },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent.width, extent.height, 1},
    };

    VkCopyImageToBufferInfo2 const copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
            .pNext = nullptr,
            .srcImage = image,
            .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .dstBuffer = readback_buffer_.buffer,
            .regionCount = 1,
            .pRegions = &region,
    };

    vkCmdCopyImageToBuffer2(command_buffer, &copy_info);

    VkImageMemoryBarrier2 const to_present{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
            .dstAccessMask = VK_ACCESS_2_NONE,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange =
                    VkImageSubresourceRange{
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                    },
    };

    VkDependencyInfo const to_present_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &to_present,
    };

    vkCmdPipelineBarrier2(command_buffer, &to_present_info);

    pending_ = true;
    pending_frame_index_ = frame_index;
    pending_extent_ = extent;
    pending_format_ = format;

    return true;
}

auto ScreenshotCapture::try_resolve(std::uint32_t frame_index) -> void {
    if (!pending_ || frame_index != pending_frame_index_) {
        return;
    }

    pending_ = false;

    auto const byte_size =
            static_cast<std::size_t>(pending_extent_.width) * static_cast<std::size_t>(pending_extent_.height) * 4U;

    std::vector<std::byte> pixels(byte_size);

    if (auto read = readback_buffer_.read(0, std::span<std::byte>(pixels)); !read) {
        error("Screenshot: failed to read back buffer: {}", describe(read.error()));
        return;
    }

    std::thread([pixels = std::move(pixels), extent = pending_extent_, format = pending_format_]() mutable {
        write_screenshot_png(std::move(pixels), extent, format);
    }).detach();
}
