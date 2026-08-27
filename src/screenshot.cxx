#include "screenshot.hxx"

#include "context.hxx"
#include "error_describe.hxx"
#include "logger.hxx"

#include <stb_image_write.h>

#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <thread>
#include <utility>
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

        char buffer[32]{};
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm_buf);

        return buffer;
    }

    // Runs entirely off the render thread.
    //
    // At this point the pixels are ordinary CPU-owned memory. No Vulkan resources
    // are touched from here onward.
    auto write_screenshot_png(std::vector<std::byte> pixels, VkExtent2D extent, VkFormat format) -> void {

        if (is_bgra(format)) {
            for (std::size_t i = 0; i + 4 <= pixels.size(); i += 4) {
                std::swap(pixels[i], pixels[i + 2]);
            }
        }

        std::error_code ec;
        std::filesystem::create_directories("screenshots", ec);

        if (ec) {
            error("Screenshot: failed to create screenshot directory: {}", ec.message());
            return;
        }

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

ScreenshotCapture::~ScreenshotCapture() {
    // Workers release cpu_busy immediately after memcpy(), before PNG
    // conversion/compression. Therefore waiting for cpu_busy == false is
    // sufficient to guarantee that no worker still references mapped Vulkan
    // memory when the readback buffers are destroyed.
    for (auto const &slot: slots_) {
        if (!slot) {
            continue;
        }

        while (slot->cpu_busy.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
}

auto ScreenshotCapture::get_or_create_slot(std::uint32_t frame_index) -> ReadbackSlot & {

    if (frame_index >= slots_.size()) {
        slots_.resize(static_cast<std::size_t>(frame_index) + 1);
    }

    if (!slots_[frame_index]) {
        slots_[frame_index] = std::make_unique<ReadbackSlot>();
    }

    return *slots_[frame_index];
}

auto ScreenshotCapture::record(VulkanContext &ctx, VkCommandBuffer command_buffer, VkImage image, VkFormat format,
                               VkExtent2D extent, std::uint32_t frame_index) -> bool {

    auto &slot = get_or_create_slot(frame_index);

    // Do not consume the request while this frame slot is unavailable.
    //
    // This means a screenshot requested while the CPU worker is still
    // consuming this slot automatically gets retried on another frame.
    if (slot.gpu_pending || slot.cpu_busy.load(std::memory_order_acquire)) {
        return false;
    }

    if (!requested_.exchange(false, std::memory_order_relaxed)) {
        return false;
    }

    auto const pixel_size = bytes_per_pixel(format);

    if (pixel_size == 0) {
        error("Screenshot: unsupported swapchain format {}", static_cast<std::uint32_t>(format));

        return false;
    }

    auto const byte_size =
            static_cast<VkDeviceSize>(extent.width) * static_cast<VkDeviceSize>(extent.height) * pixel_size;

    if (!slot.buffer.valid() || slot.buffer.size() < byte_size) {

        auto created = Buffer::create(ctx, BufferCreateInfo{
                                                   .size = byte_size,
                                                   .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                   .memory = BufferMemory::readback,
                                                   .debug_name = "screenshot_readback",
                                           });

        if (!created) {
            error("Screenshot: failed to create readback buffer: {}", describe(created.error()));

            // The screenshot request was consumed above, but no capture was
            // actually recorded. Preserve it so a later frame can retry.
            requested_.store(true, std::memory_order_relaxed);

            return false;
        }

        slot.buffer = std::move(*created);
    }

    //
    // Color attachment -> transfer source
    //

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

    //
    // Swapchain image -> readback buffer
    //

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
            .imageExtent =
                    {
                            extent.width,
                            extent.height,
                            1,
                    },
    };

    VkCopyImageToBufferInfo2 const copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
            .pNext = nullptr,
            .srcImage = image,
            .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .dstBuffer = slot.buffer.buffer,
            .regionCount = 1,
            .pRegions = &region,
    };

    vkCmdCopyImageToBuffer2(command_buffer, &copy_info);

    //
    // Transfer source -> presentation
    //

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

    //
    // Transfer write -> host read
    //
    // The frame fence supplies the execution synchronization that ensures
    // this work is complete before try_resolve() is reached for this frame
    // slot. This barrier supplies the corresponding memory dependency.
    //

    VkBufferMemoryBarrier2 const to_host{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
            .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = slot.buffer.buffer,
            .offset = 0,
            .size = byte_size,
    };

    VkDependencyInfo const finish_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &to_host,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &to_present,
    };

    vkCmdPipelineBarrier2(command_buffer, &finish_info);

    slot.gpu_pending = true;

    slot.extent = extent;

    slot.format = format;

    slot.byte_size = static_cast<std::size_t>(byte_size);

    return true;
}

auto ScreenshotCapture::try_resolve(std::uint32_t frame_index) -> void {

    if (frame_index >= slots_.size() || !slots_[frame_index]) {
        return;
    }

    auto &slot = *slots_[frame_index];

    if (!slot.gpu_pending) {
        return;
    }

    // The renderer has already waited for this frame-in-flight slot's fence,
    // so the image -> buffer copy is complete at this point.
    slot.gpu_pending = false;

    slot.cpu_busy.store(true, std::memory_order_release);

    auto *slot_ptr = &slot;

    std::thread([slot_ptr]() {
        //
        // For HOST_COHERENT allocations this is effectively a no-op.
        // For non-coherent readback memory it makes the GPU's writes
        // visible to the CPU before memcpy().
        //
        if (auto invalidated = slot_ptr->buffer.invalidate(0, slot_ptr->byte_size); !invalidated) {

            error("Screenshot: failed to invalidate "
                  "readback buffer: {}",
                  describe(invalidated.error()));

            slot_ptr->cpu_busy.store(false, std::memory_order_release);

            return;
        }

        auto const *mapped = slot_ptr->buffer.mapped_data();

        if (mapped == nullptr) {
            error("Screenshot: readback buffer is not mapped");

            slot_ptr->cpu_busy.store(false, std::memory_order_release);

            return;
        }

        auto const byte_size = slot_ptr->byte_size;

        auto const extent = slot_ptr->extent;

        auto const format = slot_ptr->format;

        std::vector<std::byte> pixels(byte_size);

        std::memcpy(pixels.data(), mapped, byte_size);

        //
        // The Vulkan allocation is no longer touched after this point.
        //
        // Release the slot before doing channel conversion, PNG
        // compression, and disk I/O; those can take considerably
        // longer than the memcpy itself.
        //
        slot_ptr->cpu_busy.store(false, std::memory_order_release);

        write_screenshot_png(std::move(pixels), extent, format);
    }).detach();
}
