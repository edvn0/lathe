#include "gpu/image.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <exr.h>
#include <glm/gtc/packing.hpp>
#include <stb_image.h>

#include "core/logger.hxx"
#include "gpu/buffer.hxx"
#include "gpu/context.hxx"
#include "gpu/vk_object_name.hxx"

namespace {
    [[nodiscard]]
    auto descriptor_view_type(ImageDescriptorView type) noexcept -> VkImageViewType {
        switch (type) {
            case ImageDescriptorView::sampled_2d:
            case ImageDescriptorView::storage_2d:
                return VK_IMAGE_VIEW_TYPE_2D;

            case ImageDescriptorView::sampled_cube:
                return VK_IMAGE_VIEW_TYPE_CUBE;

            case ImageDescriptorView::sampled_2d_array:
            case ImageDescriptorView::storage_2d_array:
                return VK_IMAGE_VIEW_TYPE_2D_ARRAY;

            case ImageDescriptorView::count:
                break;
        }

        return VK_IMAGE_VIEW_TYPE_MAX_ENUM;
    }

    [[nodiscard]]
    auto descriptor_view_usage_is_valid(ImageCreateInfo const &create_info, ImageDescriptorView type) noexcept -> bool {
        switch (type) {
            case ImageDescriptorView::sampled_2d:
                return (create_info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0 &&
                       create_info.samples == VK_SAMPLE_COUNT_1_BIT;

            case ImageDescriptorView::sampled_cube:
                return (create_info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0 &&
                       (create_info.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0 &&
                       create_info.array_layers >= 6 && create_info.samples == VK_SAMPLE_COUNT_1_BIT;

            case ImageDescriptorView::sampled_2d_array:
                return (create_info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0 && create_info.array_layers >= 1 &&
                       create_info.samples == VK_SAMPLE_COUNT_1_BIT;

            case ImageDescriptorView::storage_2d:
                return (create_info.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0 &&
                       create_info.samples == VK_SAMPLE_COUNT_1_BIT;

            case ImageDescriptorView::storage_2d_array:
                return (create_info.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0 && create_info.array_layers >= 1 &&
                       create_info.samples == VK_SAMPLE_COUNT_1_BIT;

            case ImageDescriptorView::count:
                break;
        }

        return false;
    }

    auto make_error(ImageErrorType type, std::string_view message = {}, VkResult result = VK_SUCCESS,
                    std::source_location location = std::source_location::current()) noexcept -> ImageError {
        return ImageError{
                .type = type,
                .cause = ErrorCause{ErrorContext{
                        .message = FlyString{message},
                        .vk_result = result != VK_SUCCESS ? std::optional{result} : std::nullopt,
                        .location = location,
                }},
        };
    }

    auto infer_aspect(VkFormat format) noexcept -> VkImageAspectFlags {
        switch (format) {
            case VK_FORMAT_D16_UNORM:
            case VK_FORMAT_X8_D24_UNORM_PACK32:
            case VK_FORMAT_D32_SFLOAT:
                return VK_IMAGE_ASPECT_DEPTH_BIT;

            case VK_FORMAT_S8_UINT:
                return VK_IMAGE_ASPECT_STENCIL_BIT;

            case VK_FORMAT_D16_UNORM_S8_UINT:
            case VK_FORMAT_D24_UNORM_S8_UINT:
            case VK_FORMAT_D32_SFLOAT_S8_UINT:
                return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

            default:
                return VK_IMAGE_ASPECT_COLOR_BIT;
        }
    }
} // namespace

Image::~Image() { destroy(); }

Image::Image(Image &&other) noexcept :
    context_(std::exchange(other.context_, nullptr)), image_(std::exchange(other.image_, VK_NULL_HANDLE)),
    view_(std::exchange(other.view_, VK_NULL_HANDLE)), descriptor_views_(std::exchange(other.descriptor_views_, {})),
    mip_layer_views_(std::exchange(other.mip_layer_views_, {})),
    allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)), aliased_(std::exchange(other.aliased_, false)),
    allocation_info_(std::exchange(other.allocation_info_, VmaAllocationInfo{})),
    format_(std::exchange(other.format_, VK_FORMAT_UNDEFINED)), extent_(std::exchange(other.extent_, VkExtent3D{})),
    usage_(std::exchange(other.usage_, VkImageUsageFlags{0})),
    aspect_(std::exchange(other.aspect_, VkImageAspectFlags{0})),
    samples_(std::exchange(other.samples_, VK_SAMPLE_COUNT_1_BIT)), mip_levels_(std::exchange(other.mip_levels_, 0)),
    array_layers_(std::exchange(other.array_layers_, 0)) {}

auto Image::operator=(Image &&other) noexcept -> Image & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);
    image_ = std::exchange(other.image_, VK_NULL_HANDLE);
    view_ = std::exchange(other.view_, VK_NULL_HANDLE);
    descriptor_views_ = std::exchange(other.descriptor_views_, {});
    mip_layer_views_ = std::exchange(other.mip_layer_views_, {});
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    aliased_ = std::exchange(other.aliased_, false);
    allocation_info_ = std::exchange(other.allocation_info_, VmaAllocationInfo{});
    format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
    extent_ = std::exchange(other.extent_, VkExtent3D{});
    usage_ = std::exchange(other.usage_, VkImageUsageFlags{0});
    aspect_ = std::exchange(other.aspect_, VkImageAspectFlags{0});
    samples_ = std::exchange(other.samples_, VK_SAMPLE_COUNT_1_BIT);
    mip_levels_ = std::exchange(other.mip_levels_, 0);
    array_layers_ = std::exchange(other.array_layers_, 0);

    return *this;
}

auto Image::create(VulkanContext &context, ImageCreateInfo const &create_info) -> std::expected<Image, ImageError> {
    if (context.device == VK_NULL_HANDLE || context.allocator == VK_NULL_HANDLE || create_info.extent.width == 0 ||
        create_info.extent.height == 0 || create_info.extent.depth == 0 || create_info.format == VK_FORMAT_UNDEFINED ||
        create_info.usage == 0 || create_info.mip_levels == 0 || create_info.array_layers == 0) {
        return std::unexpected(make_error(ImageErrorType::invalid_argument));
    }

    auto const aspect = create_info.aspect != 0 ? create_info.aspect : infer_aspect(create_info.format);

    VkImageCreateInfo const image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = create_info.flags,
            .imageType = create_info.image_type,
            .format = create_info.format,
            .extent = create_info.extent,
            .mipLevels = create_info.mip_levels,
            .arrayLayers = create_info.array_layers,
            .samples = create_info.samples,
            .tiling = create_info.tiling,
            .usage = create_info.usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VmaAllocationCreateInfo const allocation_info{
            .flags = 0,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = 0,
            .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .memoryTypeBits = 0,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
            .priority = 1.0F,
            .minAlignment = 0,
    };

    Image image;
    image.context_ = &context;

    VkResult result = VK_SUCCESS;
    if (create_info.alias.has_value()) {
        // Memory comes from the caller's block; only the VkImage is ours.
        image.aliased_ = true;
        result = vmaCreateAliasingImage2(context.allocator, create_info.alias->allocation, create_info.alias->offset,
                                         &image_info, &image.image_);
    } else {
        result = vmaCreateImage(context.allocator, &image_info, &allocation_info, &image.image_, &image.allocation_,
                                &image.allocation_info_);
    }

    if (result != VK_SUCCESS) {
        image.context_ = nullptr;
        image.aliased_ = false;

        return std::unexpected(make_error(ImageErrorType::image_creation_failed,
                                          std::format("{} failed for image '{}'",
                                                      create_info.alias ? "vmaCreateAliasingImage2" : "vmaCreateImage",
                                                      create_info.debug_name),
                                          result));
    }

    VkImageViewCreateInfo const view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = image.image_,
            .viewType = create_info.view_type,
            .format = create_info.format,
            .components =
                    VkComponentMapping{
                            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                    },
            .subresourceRange =
                    VkImageSubresourceRange{
                            .aspectMask = aspect,
                            .baseMipLevel = 0,
                            .levelCount = create_info.mip_levels,
                            .baseArrayLayer = 0,
                            .layerCount = create_info.array_layers,
                    },
    };

    result = vkCreateImageView(context.device, &view_info, nullptr, &image.view_);

    if (result != VK_SUCCESS) {
        image.destroy();

        return std::unexpected(
                make_error(ImageErrorType::view_creation_failed,
                           std::format("vkCreateImageView failed for image '{}'", create_info.debug_name), result));
    }

    for (std::uint32_t raw_type = 0; raw_type < static_cast<std::uint32_t>(ImageDescriptorView::count); ++raw_type) {
        auto const type = static_cast<ImageDescriptorView>(raw_type);

        if (!has_image_descriptor_view(create_info.descriptor_views, type)) {
            continue;
        }

        if (!descriptor_view_usage_is_valid(create_info, type)) {
            image.destroy();

            return std::unexpected(make_error(ImageErrorType::invalid_argument));
        }

        auto const view_type = descriptor_view_type(type);

        auto layer_count = create_info.array_layers;

        if (type == ImageDescriptorView::sampled_cube) {
            layer_count = 6;
        }

        VkImageViewCreateInfo const descriptor_view_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = image.image_,
                .viewType = view_type,
                .format = create_info.format,
                .components =
                        VkComponentMapping{
                                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                        },
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = aspect,
                                .baseMipLevel = 0,
                                .levelCount = create_info.mip_levels,
                                .baseArrayLayer = 0,
                                .layerCount = layer_count,
                        },
        };

        auto const descriptor_view_index = static_cast<std::size_t>(type);

        result = vkCreateImageView(context.device, &descriptor_view_info, nullptr,
                                   &image.descriptor_views_[descriptor_view_index]);

        if (result != VK_SUCCESS) {
            image.destroy();

            return std::unexpected(
                    make_error(ImageErrorType::view_creation_failed,
                               std::format("vkCreateImageView (descriptor view {}) failed for image '{}'", raw_type,
                                           create_info.debug_name),
                               result));
        }

        auto const descriptor_view_name =
                std::string{create_info.debug_name} + ".descriptor_view." + std::to_string(raw_type);

        vk::set_object_name(context.device, VK_OBJECT_TYPE_IMAGE_VIEW,
                            vk::object_handle(image.descriptor_views_[descriptor_view_index]), descriptor_view_name);
    }

    image.format_ = create_info.format;
    image.extent_ = create_info.extent;
    image.usage_ = create_info.usage;
    image.aspect_ = aspect;
    image.samples_ = create_info.samples;
    image.mip_levels_ = create_info.mip_levels;
    image.array_layers_ = create_info.array_layers;

    if (create_info.create_mip_layer_views) {
        image.mip_layer_views_.assign(static_cast<std::size_t>(create_info.mip_levels) * create_info.array_layers,
                                      VK_NULL_HANDLE);

        for (std::uint32_t mip = 0; mip < create_info.mip_levels; ++mip) {
            for (std::uint32_t layer = 0; layer < create_info.array_layers; ++layer) {
                VkImageViewCreateInfo const slice_view_info{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .image = image.image_,
                        .viewType = VK_IMAGE_VIEW_TYPE_2D,
                        .format = create_info.format,
                        .components =
                                VkComponentMapping{
                                        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                                        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                                        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                                        .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                                },
                        .subresourceRange =
                                VkImageSubresourceRange{
                                        .aspectMask = aspect,
                                        .baseMipLevel = mip,
                                        .levelCount = 1,
                                        .baseArrayLayer = layer,
                                        .layerCount = 1,
                                },
                };

                auto const slice_index = image.mip_layer_view_index(mip, layer);

                result = vkCreateImageView(context.device, &slice_view_info, nullptr,
                                           &image.mip_layer_views_[slice_index]);

                if (result != VK_SUCCESS) {
                    image.destroy();

                    return std::unexpected(
                            make_error(ImageErrorType::view_creation_failed,
                                       std::format("vkCreateImageView (mip {} layer {}) failed for image '{}'", mip,
                                                   layer, create_info.debug_name),
                                       result));
                }

                auto const slice_name = std::string{create_info.debug_name} + ".mip_layer_view." + std::to_string(mip) +
                                        "." + std::to_string(layer);

                vk::set_object_name(context.device, VK_OBJECT_TYPE_IMAGE_VIEW,
                                    vk::object_handle(image.mip_layer_views_[slice_index]), slice_name);
            }
        }
    }

    vk::set_object_name(context.device, VK_OBJECT_TYPE_IMAGE, vk::object_handle(image.image_), create_info.debug_name);
    auto const view_name = std::string{create_info.debug_name} + ".view";
    vk::set_object_name(context.device, VK_OBJECT_TYPE_IMAGE_VIEW, vk::object_handle(image.view_), view_name);

    return image;
}

auto Image::create(VulkanContext &context, ImageCreateInfo const &create_info, std::span<const std::byte> pixels,
                   ImageMipSource mip_source) -> std::expected<Image, ImageError> {
    // A provided chain must be exactly every level of a whole number of bytes per texel.
    std::uint32_t chain_texel_bytes = 0;
    if (mip_source == ImageMipSource::provided) {
        auto const texels =
                mip_chain_offset(create_info.extent.width, create_info.extent.height, 1, create_info.mip_levels);
        if (texels == 0 || pixels.size_bytes() % texels != 0) {
            return std::unexpected(ImageError{.type = ImageErrorType::image_creation_failed});
        }
        chain_texel_bytes = static_cast<std::uint32_t>(pixels.size_bytes() / texels);
    }

    auto image = create(context, create_info);
    if (!image) {
        return std::unexpected(image.error());
    }

    auto maybe_staging = Buffer::create(context, BufferCreateInfo{
                                                         .size = pixels.size_bytes(),
                                                         .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                         .memory = BufferMemory::upload,
                                                         .debug_name = "image_upload",
                                                 });
    if (!maybe_staging) {
        return std::unexpected(ImageError{
                .type = ImageErrorType::image_creation_failed,
                .cause = ErrorCause{Boxed<DeviceError>{maybe_staging.error()}},
        });
    }

    auto staging = std::move(*maybe_staging);
    if (!staging.write(0, pixels)) {
        return std::unexpected(ImageError{
                .type = ImageErrorType::image_creation_failed,
                .cause = ErrorCause{ErrorContext{
                        .message = FlyString{"staging buffer write failed"},
                        .vk_result = VK_ERROR_DEVICE_LOST,
                }},
        });
    }

    context.one_time_submit([&](VkCommandBuffer buf) {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        barrier.srcAccessMask = 0;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.image = image->image();
        barrier.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                    .baseMipLevel = 0,
                                    .levelCount = create_info.mip_levels,
                                    .baseArrayLayer = 0,
                                    .layerCount = 1};

        VkDependencyInfo dep_info{};
        dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_info.imageMemoryBarrierCount = 1;
        dep_info.pImageMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(buf, &dep_info);

        if (mip_source == ImageMipSource::provided) {
            std::vector<VkBufferImageCopy> copies;
            copies.reserve(create_info.mip_levels);
            for (std::uint32_t level = 0; level < create_info.mip_levels; ++level) {
                copies.push_back(VkBufferImageCopy{
                        .bufferOffset = mip_chain_offset(create_info.extent.width, create_info.extent.height,
                                                         chain_texel_bytes, level),
                        .bufferRowLength = 0,
                        .bufferImageHeight = 0,
                        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                             .mipLevel = level,
                                             .baseArrayLayer = 0,
                                             .layerCount = 1},
                        .imageOffset = {},
                        .imageExtent = {.width = std::max(create_info.extent.width >> level, 1U),
                                        .height = std::max(create_info.extent.height >> level, 1U),
                                        .depth = 1},
                });
            }
            vkCmdCopyBufferToImage(buf, staging.buffer, image->image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   static_cast<std::uint32_t>(copies.size()), copies.data());

            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = create_info.mip_levels;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier2(buf, &dep_info);
            return;
        }

        VkBufferImageCopy copy{};
        copy.imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1};
        copy.imageExtent = {.width = create_info.extent.width, .height = create_info.extent.height, .depth = 1};
        vkCmdCopyBufferToImage(buf, staging.buffer, image->image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        // Generate mips.
        auto mip_width = static_cast<std::int32_t>(create_info.extent.width);
        auto mip_height = static_cast<std::int32_t>(create_info.extent.height);

        for (uint32_t i = 1; i < create_info.mip_levels; i++) {
            // Level i-1 to TRANSFER_SRC_OPTIMAL.
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.subresourceRange.levelCount = 1;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; // from buffer copy or previous blit
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            vkCmdPipelineBarrier2(buf, &dep_info);

            std::int32_t next_width = mip_width > 1 ? mip_width / 2 : 1;
            std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;

            VkImageBlit blit{};
            blit.srcOffsets[0] = {.x = 0, .y = 0, .z = 0};
            blit.srcOffsets[1] = {.x = mip_width, .y = mip_height, .z = 1};
            blit.srcSubresource = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = i - 1, .baseArrayLayer = 0, .layerCount = 1};
            blit.dstOffsets[0] = {.x = 0, .y = 0, .z = 0};
            blit.dstOffsets[1] = {.x = next_width, .y = next_height, .z = 1};
            blit.dstSubresource = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = i, .baseArrayLayer = 0, .layerCount = 1};

            vkCmdBlitImage(buf, image->image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image->image(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

            // Level i-1 to SHADER_READ_ONLY_OPTIMAL.
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier2(buf, &dep_info);

            mip_width = next_width;
            mip_height = next_height;
        }

        // The last level.
        barrier.subresourceRange.baseMipLevel = create_info.mip_levels - 1;
        barrier.subresourceRange.levelCount = 1;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier2(buf, &dep_info);
    });

    return image;
}

auto Image::memory_requirements(VulkanContext &context, ImageCreateInfo const &create_info) -> VkMemoryRequirements {
    VkImageCreateInfo const image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = create_info.flags,
            .imageType = create_info.image_type,
            .format = create_info.format,
            .extent = create_info.extent,
            .mipLevels = create_info.mip_levels,
            .arrayLayers = create_info.array_layers,
            .samples = create_info.samples,
            .tiling = create_info.tiling,
            .usage = create_info.usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkDeviceImageMemoryRequirements const query{
            .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS,
            .pNext = nullptr,
            .pCreateInfo = &image_info,
            .planeAspect = VK_IMAGE_ASPECT_NONE,
    };
    VkMemoryRequirements2 requirements{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetDeviceImageMemoryRequirements(context.device, &query, &requirements);
    return requirements.memoryRequirements;
}

auto Image::destroy() noexcept -> void {
    if (context_ == nullptr) {
        return;
    }

    for (auto &descriptor_view: descriptor_views_) {
        if (descriptor_view != VK_NULL_HANDLE && context_->device != VK_NULL_HANDLE) {
            vkDestroyImageView(context_->device, descriptor_view, nullptr);
        }

        descriptor_view = VK_NULL_HANDLE;
    }

    if (view_ != VK_NULL_HANDLE && context_->device != VK_NULL_HANDLE) {
        vkDestroyImageView(context_->device, view_, nullptr);
    }

    view_ = VK_NULL_HANDLE;

    if (aliased_) {
        if (image_ != VK_NULL_HANDLE && context_->device != VK_NULL_HANDLE) {
            vkDestroyImage(context_->device, image_, nullptr);
        }
    } else if (image_ != VK_NULL_HANDLE && allocation_ != VK_NULL_HANDLE && context_->allocator != VK_NULL_HANDLE) {
        vmaDestroyImage(context_->allocator, image_, allocation_);
    }

    for (auto &slice_view: mip_layer_views_) {
        if (slice_view != VK_NULL_HANDLE && context_->device != VK_NULL_HANDLE) {
            vkDestroyImageView(context_->device, slice_view, nullptr);
        }
    }

    mip_layer_views_.clear();

    image_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    aliased_ = false;
    allocation_info_ = {};

    format_ = VK_FORMAT_UNDEFINED;
    extent_ = {};
    usage_ = 0;
    aspect_ = 0;
    samples_ = VK_SAMPLE_COUNT_1_BIT;
    mip_levels_ = 0;
    array_layers_ = 0;

    context_ = nullptr;
}

namespace {

    [[nodiscard]]
    auto lowercase_extension(std::string_view path) -> std::string {
        auto extension = std::filesystem::path{std::string{path}}.extension().string();

        std::ranges::transform(extension, extension.begin(),
                               [](unsigned char value) { return static_cast<char>(std::tolower(value)); });

        return extension;
    }

    [[nodiscard]]
    auto exr_channel_leaf_name(char const *name) noexcept -> std::string_view {
        auto const full_name = std::string_view{name};

        auto const separator = full_name.find_last_of('.');

        if (separator == std::string_view::npos) {
            return full_name;
        }

        return full_name.substr(separator + 1);
    }

    [[nodiscard]]
    auto exr_channel_index(exr_header const &header, std::string_view wanted_name) noexcept
            -> std::optional<std::size_t> {
        for (std::int32_t index = 0; index < header.num_channels; ++index) {
            auto const channel_name = exr_channel_leaf_name(header.channels[index].name);

            if (channel_name == wanted_name) {
                return static_cast<std::size_t>(index);
            }
        }

        return std::nullopt;
    }

    [[nodiscard]]
    auto exr_channel_value(exr_part const &part, std::size_t channel_index, std::size_t pixel_index) noexcept
            -> std::optional<float> {
        if (channel_index >= static_cast<std::size_t>(part.header.num_channels)) {
            return std::nullopt;
        }

        auto const &channel = part.header.channels[channel_index];

        if (channel.x_sampling != 1 || channel.y_sampling != 1) {
            return std::nullopt;
        }

        auto const *channel_data = part.images[channel_index];

        if (channel_data == nullptr) {
            return std::nullopt;
        }

        switch (channel.pixel_type) {
            case EXR_PIXEL_HALF: {
                auto const *values = static_cast<std::uint16_t const *>(channel_data);

                return glm::unpackHalf1x16(values[pixel_index]);
            }

            case EXR_PIXEL_FLOAT: {
                auto const *values = static_cast<float const *>(channel_data);

                return values[pixel_index];
            }

            case EXR_PIXEL_UINT: {
                // OpenEXR UINT channels are integers, not normalized colour.
                return std::nullopt;
            }
        }

        return std::nullopt;
    }

} // namespace

[[nodiscard]]
auto DecodedImage::decode_exr(std::string_view path) -> std::optional<DecodedImage> {
    auto const path_string = std::string{path};

    exr_image image{};

    auto const result = exr_load_from_file(path_string.c_str(), nullptr, &image);

    if (result != EXR_SUCCESS) {
        error("Could not decode EXR '{}': {}", path, exr_result_string(result));

        return std::nullopt;
    }

    struct ImageCleanup {
        exr_image *image = nullptr;

        ~ImageCleanup() {
            if (image != nullptr) {
                exr_image_free(image);
            }
        }
    };

    auto const cleanup = ImageCleanup{&image};

    if (image.num_parts != 1 || image.parts == nullptr) {
        error("EXR '{}' has {} parts; texture loading currently requires exactly one", path, image.num_parts);

        return std::nullopt;
    }

    auto const &part = image.parts[0];

    if (part.is_deep != 0) {
        error("EXR '{}' is a deep image, which is not supported as a 2D texture", path);

        return std::nullopt;
    }

    if (part.images == nullptr || part.width <= 0 || part.height <= 0) {
        error("EXR '{}' contains no usable pixel data", path);
        return std::nullopt;
    }

    auto const width = static_cast<std::uint32_t>(part.width);
    auto const height = static_cast<std::uint32_t>(part.height);
    auto const width_size = static_cast<std::size_t>(width);
    auto const height_size = static_cast<std::size_t>(height);

    if (height_size != 0 && width_size > std::numeric_limits<std::size_t>::max() / height_size) {
        error("EXR '{}' dimensions overflow size_t", path);
        return std::nullopt;
    }

    auto const pixel_count = width_size * height_size;
    auto const &header = part.header;
    auto const red_channel = exr_channel_index(header, "R");
    auto const green_channel = exr_channel_index(header, "G");
    auto const blue_channel = exr_channel_index(header, "B");
    auto const alpha_channel = exr_channel_index(header, "A");
    auto const luminance_channel = exr_channel_index(header, "Y");

    std::optional<std::size_t> scalar_channel;
    if (luminance_channel.has_value()) {
        scalar_channel = luminance_channel;
    } else if (header.num_channels == 1) {
        scalar_channel = 0;
    }

    auto const has_rgb = red_channel.has_value() || green_channel.has_value() || blue_channel.has_value();

    if (!has_rgb && !scalar_channel.has_value()) {
        error("EXR '{}' does not contain RGB, Y, or a single scalar channel", path);

        return std::nullopt;
    }

    std::vector<std::uint16_t> half_pixels;
    half_pixels.resize(pixel_count * 4);

    auto sample = [&](std::optional<std::size_t> channel, std::size_t pixel, float fallback) -> std::optional<float> {
        if (!channel.has_value()) {
            return fallback;
        }

        return exr_channel_value(part, *channel, pixel);
    };

    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
        float scalar = 0.0F;

        if (scalar_channel.has_value()) {
            auto const value = exr_channel_value(part, *scalar_channel, pixel);

            if (!value.has_value()) {
                error("EXR '{}' contains an unsupported scalar channel format", path);

                return std::nullopt;
            }

            scalar = *value;
        }

        auto red = sample(red_channel, pixel, scalar);

        auto green = sample(green_channel, pixel, scalar);

        auto blue = sample(blue_channel, pixel, scalar);

        auto alpha = sample(alpha_channel, pixel, 1.0F);

        if (!red || !green || !blue || !alpha) {
            error("EXR '{}' contains an unsupported channel format", path);

            return std::nullopt;
        }

        auto const destination = pixel * 4;

        half_pixels[destination + 0] = glm::packHalf1x16(*red);

        half_pixels[destination + 1] = glm::packHalf1x16(*green);

        half_pixels[destination + 2] = glm::packHalf1x16(*blue);

        half_pixels[destination + 3] = glm::packHalf1x16(*alpha);
    }

    auto const half_bytes = std::as_bytes(std::span<std::uint16_t const>{half_pixels});

    std::vector<std::byte> pixels{
            half_bytes.begin(),
            half_bytes.end(),
    };

    debug("Decoded EXR '{}' as {}x{} RGBA16F, {} source channels", path, width, height, header.num_channels);

    return DecodedImage{
            std::move(pixels),
            width,
            height,
            VK_FORMAT_R16G16B16A16_SFLOAT,
    };
}

[[nodiscard]]
auto DecodedImage::decode_stbi(std::string_view path, ImageColourSpace colour_space) -> std::optional<DecodedImage> {
    auto const path_string = std::string{path};

    int width = 0;
    int height = 0;
    int source_channels = 0;

    auto *decoded = stbi_load(path_string.c_str(), &width, &height, &source_channels, STBI_rgb_alpha);

    if (decoded == nullptr) {
        error("Could not decode image '{}': {}", path, stbi_failure_reason());

        return std::nullopt;
    }

    struct StbiCleanup {
        unsigned char *pixels = nullptr;

        ~StbiCleanup() {
            if (pixels != nullptr) {
                stbi_image_free(pixels);
            }
        }
    };

    auto const cleanup = StbiCleanup{decoded};

    if (width <= 0 || height <= 0) {
        error("Decoded image '{}' has invalid dimensions {}x{}", path, width, height);

        return std::nullopt;
    }

    auto const width_size = static_cast<std::size_t>(width);

    auto const height_size = static_cast<std::size_t>(height);

    if (height_size != 0 && width_size > std::numeric_limits<std::size_t>::max() / height_size) {
        error("Decoded image '{}' dimensions overflow size_t", path);

        return std::nullopt;
    }

    auto const pixel_count = width_size * height_size;

    if (pixel_count > std::numeric_limits<std::size_t>::max() / 4) {
        error("Decoded image '{}' byte count overflows size_t", path);

        return std::nullopt;
    }

    auto const byte_count = pixel_count * 4;

    auto const *begin = reinterpret_cast<std::byte const *>(decoded);

    std::vector<std::byte> pixels{
            begin,
            begin + byte_count,
    };

    auto const format = colour_space == ImageColourSpace::srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

    return DecodedImage{
            std::move(pixels),
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            format,
    };
}


namespace {
    // Block decoders for the legacy DDS fourCCs (DXT1/DXT5/ATI2), base mip only. The texture pipeline re-encodes and
    // mips the result, so only the pixels matter here.

    auto expand_565(std::uint16_t packed) noexcept -> std::array<std::uint8_t, 3> {
        auto const r = static_cast<std::uint32_t>((packed >> 11) & 0x1FU);
        auto const g = static_cast<std::uint32_t>((packed >> 5) & 0x3FU);
        auto const b = static_cast<std::uint32_t>(packed & 0x1FU);

        return {static_cast<std::uint8_t>((r << 3) | (r >> 2)), static_cast<std::uint8_t>((g << 2) | (g >> 4)),
                static_cast<std::uint8_t>((b << 3) | (b >> 2))};
    }

    template<typename T>
    auto read_le(std::byte const *source) noexcept -> T {
        T value{};
        std::memcpy(&value, source, sizeof(T));
        return value;
    }

    // Writes one 4x4 colour block (BC1 layout) as RGBA8 into `out`; `allow_alpha` enables BC1's 1-bit punch-through.
    auto decode_bc1_colour(std::byte const *block, bool allow_alpha, std::array<std::uint8_t, 64> &out) noexcept -> void {
        auto const c0 = read_le<std::uint16_t>(block);
        auto const c1 = read_le<std::uint16_t>(block + 2);
        auto const indices = read_le<std::uint32_t>(block + 4);

        std::array<std::array<std::uint8_t, 4>, 4> palette{};
        auto const e0 = expand_565(c0);
        auto const e1 = expand_565(c1);

        for (std::size_t i = 0; i < 3; ++i) {
            palette[0][i] = e0[i];
            palette[1][i] = e1[i];

            if (c0 > c1 || !allow_alpha) {
                palette[2][i] = static_cast<std::uint8_t>((2U * e0[i] + e1[i]) / 3U);
                palette[3][i] = static_cast<std::uint8_t>((e0[i] + 2U * e1[i]) / 3U);
            } else {
                palette[2][i] = static_cast<std::uint8_t>((e0[i] + e1[i]) / 2U);
                palette[3][i] = 0;
            }
        }

        palette[0][3] = palette[1][3] = palette[2][3] = 255;
        palette[3][3] = (c0 > c1 || !allow_alpha) ? 255 : 0;

        for (std::size_t texel = 0; texel < 16; ++texel) {
            auto const &colour = palette[(indices >> (2U * texel)) & 0x3U];
            std::ranges::copy(colour, out.begin() + static_cast<std::ptrdiff_t>(texel * 4));
        }
    }

    // BC4-style single-channel block: 8 bytes -> 16 values.
    auto decode_bc4_block(std::byte const *block, std::array<std::uint8_t, 16> &out) noexcept -> void {
        auto const a0 = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(block[0]));
        auto const a1 = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(block[1]));

        std::array<std::uint8_t, 8> palette{};
        palette[0] = static_cast<std::uint8_t>(a0);
        palette[1] = static_cast<std::uint8_t>(a1);

        if (a0 > a1) {
            for (std::uint32_t i = 1; i <= 6; ++i) {
                palette[i + 1] = static_cast<std::uint8_t>(((7U - i) * a0 + i * a1) / 7U);
            }
        } else {
            for (std::uint32_t i = 1; i <= 4; ++i) {
                palette[i + 1] = static_cast<std::uint8_t>(((5U - i) * a0 + i * a1) / 5U);
            }
            palette[6] = 0;
            palette[7] = 255;
        }

        std::uint64_t bits = 0;
        for (std::size_t i = 0; i < 6; ++i) {
            bits |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(block[2 + i])) << (8U * i);
        }

        for (std::size_t texel = 0; texel < 16; ++texel) {
            out[texel] = palette[(bits >> (3U * texel)) & 0x7U];
        }
    }

    constexpr std::uint32_t dds_magic = 0x20534444U;
    constexpr std::uint32_t dds_header_size = 124;

    constexpr auto make_fourcc(char a, char b, char c, char d) noexcept -> std::uint32_t {
        return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8U) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16U) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24U);
    }

    auto read_whole_file(std::string_view path) -> std::optional<std::vector<std::byte>> {
        std::ifstream file{std::filesystem::path{path}, std::ios::binary | std::ios::ate};

        if (!file) {
            return std::nullopt;
        }

        auto const size = file.tellg();

        if (size <= 0) {
            return std::nullopt;
        }

        std::vector<std::byte> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(bytes.data()), size);

        if (!file) {
            return std::nullopt;
        }

        return bytes;
    }
} // namespace

auto DecodedImage::decode_dds(std::string_view path, ImageColourSpace colour_space) -> std::optional<DecodedImage> {
    auto const bytes = read_whole_file(path);

    if (!bytes || bytes->size() < 4 + dds_header_size || read_le<std::uint32_t>(bytes->data()) != dds_magic ||
        read_le<std::uint32_t>(bytes->data() + 4) != dds_header_size) {
        error("Could not decode DDS '{}': not a DDS file", path);
        return std::nullopt;
    }

    auto const *header = bytes->data() + 4;
    auto const height = read_le<std::uint32_t>(header + 8);
    auto const width = read_le<std::uint32_t>(header + 12);
    auto const fourcc = read_le<std::uint32_t>(header + 80);

    auto const is_bc1 = fourcc == make_fourcc('D', 'X', 'T', '1');
    auto const is_bc3 = fourcc == make_fourcc('D', 'X', 'T', '5');
    auto const is_bc5 = fourcc == make_fourcc('A', 'T', 'I', '2');
    auto const is_bc4 = fourcc == make_fourcc('A', 'T', 'I', '1');

    if (!is_bc1 && !is_bc3 && !is_bc5 && !is_bc4) {
        error("Could not decode DDS '{}': unsupported fourCC 0x{:08X}", path, fourcc);
        return std::nullopt;
    }

    if (width == 0 || height == 0 || width > 16384 || height > 16384) {
        error("Could not decode DDS '{}': invalid size {}x{}", path, width, height);
        return std::nullopt;
    }

    auto const block_bytes = (is_bc1 || is_bc4) ? std::size_t{8} : std::size_t{16};
    auto const blocks_x = (static_cast<std::size_t>(width) + 3U) / 4U;
    auto const blocks_y = (static_cast<std::size_t>(height) + 3U) / 4U;
    auto const data_offset = 4U + dds_header_size;

    if (bytes->size() < data_offset + (blocks_x * blocks_y * block_bytes)) {
        error("Could not decode DDS '{}': truncated", path);
        return std::nullopt;
    }

    std::vector<std::byte> pixels(static_cast<std::size_t>(width) * height * 4U);
    auto *out_pixels = reinterpret_cast<std::uint8_t *>(pixels.data());

    for (std::size_t block_y = 0; block_y < blocks_y; ++block_y) {
        for (std::size_t block_x = 0; block_x < blocks_x; ++block_x) {
            auto const *block = bytes->data() + data_offset + ((block_y * blocks_x) + block_x) * block_bytes;
            std::array<std::uint8_t, 64> rgba{};

            if (is_bc1) {
                decode_bc1_colour(block, true, rgba);
            } else if (is_bc3) {
                std::array<std::uint8_t, 16> alpha{};
                decode_bc4_block(block, alpha);
                decode_bc1_colour(block + 8, false, rgba);

                for (std::size_t texel = 0; texel < 16; ++texel) {
                    rgba[(texel * 4) + 3] = alpha[texel];
                }
            } else if (is_bc4) {
                std::array<std::uint8_t, 16> red{};
                decode_bc4_block(block, red);

                for (std::size_t texel = 0; texel < 16; ++texel) {
                    rgba[(texel * 4) + 0] = rgba[(texel * 4) + 1] = rgba[(texel * 4) + 2] = red[texel];
                    rgba[(texel * 4) + 3] = 255;
                }
            } else {
                std::array<std::uint8_t, 16> red{};
                std::array<std::uint8_t, 16> green{};
                decode_bc4_block(block, red);
                decode_bc4_block(block + 8, green);

                for (std::size_t texel = 0; texel < 16; ++texel) {
                    auto const x = (static_cast<float>(red[texel]) / 127.5F) - 1.0F;
                    auto const y = (static_cast<float>(green[texel]) / 127.5F) - 1.0F;
                    auto const z = std::sqrt(std::max(0.0F, 1.0F - (x * x) - (y * y)));

                    rgba[(texel * 4) + 0] = red[texel];
                    rgba[(texel * 4) + 1] = green[texel];
                    rgba[(texel * 4) + 2] = static_cast<std::uint8_t>(std::lround((z * 0.5F + 0.5F) * 255.0F));
                    rgba[(texel * 4) + 3] = 255;
                }
            }

            for (std::size_t texel = 0; texel < 16; ++texel) {
                auto const x = (block_x * 4) + (texel % 4);
                auto const y = (block_y * 4) + (texel / 4);

                if (x < width && y < height) {
                    std::memcpy(out_pixels + (((y * width) + x) * 4U), rgba.data() + (texel * 4), 4);
                }
            }
        }
    }

    return DecodedImage{std::move(pixels), width, height,
                        colour_space == ImageColourSpace::srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM};
}

auto classify_dds_alpha(std::string_view path) -> std::optional<AlphaCoverage> {
    auto const bytes = read_whole_file(path);

    if (!bytes || bytes->size() < 4 + dds_header_size || read_le<std::uint32_t>(bytes->data()) != dds_magic) {
        return std::nullopt;
    }

    auto const *header = bytes->data() + 4;

    // Only BC3 carries a full alpha channel; the other legacy formats here have none (BC1's 1-bit alpha is not used).
    if (read_le<std::uint32_t>(header + 80) != make_fourcc('D', 'X', 'T', '5')) {
        return AlphaCoverage::opaque;
    }

    auto const height = read_le<std::uint32_t>(header + 8);
    auto const width = read_le<std::uint32_t>(header + 12);
    auto const blocks = ((static_cast<std::size_t>(width) + 3U) / 4U) * ((static_cast<std::size_t>(height) + 3U) / 4U);
    auto const data_offset = 4U + dds_header_size;

    if (blocks == 0 || bytes->size() < data_offset + (blocks * 16U)) {
        return std::nullopt;
    }

    std::size_t transparent = 0; // alpha below half
    std::size_t partial = 0; // neither (nearly) clear nor (nearly) solid

    for (std::size_t block = 0; block < blocks; ++block) {
        std::array<std::uint8_t, 16> alpha{};
        decode_bc4_block(bytes->data() + data_offset + (block * 16U), alpha);

        for (auto const value: alpha) {
            transparent += value < 128 ? 1U : 0U;
            partial += (value > 8 && value < 247) ? 1U : 0U;
        }
    }

    if (transparent == 0) {
        return AlphaCoverage::opaque;
    }

    // Cut-outs (leaves, grilles) are almost all clear or solid with a thin soft edge; real translucency isn't.
    constexpr double max_mask_partial_fraction = 0.12;

    return static_cast<double>(partial) / static_cast<double>(blocks * 16U) <= max_mask_partial_fraction
                   ? AlphaCoverage::mask
                   : AlphaCoverage::blend;
}

DecodedImage::DecodedImage(std::vector<std::byte> pixels, std::uint32_t width, std::uint32_t height,
                           VkFormat format) noexcept :
    pixels_(std::move(pixels)), width_(width), height_(height), format_(format) {}

auto DecodedImage::load_from_file(std::string_view path, ImageColourSpace colour_space) -> std::optional<DecodedImage> {
    auto const extension = lowercase_extension(path);

    if (extension == ".exr") {
        // OpenEXR data is linear, so never use an sRGB format.
        if (colour_space == ImageColourSpace::srgb) {
            warn("EXR '{}' requested as sRGB; EXR texture data is loaded as linear", path);
        }

        return decode_exr(path);
    }

    if (extension == ".dds") {
        return decode_dds(path, colour_space);
    }

    return decode_stbi(path, colour_space);
}

auto DecodedImage::span() const noexcept -> std::span<std::byte const> { return pixels_; }

auto DecodedImage::load_from_memory(std::span<std::byte const> encoded, ImageColourSpace colour_space)
        -> std::optional<DecodedImage> {
    if (encoded.empty()) {
        error("Could not decode image from empty memory buffer");
        return std::nullopt;
    }

    if (encoded.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        error("Could not decode image from memory: {} byte buffer exceeds stb_image limit", encoded.size());

        return std::nullopt;
    }

    int width = 0;
    int height = 0;
    int source_channels = 0;

    auto *decoded =
            stbi_load_from_memory(reinterpret_cast<stbi_uc const *>(encoded.data()), static_cast<int>(encoded.size()),
                                  &width, &height, &source_channels, STBI_rgb_alpha);

    if (decoded == nullptr) {
        error("Could not decode image from memory: {}", stbi_failure_reason());

        return std::nullopt;
    }

    struct StbiCleanup {
        unsigned char *pixels = nullptr;

        ~StbiCleanup() {
            if (pixels != nullptr) {
                stbi_image_free(pixels);
            }
        }
    };

    auto const cleanup = StbiCleanup{
            .pixels = decoded,
    };

    if (width <= 0 || height <= 0) {
        error("Decoded image has invalid dimensions {}x{}", width, height);

        return std::nullopt;
    }

    auto const width_size = static_cast<std::size_t>(width);

    auto const height_size = static_cast<std::size_t>(height);

    if (height_size != 0 && width_size > std::numeric_limits<std::size_t>::max() / height_size) {
        error("Decoded image dimensions overflow size_t: {}x{}", width, height);

        return std::nullopt;
    }

    auto const pixel_count = width_size * height_size;

    if (pixel_count > std::numeric_limits<std::size_t>::max() / 4U) {
        error("Decoded image byte count overflows size_t");
        return std::nullopt;
    }

    auto const byte_count = pixel_count * 4U;

    auto const *begin = reinterpret_cast<std::byte const *>(decoded);

    std::vector<std::byte> pixels{
            begin,
            begin + byte_count,
    };

    auto const format = colour_space == ImageColourSpace::srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

    return DecodedImage{
            std::move(pixels),
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            format,
    };
}
