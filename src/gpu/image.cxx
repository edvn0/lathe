#include "gpu/image.hxx"

#include <expected>
#include <format>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <exr.h>
#include <glm/gtc/packing.hpp>
#include <stb_image.h>

#include "gpu/buffer.hxx"
#include "gpu/context.hxx"
#include "core/logger.hxx"
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
    allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
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

    auto result = vmaCreateImage(context.allocator, &image_info, &allocation_info, &image.image_, &image.allocation_,
                                 &image.allocation_info_);

    if (result != VK_SUCCESS) {
        image.context_ = nullptr;

        return std::unexpected(make_error(ImageErrorType::image_creation_failed,
                                          std::format("vmaCreateImage failed for image '{}'", create_info.debug_name),
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
        vmaDestroyImage(context.allocator, image.image_, image.allocation_);

        image.context_ = nullptr;
        image.image_ = VK_NULL_HANDLE;
        image.allocation_ = VK_NULL_HANDLE;

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

auto Image::create(VulkanContext &context, ImageCreateInfo const &create_info, std::span<const std::byte> pixels)
        -> std::expected<Image, ImageError> {
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
                .cause = ErrorCause{Boxed<DeviceError>{std::move(maybe_staging.error())}},
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
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, create_info.mip_levels, 0, 1};

        VkDependencyInfo dep_info{};
        dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_info.imageMemoryBarrierCount = 1;
        dep_info.pImageMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(buf, &dep_info);

        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {create_info.extent.width, create_info.extent.height, 1};
        vkCmdCopyBufferToImage(buf, staging.buffer, image->image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        // 3. Generate Mips
        std::int32_t mip_width = static_cast<std::int32_t>(create_info.extent.width);
        std::int32_t mip_height = static_cast<std::int32_t>(create_info.extent.height);

        for (uint32_t i = 1; i < create_info.mip_levels; i++) {
            // Transition i-1 to TRANSFER_SRC_OPTIMAL
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.subresourceRange.levelCount = 1;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; // from buffer copy or previous blit
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            vkCmdPipelineBarrier2(buf, &dep_info);

            // Calculate next mip dimensions (preventing going below 1)
            std::int32_t next_width = mip_width > 1 ? mip_width / 2 : 1;
            std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;

            VkImageBlit blit{};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mip_width, mip_height, 1};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {next_width, next_height, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};

            vkCmdBlitImage(buf, image->image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image->image(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

            // Transition i-1 to SHADER_READ_ONLY_OPTIMAL
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

        // 4. Transition the final mip level
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

    if (image_ != VK_NULL_HANDLE && allocation_ != VK_NULL_HANDLE && context_->allocator != VK_NULL_HANDLE) {
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
                // UINT in OpenEXR is an actual integer channel, not a
                // normalized colour component. We don't silently reinterpret
                // it as UNORM.
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

DecodedImage::DecodedImage(std::vector<std::byte> pixels, std::uint32_t width, std::uint32_t height,
                           VkFormat format) noexcept :
    pixels_(std::move(pixels)), width_(width), height_(height), format_(format) {}

auto DecodedImage::load_from_file(std::string_view path, ImageColourSpace colour_space) -> std::optional<DecodedImage> {
    auto const extension = lowercase_extension(path);

    if (extension == ".exr") {
        //
        // OpenEXR data is linear. Do not apply an sRGB Vulkan format to it.
        //
        if (colour_space == ImageColourSpace::srgb) {
            warn("EXR '{}' requested as sRGB; EXR texture data is loaded as linear", path);
        }

        return decode_exr(path);
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
