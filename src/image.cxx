#include "image.hxx"

#include <string>
#include <type_traits>
#include <utility>

#include "context.hxx"

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

    auto make_error(ImageErrorType type, VkResult result = VK_SUCCESS) noexcept -> ImageError {
        return ImageError{
                .type = type,
                .result = result,
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

    template<typename Handle>
    auto object_handle(Handle handle) noexcept -> std::uint64_t {
        if constexpr (std::is_pointer_v<Handle>) {
            return reinterpret_cast<std::uint64_t>(handle);
        } else {
            return static_cast<std::uint64_t>(handle);
        }
    }

    auto set_object_name(VkDevice device, VkObjectType object_type, std::uint64_t handle,
                         std::string_view name) noexcept -> void {
        if (device == VK_NULL_HANDLE || handle == 0 || name.empty() || vkSetDebugUtilsObjectNameEXT == nullptr) {
            return;
        }

        std::string null_terminated_name{name};

        VkDebugUtilsObjectNameInfoEXT const info{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                .pNext = nullptr,
                .objectType = object_type,
                .objectHandle = handle,
                .pObjectName = null_terminated_name.c_str(),
        };

        static_cast<void>(vkSetDebugUtilsObjectNameEXT(device, &info));
    }
} // namespace

Image::~Image() { destroy(); }

Image::Image(Image &&other) noexcept :
    context_(std::exchange(other.context_, nullptr)), image_(std::exchange(other.image_, VK_NULL_HANDLE)),
    view_(std::exchange(other.view_, VK_NULL_HANDLE)), descriptor_views_(std::exchange(other.descriptor_views_, {})),
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
    };

    Image image;
    image.context_ = &context;

    auto result = vmaCreateImage(context.allocator, &image_info, &allocation_info, &image.image_, &image.allocation_,
                                 &image.allocation_info_);

    if (result != VK_SUCCESS) {
        image.context_ = nullptr;

        return std::unexpected(make_error(ImageErrorType::image_creation_failed, result));
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

        return std::unexpected(make_error(ImageErrorType::view_creation_failed, result));
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

            return std::unexpected(make_error(ImageErrorType::view_creation_failed, result));
        }

        auto const descriptor_view_name =
                std::string{create_info.debug_name} + ".descriptor_view." + std::to_string(raw_type);

        set_object_name(context.device, VK_OBJECT_TYPE_IMAGE_VIEW,
                        object_handle(image.descriptor_views_[descriptor_view_index]), descriptor_view_name);
    }

    image.format_ = create_info.format;

    image.extent_ = create_info.extent;

    image.usage_ = create_info.usage;

    image.aspect_ = aspect;

    image.samples_ = create_info.samples;

    image.mip_levels_ = create_info.mip_levels;

    image.array_layers_ = create_info.array_layers;

    set_object_name(context.device, VK_OBJECT_TYPE_IMAGE, object_handle(image.image_), create_info.debug_name);

    auto const view_name = std::string{create_info.debug_name} + ".view";

    set_object_name(context.device, VK_OBJECT_TYPE_IMAGE_VIEW, object_handle(image.view_), view_name);

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
