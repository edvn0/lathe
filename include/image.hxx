#pragma once

#include <span>
#include <volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <string_view>
#include <vector>

#include <vk_mem_alloc.h>

#include <optional>

#include "device_error.hxx"
#include "error_context.hxx"
#include "forward.hxx"

inline constexpr auto invalid_image_index = std::numeric_limits<std::uint32_t>::max();

struct ImageHandle {
    std::uint32_t index = invalid_image_index;

    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return generation != 0 && index != invalid_image_index;
    }

    auto operator==(ImageHandle const &) const -> bool = default;
};

enum class ImageDescriptorView : std::uint8_t {
    sampled_2d = 0,
    sampled_cube,
    sampled_2d_array,
    storage_2d,
    storage_2d_array,
    count,
};

using ImageDescriptorViewFlags = std::uint32_t;

[[nodiscard]]
constexpr auto image_descriptor_view_bit(ImageDescriptorView view) noexcept -> ImageDescriptorViewFlags {
    return ImageDescriptorViewFlags{1} << static_cast<std::uint32_t>(view);
}

[[nodiscard]]
constexpr auto has_image_descriptor_view(ImageDescriptorViewFlags flags, ImageDescriptorView view) noexcept -> bool {
    return (flags & image_descriptor_view_bit(view)) != 0;
}

enum class ImageErrorType : std::uint8_t {
    invalid_argument,
    image_creation_failed,
    view_creation_failed,
};

struct ImageError {
    ImageErrorType type = ImageErrorType::invalid_argument;

    std::optional<ErrorCause> cause;
};

struct ImageCreateInfo {
    VkExtent3D extent{
            .width = 1,
            .height = 1,
            .depth = 1,
    };

    VkFormat format = VK_FORMAT_UNDEFINED;

    VkImageUsageFlags usage = 0;
    VkImageAspectFlags aspect = 0;

    VkImageType image_type = VK_IMAGE_TYPE_2D;

    /*
     * The primary view used for attachments and ordinary
     * CPU-side access through Image::view().
     */
    VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;

    /*
     * Additional views exposed through the global GPU
     * resource table.
     */
    ImageDescriptorViewFlags descriptor_views = 0;

    VkImageCreateFlags flags = 0;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;

    std::uint32_t mip_levels = 1;
    std::uint32_t array_layers = 1;

    bool create_mip_layer_views = false;

    std::string_view debug_name = "image";
};

class Image {
public:
    Image() = default;
    ~Image();

    Image(Image const &) = delete;

    auto operator=(Image const &) -> Image & = delete;

    Image(Image &&other) noexcept;

    auto operator=(Image &&other) noexcept -> Image &;

    [[nodiscard]]
    static auto create(VulkanContext &context, ImageCreateInfo const &create_info) -> std::expected<Image, ImageError>;

    [[nodiscard]]
    static auto create(VulkanContext &context, ImageCreateInfo const &create_info, std::span<const std::byte> pixels)
            -> std::expected<Image, ImageError>;

    auto destroy() noexcept -> void;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return image_ != VK_NULL_HANDLE;
    }

    [[nodiscard]]
    auto image() const noexcept -> VkImage {
        return image_;
    }

    /*
     * Primary full-resource view. This remains suitable
     * for colour/depth attachments.
     */
    [[nodiscard]]
    auto view() const noexcept -> VkImageView {
        return view_;
    }

    [[nodiscard]]
    auto descriptor_view(ImageDescriptorView type) const noexcept -> VkImageView {
        auto const index = static_cast<std::size_t>(type);

        if (index >= descriptor_views_.size()) {
            return VK_NULL_HANDLE;
        }

        return descriptor_views_[index];
    }

    [[nodiscard]]
    auto has_descriptor_view(ImageDescriptorView type) const noexcept -> bool {
        return descriptor_view(type) != VK_NULL_HANDLE;
    }

    [[nodiscard]]
    auto mip_layer_view(std::uint32_t mip, std::uint32_t layer) const noexcept -> VkImageView {
        if (mip >= mip_levels_ || layer >= array_layers_ || mip_layer_views_.empty()) {
            return VK_NULL_HANDLE;
        }

        return mip_layer_views_[mip_layer_view_index(mip, layer)];
    }

    [[nodiscard]]
    auto has_mip_layer_views() const noexcept -> bool {
        return !mip_layer_views_.empty();
    }

    [[nodiscard]]
    auto allocation() const noexcept -> VmaAllocation {
        return allocation_;
    }

    [[nodiscard]]
    auto format() const noexcept -> VkFormat {
        return format_;
    }

    [[nodiscard]]
    auto extent() const noexcept -> VkExtent3D {
        return extent_;
    }

    [[nodiscard]]
    auto extent_2d() const noexcept -> VkExtent2D {
        return VkExtent2D{
                .width = extent_.width,
                .height = extent_.height,
        };
    }

    [[nodiscard]]
    auto mip_extent(std::uint32_t mip) const noexcept -> VkExtent2D {
        auto width = extent_.width;
        auto height = extent_.height;

        for (std::uint32_t i = 0; i < mip; ++i) {
            width = width > 1 ? width / 2 : 1;
            height = height > 1 ? height / 2 : 1;
        }

        return VkExtent2D{.width = width, .height = height};
    }

    [[nodiscard]]
    auto usage() const noexcept -> VkImageUsageFlags {
        return usage_;
    }

    [[nodiscard]]
    auto aspect() const noexcept -> VkImageAspectFlags {
        return aspect_;
    }

    [[nodiscard]]
    auto mip_levels() const noexcept -> std::uint32_t {
        return mip_levels_;
    }

    [[nodiscard]]
    auto array_layers() const noexcept -> std::uint32_t {
        return array_layers_;
    }

    [[nodiscard]]
    auto samples() const noexcept -> VkSampleCountFlagBits {
        return samples_;
    }

private:
    VulkanContext *context_ = nullptr;

    VkImage image_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;

    std::array<VkImageView, static_cast<std::size_t>(ImageDescriptorView::count)> descriptor_views_{};
    [[nodiscard]]
    auto mip_layer_view_index(std::uint32_t mip, std::uint32_t layer) const noexcept -> std::size_t {
        return static_cast<std::size_t>(mip) * array_layers_ + layer;
    }

    std::vector<VkImageView> mip_layer_views_;
    VmaAllocation allocation_ = VK_NULL_HANDLE;

    VmaAllocationInfo allocation_info_{};

    VkFormat format_ = VK_FORMAT_UNDEFINED;

    VkExtent3D extent_{};

    VkImageUsageFlags usage_ = 0;
    VkImageAspectFlags aspect_ = 0;

    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;

    std::uint32_t mip_levels_ = 0;
    std::uint32_t array_layers_ = 0;
};

template<>
struct std::formatter<ImageErrorType> : std::formatter<std::string_view> {
    constexpr auto format(ImageErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case ImageErrorType::invalid_argument:
                    return "invalid_argument";
                case ImageErrorType::image_creation_failed:
                    return "image_creation_failed";
                case ImageErrorType::view_creation_failed:
                    return "view_creation_failed";
            }

            return "unknown_image_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

class DecodedImage {
    struct StbiDeleter {
        auto operator()(unsigned char *pixels) const noexcept -> void;
    };

    std::unique_ptr<unsigned char, StbiDeleter> pixels_;
    int width_ = 0;
    int height_ = 0;

    DecodedImage(std::unique_ptr<unsigned char, StbiDeleter> pixels, int width, int height) noexcept;

public:
    static auto load_from_file(std::string_view path) -> std::optional<DecodedImage>;

    [[nodiscard]] auto span() const noexcept -> std::span<std::byte const>;
    [[nodiscard]] constexpr auto width() const noexcept -> std::uint32_t { return static_cast<std::uint32_t>(width_); }
    [[nodiscard]] constexpr auto height() const noexcept -> std::uint32_t {
        return static_cast<std::uint32_t>(height_);
    }
};
