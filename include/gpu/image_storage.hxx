#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/error_context.hxx"

#include "core/fly_string.hxx"
#include "core/forward.hxx"
#include "core/holder.hxx"
#include "core/object_pool.hxx"
#include "gpu/buffer.hxx"
#include "gpu/compressed_texture.hxx"
#include "gpu/image.hxx"

enum class ImageDescriptorClass : std::uint8_t {
    sampled_2d,
    sampled_cube,
    sampled_2d_array,
    storage_2d,
    storage_2d_array,
};

struct ImageDescriptorRecord {
    VkImageView sampled_2d = VK_NULL_HANDLE;
    VkImageView sampled_cube = VK_NULL_HANDLE;
    VkImageView sampled_2d_array = VK_NULL_HANDLE;
    VkImageView storage_2d = VK_NULL_HANDLE;
    VkImageView storage_2d_array = VK_NULL_HANDLE;

    std::uint64_t revision = 0;
    bool occupied = false;
};

enum class DefaultImage : std::uint8_t {
    white = 0,
    black = 1,
    flat_normal = 2,
    metallic_roughness = 3,
    occlusion = 4,
    emissive = 5,
};

inline constexpr std::uint32_t default_image_count = 6;

[[nodiscard]]
constexpr auto default_image_handle(DefaultImage image) noexcept -> ImageHandle {
    return ImageHandle{
            .index = static_cast<std::uint32_t>(image),
            .generation = 1,
    };
}

enum class ImageStorageErrorType : std::uint8_t {
    invalid_argument,
    invalid_handle,
    protected_default,
    capacity_exceeded,
    image_error,
    device_error,
};

struct ImageStorageError {
    ImageStorageErrorType type = ImageStorageErrorType::invalid_argument;

    std::optional<ErrorCause> cause;
};

template<>
struct std::formatter<ImageStorageErrorType> : std::formatter<std::string_view> {
    constexpr auto format(ImageStorageErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case ImageStorageErrorType::invalid_argument:
                    return "invalid_argument";
                case ImageStorageErrorType::invalid_handle:
                    return "invalid_handle";
                case ImageStorageErrorType::protected_default:
                    return "protected_default";
                case ImageStorageErrorType::capacity_exceeded:
                    return "capacity_exceeded";
                case ImageStorageErrorType::image_error:
                    return "image_error";
                case ImageStorageErrorType::device_error:
                    return "device_error";
            }

            return "unknown_image_storage_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct ImageStorageCreateInfo {
    std::uint32_t capacity = 0;
    std::string_view debug_name = "image_storage";
};

struct ImageViewRegistration {
    VkImageView sampled_2d = VK_NULL_HANDLE;
    VkImageView storage_2d = VK_NULL_HANDLE;
};

// `revision` survives slot reuse and only increases, so GpuResourceTable never mistakes a reused slot for an
// unchanged one.
struct ImageSlotData {
    Image image{};

    VkImageView alias_sampled_2d = VK_NULL_HANDLE;
    VkImageView alias_storage_2d = VK_NULL_HANDLE;
    bool is_alias = false;

    bool protected_default = false;
    std::uint64_t revision = 1;
};

class ImageStorage {
public:
    ImageStorage() = default;
    ~ImageStorage();

    ImageStorage(ImageStorage const &) = delete;
    auto operator=(ImageStorage const &) -> ImageStorage & = delete;
    ImageStorage(ImageStorage &&other) noexcept;
    auto operator=(ImageStorage &&other) noexcept -> ImageStorage &;

    [[nodiscard]]
    static auto create(VulkanContext &context, ImageStorageCreateInfo const &create_info)
            -> std::expected<ImageStorage, ImageStorageError>;

    // Reserves a bindless slot for a view the caller already owns (e.g. another image's mip_layer_view()). The
    // source image must outlive this handle.
    [[nodiscard]]
    auto register_view(ImageViewRegistration const &registration) -> std::expected<ImageHandle, ImageStorageError>;
    [[nodiscard]]
    auto create_image(ImageCreateInfo const &create_info) -> std::expected<ImageHandle, ImageStorageError>;
    // `pixels` as for Image::create(): level 0, or every level with ImageMipSource::provided.
    [[nodiscard]]
    auto
    create_image(ImageCreateInfo const &create_info, std::span<const std::byte> pixels,
                 ImageMipSource mip_source = ImageMipSource::generate) -> std::expected<ImageHandle, ImageStorageError>;

    [[nodiscard]] auto create_image(ImageCreateInfo const &create_info, std::span<const std::byte> pixels,
                                    VkCommandBuffer command_buffer) -> std::expected<ImageHandle, ImageStorageError>;

    // Reserves a slot aliasing `fallback`'s views, usable right away. upgrade_pending_image() later installs the
    // real image under the same handle.
    [[nodiscard]]
    auto create_pending_image(ImageHandle fallback) -> std::expected<ImageHandle, ImageStorageError>;

    // Uploads a block-compressed, pre-mipped texture into `handle`'s slot under the same handle.
    //
    // Returns the staging buffer, which the caller must keep alive until `command_buffer` has executed.
    [[nodiscard]]
    auto upgrade_pending_image(ImageHandle handle, CompressedTexture const &texture, VkCommandBuffer command_buffer)
            -> std::expected<Buffer, ImageStorageError>;

    auto release_completed_uploads() -> void;

    [[nodiscard]]
    auto destroy_image(ImageHandle handle) -> std::expected<void, ImageStorageError>;

    // Records the six 1x1 default image uploads. Call before any shader samples them.
    [[nodiscard]]
    auto prepare_frame(VkCommandBuffer command_buffer) -> std::expected<void, ImageStorageError>;

    [[nodiscard]]
    auto get(ImageHandle handle) noexcept -> Image *;

    [[nodiscard]]
    auto get(ImageHandle handle) const noexcept -> Image const *;

    [[nodiscard]]
    auto contains(ImageHandle handle) const noexcept -> bool {
        return get(handle) != nullptr;
    }

    [[nodiscard]]
    auto white() const noexcept -> ImageHandle {
        return default_image_handle(DefaultImage::white);
    }

    [[nodiscard]]
    auto black() const noexcept -> ImageHandle {
        return default_image_handle(DefaultImage::black);
    }

    [[nodiscard]]
    auto flat_normal() const noexcept -> ImageHandle {
        return default_image_handle(DefaultImage::flat_normal);
    }

    [[nodiscard]]
    auto metallic_roughness() const noexcept -> ImageHandle {
        return default_image_handle(DefaultImage::metallic_roughness);
    }

    [[nodiscard]]
    auto occlusion() const noexcept -> ImageHandle {
        return default_image_handle(DefaultImage::occlusion);
    }

    [[nodiscard]]
    auto emissive() const noexcept -> ImageHandle {
        return default_image_handle(DefaultImage::emissive);
    }

    // A 1x1 black cube. GpuResourceTable writes its view into every cube binding slot that has no cube view of its
    // own, because the table is not PARTIALLY_BOUND. It is not a slot, so default_image_count is unaffected.
    [[nodiscard]]
    auto black_cube_view() const noexcept -> VkImageView {
        return black_cube_.descriptor_view(ImageDescriptorView::sampled_cube);
    }

    [[nodiscard]]
    auto size() const noexcept -> std::uint32_t {
        return slots_.size();
    }

    [[nodiscard]]
    auto capacity() const noexcept -> std::uint32_t {
        return slots_.capacity();
    }

    [[nodiscard]]
    auto descriptor_record(std::uint32_t index) const noexcept -> ImageDescriptorRecord;

    [[nodiscard]]
    auto descriptor_revision(std::uint32_t index) const noexcept -> std::uint64_t;

    [[nodiscard]]
    auto occupied(std::uint32_t index) const noexcept -> bool;

    auto destroy() noexcept -> void;

private:
    static constexpr auto bump_revision = [](ImageSlotData &slot) noexcept {
        ++slot.revision;

        if (slot.revision == 0) {
            slot.revision = 1;
        }
    };

    [[nodiscard]]
    auto create_default_images() -> std::expected<void, ImageStorageError>;

    [[nodiscard]]
    auto create_black_cube() -> std::expected<void, ImageStorageError>;

    auto record_black_cube_clear(VkCommandBuffer command_buffer) const noexcept -> void;

    VulkanContext *context_ = nullptr;

    ObjectPool<ImageSlotData> slots_;

    Buffer default_upload_buffer_{};

    Image black_cube_{};

    bool defaults_uploaded_ = false;

    std::vector<Buffer> pending_uploads_;

    FlyString debug_name_;
};

// Owns an ImageStorage slot; dropping it runs destroy_image(), which also covers register_view() aliases. Drop it only
// once the GPU is done with the image.
using ImageHolder = Holder<ImageStorage, ImageHandle, &ImageStorage::destroy_image>;

// create_image(), with the slot owned by the returned Holder.
[[nodiscard]]
inline auto create_held_image(ImageStorage &image_storage, ImageCreateInfo const &create_info)
        -> std::expected<ImageHolder, ImageStorageError> {
    return image_storage.create_image(create_info).transform([&image_storage](ImageHandle handle) {
        return ImageHolder{image_storage, handle};
    });
}

// register_view(), with the slot owned by the returned Holder. The source image must outlive the Holder.
[[nodiscard]]
inline auto register_held_view(ImageStorage &image_storage, ImageViewRegistration const &registration)
        -> std::expected<ImageHolder, ImageStorageError> {
    return image_storage.register_view(registration).transform([&image_storage](ImageHandle handle) {
        return ImageHolder{image_storage, handle};
    });
}
