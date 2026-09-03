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

#include "gpu/buffer.hxx"
#include "gpu/compressed_texture.hxx"
#include "core/forward.hxx"
#include "gpu/image.hxx"
#include "core/object_pool.hxx"

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

enum class DefaultImage : std::uint32_t {
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

// Backs ImageHandle = Handle<ImageSlotData> (see image.hxx). revision is
// deliberately NOT reset by ObjectPool across a release()/allocate() cycle
// -- it's bumped monotonically so GpuResourceTable::prepare_frame's
// per-frame cached revisions never mistake a reused slot for an unchanged
// one (see SamplerSlotData's identical comment in sampler_storage.hxx).
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

    /*
     * Reserves a bindless slot backed by a view the caller already
     * owns (e.g. Image::mip_layer_view() of some other slot's
     * image), rather than creating a new VkImage. The caller is
     * responsible for keeping the source image alive at least as
     * long as this handle, and must destroy_image() this handle
     * before (or without ever) destroying the source image.
     */
    [[nodiscard]]
    auto register_view(ImageViewRegistration const &registration) -> std::expected<ImageHandle, ImageStorageError>;
    [[nodiscard]]
    auto create_image(ImageCreateInfo const &create_info) -> std::expected<ImageHandle, ImageStorageError>;
    [[nodiscard]]
    auto create_image(ImageCreateInfo const &create_info, std::span<const std::byte>)
            -> std::expected<ImageHandle, ImageStorageError>;

    [[nodiscard]] auto create_image(ImageCreateInfo const &create_info, std::span<const std::byte> pixels,
                                    VkCommandBuffer command_buffer) -> std::expected<ImageHandle, ImageStorageError>;

    /*
     * Reserves a slot immediately, aliased onto `fallback`'s own descriptor
     * views (normally one of this storage's default images), and returns a
     * handle the caller can bind into materials right away. The real GPU
     * image is installed later via upgrade_pending_image() -- every
     * consumer holding this handle transparently starts sampling the real
     * texture once that happens, with no handle churn.
     */
    [[nodiscard]]
    auto create_pending_image(ImageHandle fallback) -> std::expected<ImageHandle, ImageStorageError>;

    /*
     * Uploads a fully block-compressed, pre-mipped texture (see
     * texture_pipeline.hxx) and installs it into `handle`'s slot, replacing
     * whatever occupied it before (typically the fallback alias from
     * create_pending_image()). The handle's index/generation are unchanged.
     *
     * Returns the staging buffer the upload commands reference. It must be
     * kept alive (and eventually destroy()ed) by the caller until the GPU
     * has finished executing `command_buffer` -- this call only records
     * commands, it does not know when they complete.
     */
    [[nodiscard]]
    auto upgrade_pending_image(ImageHandle handle, CompressedTexture const &texture, VkCommandBuffer command_buffer)
            -> std::expected<Buffer, ImageStorageError>;

    auto release_completed_uploads() -> void;

    [[nodiscard]]
    auto destroy_image(ImageHandle handle) -> std::expected<void, ImageStorageError>;

    /*
     * Records the initial six 1x1 uploads once.
     *
     * Call from Renderer::prepare_frame() before any
     * shader samples the default images.
     */
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

    VulkanContext *context_ = nullptr;

    ObjectPool<ImageSlotData> slots_;

    Buffer default_upload_buffer_{};

    bool defaults_uploaded_ = false;

    std::vector<Buffer> pending_uploads_;

    std::string debug_name_;
};
