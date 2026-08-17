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

#include "error_context.hxx"

#include "buffer.hxx"
#include "forward.hxx"
#include "image.hxx"

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

    [[nodiscard]]
    auto create_image(ImageCreateInfo const &create_info) -> std::expected<ImageHandle, ImageStorageError>;
    [[nodiscard]]
    auto create_image(ImageCreateInfo const &create_info, std::span<const std::byte>)
            -> std::expected<ImageHandle, ImageStorageError>;

    [[nodiscard]] auto create_image(ImageCreateInfo const &create_info, std::span<const std::byte> pixels,
                                    VkCommandBuffer command_buffer) -> std::expected<ImageHandle, ImageStorageError>;
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
        return size_;
    }

    [[nodiscard]]
    auto capacity() const noexcept -> std::uint32_t {
        return capacity_;
    }

    [[nodiscard]]
    auto descriptor_record(std::uint32_t index) const noexcept -> ImageDescriptorRecord;

    [[nodiscard]]
    auto descriptor_revision(std::uint32_t index) const noexcept -> std::uint64_t;

    [[nodiscard]]
    auto occupied(std::uint32_t index) const noexcept -> bool;

    auto destroy() noexcept -> void;

private:
    struct Slot {
        Image image{};

        std::uint32_t generation = 1;
        std::uint32_t next_free = 0;

        bool occupied = false;
        bool protected_default = false;
        std::uint64_t revision = 1;
    };
    static constexpr auto bump_revision = [](Slot &slot) noexcept {
        ++slot.revision;

        if (slot.revision == 0) {
            slot.revision = 1;
        }
    };

    [[nodiscard]]
    auto slot_for(ImageHandle handle) noexcept -> Slot *;

    [[nodiscard]]
    auto slot_for(ImageHandle handle) const noexcept -> Slot const *;

    [[nodiscard]]
    auto create_default_images() -> std::expected<void, ImageStorageError>;

    VulkanContext *context_ = nullptr;

    std::vector<Slot> slots_;

    Buffer default_upload_buffer_{};

    std::uint32_t free_head_ = 0;
    std::uint32_t capacity_ = 0;
    std::uint32_t size_ = 0;

    bool defaults_uploaded_ = false;

    std::vector<Buffer> pending_uploads_;

    std::string debug_name_;
};
