#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/error_context.hxx"
#include "core/forward.hxx"
#include "core/object_pool.hxx"
#include "gpu/sampler.hxx"

inline constexpr std::uint32_t default_sampler_count = 5;

struct SamplerCreateInfo {
    VkFilter mag_filter = VK_FILTER_LINEAR;
    VkFilter min_filter = VK_FILTER_LINEAR;

    VkSamplerMipmapMode mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    VkSamplerAddressMode address_mode_u = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    VkSamplerAddressMode address_mode_v = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    VkSamplerAddressMode address_mode_w = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    float mip_lod_bias = 0.0F;
    float max_anisotropy = 6.0F;

    VkCompareOp compare_op = VK_COMPARE_OP_NEVER;

    float min_lod = 0.0F;
    float max_lod = VK_LOD_CLAMP_NONE;

    SamplerClass sampler_class = SamplerClass::regular;

    std::string_view debug_name = "sampler";
};

enum class SamplerStorageErrorType : std::uint8_t {
    invalid_argument,
    invalid_handle,
    protected_default,
    capacity_exceeded,
    sampler_creation_failed,
};

struct SamplerStorageError {
    SamplerStorageErrorType type = SamplerStorageErrorType::invalid_argument;

    std::optional<ErrorContext> context;
};

template<>
struct std::formatter<SamplerStorageErrorType> : std::formatter<std::string_view> {
    constexpr auto format(SamplerStorageErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case SamplerStorageErrorType::invalid_argument:
                    return "invalid_argument";
                case SamplerStorageErrorType::invalid_handle:
                    return "invalid_handle";
                case SamplerStorageErrorType::protected_default:
                    return "protected_default";
                case SamplerStorageErrorType::capacity_exceeded:
                    return "capacity_exceeded";
                case SamplerStorageErrorType::sampler_creation_failed:
                    return "sampler_creation_failed";
            }

            return "unknown_sampler_storage_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct SamplerDescriptorRecord {
    VkSampler sampler = VK_NULL_HANDLE;

    std::uint64_t revision = 0;

    SamplerClass sampler_class = SamplerClass::regular;

    bool occupied = false;
};

// Backs SamplerHandle = Handle<SamplerSlotData> (see sampler.hxx) and is the
// payload type of SamplerStorage's ObjectPool. descriptor_revision is
// deliberately NOT reset by ObjectPool across a release()/allocate() cycle
// -- it's bumped monotonically by create_sampler()/destroy_sampler() so
// GpuResourceTable::prepare_frame's per-frame cached revisions never mistake
// a reused slot for an unchanged one.
struct SamplerSlotData {
    VkSampler sampler = VK_NULL_HANDLE;

    std::uint64_t descriptor_revision = 1;

    SamplerClass sampler_class = SamplerClass::regular;

    bool protected_default = false;
};

class SamplerStorage {
public:
    SamplerStorage() = default;
    ~SamplerStorage();

    SamplerStorage(SamplerStorage const &) = delete;

    auto operator=(SamplerStorage const &) -> SamplerStorage & = delete;

    SamplerStorage(SamplerStorage &&other) noexcept;

    auto operator=(SamplerStorage &&other) noexcept -> SamplerStorage &;

    [[nodiscard]]
    static auto create(VulkanContext &context, std::uint32_t capacity, std::string_view debug_name = "sampler_storage")
            -> std::expected<SamplerStorage, SamplerStorageError>;

    [[nodiscard]]
    auto create_sampler(SamplerCreateInfo const &create_info) -> std::expected<SamplerHandle, SamplerStorageError>;

    [[nodiscard]]
    auto descriptor_record(std::uint32_t index) const noexcept -> SamplerDescriptorRecord;

    [[nodiscard]]
    auto capacity() const noexcept -> std::uint32_t {
        return slots_.capacity();
    }

    [[nodiscard]]
    auto linear_repeat() const noexcept -> SamplerHandle {
        return {
                .index = 0,
                .generation = 1,
        };
    }

    [[nodiscard]]
    auto linear_clamp() const noexcept -> SamplerHandle {
        return {
                .index = 1,
                .generation = 1,
        };
    }

    [[nodiscard]]
    auto nearest_repeat() const noexcept -> SamplerHandle {
        return {
                .index = 2,
                .generation = 1,
        };
    }

    [[nodiscard]]
    auto nearest_clamp() const noexcept -> SamplerHandle {
        return {
                .index = 3,
                .generation = 1,
        };
    }

    [[nodiscard]]
    auto shadow_compare() const noexcept -> SamplerHandle {
        return {
                .index = 4,
                .generation = 1,
        };
    }

    [[nodiscard]]
    auto destroy_sampler(SamplerHandle handle) -> std::expected<void, SamplerStorageError>;

    auto destroy() noexcept -> void;

private:
    VulkanContext *context_ = nullptr;

    ObjectPool<SamplerSlotData> slots_;

    std::string debug_name_;
};
