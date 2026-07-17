#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "forward.hxx"

inline constexpr auto invalid_sampler_index = std::numeric_limits<std::uint32_t>::max();

struct SamplerHandle {
    std::uint32_t index = invalid_sampler_index;

    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return generation != 0 && index != invalid_sampler_index;
    }

    auto operator==(SamplerHandle const &) const -> bool = default;
};

enum class DefaultSampler : std::uint32_t {
    linear_repeat = 0,
    linear_clamp = 1,
    nearest_repeat = 2,
    nearest_clamp = 3,
    shadow_compare = 4,
};

inline constexpr std::uint32_t default_sampler_count = 5;

enum class SamplerClass : std::uint8_t {
    regular,
    comparison,
};

struct SamplerCreateInfo {
    VkFilter mag_filter = VK_FILTER_LINEAR;
    VkFilter min_filter = VK_FILTER_LINEAR;

    VkSamplerMipmapMode mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    VkSamplerAddressMode address_mode_u = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    VkSamplerAddressMode address_mode_v = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    VkSamplerAddressMode address_mode_w = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    float mip_lod_bias = 0.0F;
    float max_anisotropy = 1.0F;

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

    VkResult result = VK_SUCCESS;
};

struct SamplerDescriptorRecord {
    VkSampler sampler = VK_NULL_HANDLE;

    std::uint64_t revision = 0;

    SamplerClass sampler_class = SamplerClass::regular;

    bool occupied = false;
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
        return capacity_;
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
    struct Slot {
        VkSampler sampler = VK_NULL_HANDLE;

        std::uint32_t generation = 1;
        std::uint32_t next_free = 0;

        std::uint64_t descriptor_revision = 1;

        SamplerClass sampler_class = SamplerClass::regular;

        bool occupied = false;
        bool protected_default = false;
    };

    VulkanContext *context_ = nullptr;

    std::vector<Slot> slots_;

    std::uint32_t free_head_ = 0;
    std::uint32_t capacity_ = 0;

    std::string debug_name_;
};
