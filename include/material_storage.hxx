#pragma once

#include <volk.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <expected>
#include <format>
#include <string_view>
#include <type_traits>
#include <vector>

#include <optional>

#include "buffer.hxx"
#include "device_error.hxx"
#include "error_context.hxx"
#include "forward.hxx"
#include "image_storage.hxx"
#include "sampler_storage.hxx"

struct MaterialHandle {
    std::uint32_t index = 0;
    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return index != 0;
    }

    auto operator==(MaterialHandle const &) const -> bool = default;
};

enum class AlphaMode : std::uint32_t {
    opaque,
    mask,
    blend,
};

struct alignas(16) GpuMaterial {
    glm::vec4 base_colour_factor{1.0F};

    glm::vec3 emissive_factor{0.0F};
    float emissive_strength = 1.0F;

    float metallic_factor = 1.0F;
    float roughness_factor = 1.0F;
    float normal_scale = 1.0F;
    float occlusion_strength = 1.0F;

    std::uint32_t base_colour_texture = 0;
    std::uint32_t normal_texture = 0;
    std::uint32_t metallic_roughness_texture = 0;
    std::uint32_t occlusion_texture = 0;

    std::uint32_t emissive_texture = 0;
    std::uint32_t sampler_index = 0;
    AlphaMode alpha_mode = AlphaMode::opaque;
    float alpha_cutoff = 0.5F;
};

static_assert(std::is_trivially_copyable_v<GpuMaterial>);
static_assert(sizeof(GpuMaterial) % 16 == 0);
static_assert(alignof(GpuMaterial) == 16);


struct MaterialCreateInfo {
    glm::vec4 base_colour_factor{1.0F};
    glm::vec3 emissive_factor{0.0F};

    float emissive_strength = 1.0F;
    float metallic_factor = 1.0F;
    float roughness_factor = 1.0F;
    float normal_scale = 1.0F;
    float occlusion_strength = 1.0F;
    float alpha_cutoff = 0.5F;

    ImageHandle base_colour_texture{};
    ImageHandle normal_texture{};
    ImageHandle metallic_roughness_texture{};
    ImageHandle occlusion_texture{};
    ImageHandle emissive_texture{};

    SamplerHandle sampler{};

    AlphaMode alpha_mode = AlphaMode::opaque;
};

auto to_gpu_material(MaterialCreateInfo const &) noexcept -> GpuMaterial;

static constexpr auto default_material_index = 0U;

enum class MaterialStorageErrorType : std::uint8_t {
    invalid_argument,
    invalid_handle,
    capacity_exceeded,
    device_error,
};

struct MaterialStorageError {
    MaterialStorageErrorType type = MaterialStorageErrorType::invalid_argument;

    std::optional<ErrorCause> cause;
};

template<>
struct std::formatter<MaterialStorageErrorType> : std::formatter<std::string_view> {
    constexpr auto format(MaterialStorageErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case MaterialStorageErrorType::invalid_argument:
                    return "invalid_argument";
                case MaterialStorageErrorType::invalid_handle:
                    return "invalid_handle";
                case MaterialStorageErrorType::capacity_exceeded:
                    return "capacity_exceeded";
                case MaterialStorageErrorType::device_error:
                    return "device_error";
            }

            return "unknown_material_storage_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct MaterialStorageCreateInfo {
    std::uint32_t capacity = 0;
    std::string_view debug_name = "material_storage";
};

struct MaterialStorage {
    MaterialStorage() = default;

    MaterialStorage(MaterialStorage const &) = delete;
    auto operator=(MaterialStorage const &) -> MaterialStorage & = delete;

    MaterialStorage(MaterialStorage &&other) noexcept;
    auto operator=(MaterialStorage &&other) noexcept -> MaterialStorage &;

    [[nodiscard]]
    static auto create(VulkanContext &context, MaterialStorageCreateInfo const &create_info)
            -> std::expected<MaterialStorage, MaterialStorageError>;

    [[nodiscard]]
    auto create_material(GpuMaterial const &material) -> std::expected<MaterialHandle, MaterialStorageError>;

    [[nodiscard]]
    auto update_material(MaterialHandle handle, GpuMaterial const &material)
            -> std::expected<void, MaterialStorageError>;

    [[nodiscard]]
    auto destroy_material(MaterialHandle handle) -> std::expected<void, MaterialStorageError>;

    [[nodiscard]]
    auto get(MaterialHandle handle) const noexcept -> GpuMaterial const *;

    [[nodiscard]]
    auto gpu_index(MaterialHandle handle) const noexcept -> std::uint32_t;

    [[nodiscard]]
    auto prepare_frame(VkCommandBuffer command_buffer, std::uint32_t frame_index)
            -> std::expected<void, MaterialStorageError>;

    [[nodiscard]]
    auto device_address() const noexcept -> VkDeviceAddress {
        return gpu_buffer_.device_address;
    }

    [[nodiscard]]
    auto buffer() const noexcept -> VkBuffer {
        return gpu_buffer_.buffer;
    }

    [[nodiscard]]
    auto capacity() const noexcept -> std::uint32_t {
        return capacity_;
    }

    auto destroy() noexcept -> void;

private:
    struct Slot {
        GpuMaterial material{};

        std::uint32_t generation = 1;
        std::uint32_t next_free = 0;

        bool occupied = false;
        bool dirty = false;
    };

    [[nodiscard]]
    auto slot_for(MaterialHandle handle) noexcept -> Slot *;

    [[nodiscard]]
    auto slot_for(MaterialHandle handle) const noexcept -> Slot const *;

    VulkanContext *context_ = nullptr;

    std::vector<Slot> slots_;
    std::uint32_t free_head_ = 0;
    std::uint32_t capacity_ = 0;

    Buffer gpu_buffer_{};
    Buffer upload_buffer_{};
};
