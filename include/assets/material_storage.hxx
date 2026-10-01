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

#include "assets/material.hxx"
#include "core/error_context.hxx"
#include "core/forward.hxx"
#include "core/object_pool.hxx"
#include "gpu/buffer.hxx"

#include "gpu/image.hxx"
#include "gpu/sampler.hxx"

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

    float wind_strength = 0.0F;

    // The farthest cascade this material casts into. Lower it for detail geometry whose shadow disappears at
    // distance, or use GpuMaterial::no_shadow_cascade to never cast.
    std::uint32_t max_shadow_cascade = shadow_cascade_count - 1;

    // Shade with one flat colour per meshlet instead of the base colour, to inspect how meshoptimizer split the
    // geometry. Lighting still applies so the surface stays readable. Instanced (non-meshlet) draws get one colour
    // per instance.
    bool debug_meshlet_colours = false;
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

struct MaterialSlotData {
    GpuMaterial material{};

    // The create info the material was last built from, so the editor can read back its handles.
    MaterialCreateInfo source{};

    std::uint32_t ref_count = 0;

    bool dirty = false;
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
    auto create_material(MaterialCreateInfo const &create_info) -> std::expected<MaterialHandle, MaterialStorageError>;

    [[nodiscard]]
    auto update_material(MaterialHandle handle, MaterialCreateInfo const &create_info)
            -> std::expected<void, MaterialStorageError>;

    [[nodiscard]]
    auto retain_material(MaterialHandle handle) -> std::expected<void, MaterialStorageError>;

    // Drops a reference; the last one frees the slot.
    [[nodiscard]]
    auto destroy_material(MaterialHandle handle) -> std::expected<void, MaterialStorageError>;

    [[nodiscard]]
    auto ref_count(MaterialHandle handle) const noexcept -> std::uint32_t;

    [[nodiscard]]
    auto get(MaterialHandle handle) const noexcept -> GpuMaterial const *;

    // The slot's MaterialSlotData::source, or nullptr for an invalid handle.
    [[nodiscard]]
    auto create_info(MaterialHandle handle) const noexcept -> MaterialCreateInfo const *;

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
        return slots_.capacity();
    }

    auto destroy() noexcept -> void;

private:
    VulkanContext *context_ = nullptr;

    ObjectPool<MaterialSlotData, 0> slots_;

    Buffer gpu_buffer_{};
    Buffer upload_buffer_{};
};
