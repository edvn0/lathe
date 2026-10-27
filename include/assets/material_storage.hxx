#pragma once

#include <volk.h>

#include <glm/glm.hpp>

#include <array>
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

// How a water material (MaterialCreateInfo::water) looks; see docs/water.md. Games do not make these: the WaterSystem
// (rendering/water_system.hxx) builds the material from an entity's Components::Water. The base colour factor's rgb is the colour
// light scatters back out of deep water, and the roughness factor blurs the environment reflection.
struct WaterSettings {
    // Beer-Lambert absorption per metre of water: red goes first, so deep water turns blue-green.
    glm::vec3 absorption{0.45F, 0.09F, 0.06F};
    // The water depth (m) over which the surface fades in from the shoreline.
    float shore_fade = 0.35F;
    // Wave pattern frequency (1/m): larger is smaller ripples.
    float wave_scale = 0.35F;
    float wave_speed = 1.0F;
    float normal_strength = 0.6F;
    // How far (in screen fractions at 1 m) the refraction is offset by the waves.
    float refraction_strength = 0.04F;
    float foam = 1.0F;
    float caustics = 1.0F;
    // How far (m) the surface is displaced up and down by the waves (0 keeps it flat), and how much foam the steepest
    // crests break into.
    float wave_height = 0.1F;
    float whitecaps = 0.0F;
    // The world rectangle (min x, min z, max x, max z) the surface is clipped to.
    glm::vec4 zone{-1e9F, -1e9F, 1e9F, 1e9F};
    // A spherical surface (see GpuMaterial::water_sphere_centre).
    glm::vec4 sphere_centre{0.0F};
    glm::vec4 anchor{0.0F};
    glm::vec4 east{1.0F, 0.0F, 0.0F, 0.0F};
    glm::vec4 north{0.0F, 0.0F, 1.0F, 0.0F};
    std::array<glm::vec4, 3> phases{};

    auto operator==(WaterSettings const &) const noexcept -> bool = default;
};

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

    std::uint32_t max_shadow_cascade = shadow_cascade_count - 1;

    bool debug_meshlet_colours = false;

    float meshlet_extrude = 0.0F;
    float meshlet_shrink = 0.0F;

    bool double_sided = false;

    bool alpha_to_coverage = false;

    bool outlined = false;

    bool grid = false;

    // Shades the surface as water (needs AlphaMode::blend): refraction of the scene behind it, reflection, foam.
    bool water = false;
    // A planet's terrain (GpuMaterial::flag_planet); its frame is in water_settings' sphere fields.
    bool planet_surface = false;
    WaterSettings water_settings{};

    MaterialHandle far_material{};
    std::uint32_t far_material_lod = 2;
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

    [[nodiscard]]
    auto destroy_material(MaterialHandle handle) -> std::expected<void, MaterialStorageError>;

    [[nodiscard]]
    auto ref_count(MaterialHandle handle) const noexcept -> std::uint32_t;

    [[nodiscard]]
    auto get(MaterialHandle handle) const noexcept -> GpuMaterial const *;

    [[nodiscard]]
    auto create_info(MaterialHandle handle) const noexcept -> MaterialCreateInfo const *;

    [[nodiscard]]
    auto gpu_index(MaterialHandle handle) const noexcept -> std::uint32_t;

    [[nodiscard]]
    auto material_for_lod(MaterialHandle handle, std::uint32_t lod) const noexcept -> MaterialHandle;

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
