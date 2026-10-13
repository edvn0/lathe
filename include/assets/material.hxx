#pragma once

#include <cstdint>
#include <glm/glm.hpp>

#include "core/config.hxx"
#include "core/handle.hxx"

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

    float wind_strength = 0.0F;

    std::uint32_t max_shadow_cascade = shadow_cascade_count - 1;

    std::uint32_t debug_meshlet_colours = 0;

    std::uint32_t flags = 0;

    float meshlet_extrude = 0.0F;
    float meshlet_shrink = 0.0F;
    float _pad1 = 0.0F;
    float _pad2 = 0.0F;

    static constexpr std::uint32_t no_shadow_cascade = ~0U;

    static constexpr std::uint32_t flag_double_sided = 1U << 0U;

    static constexpr std::uint32_t flag_alpha_to_coverage = 1U << 1U;

    static constexpr std::uint32_t flag_outlined = 1U << 2U;

    // Draws a procedural world-space grid over the surface (see grid_line in forward_geom.slang).
    static constexpr std::uint32_t flag_grid = 1U << 3U;
};

static_assert(std::is_trivially_copyable_v<GpuMaterial>);
static_assert(sizeof(GpuMaterial) % 16 == 0);
static_assert(alignof(GpuMaterial) == 16);

struct MaterialSlotData;

using MaterialHandle = Handle<MaterialSlotData, 0>;

struct MaterialSlotOverride {
    MaterialHandle source{};
    MaterialHandle material{};
};
