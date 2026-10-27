#pragma once

#include <array>
#include <cstddef>
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
    // Water (flag_water): foam amount and the strength of the caustics drawn on what lies under the surface.
    float water_foam = 0.0F;
    float water_caustics = 0.0F;

    // Water: Beer-Lambert absorption per metre of water (rgb), and the depth over which the shoreline fades in (w).
    glm::vec4 water_absorption{0.0F};
    // Water: wave pattern frequency (1/m), wave speed, normal strength, refraction offset strength.
    glm::vec4 water_waves{0.0F};
    // Water: wave height (m) the vertex stage displaces the surface by (x), whitecap amount (y); zw unused.
    glm::vec4 water_shape{0.0F};
    // Water: the rectangle (min x, min z, max x, max z) the vertex stage clips the surface to.
    glm::vec4 water_zone{-1e9F, -1e9F, 1e9F, 1e9F};
    // Water on a sphere (a planet's ocean, see PlanetSystem); all in camera-relative render space and rewritten every
    // frame. water_sphere_centre.w is 1 for a spherical surface, 0 for a flat one. Waves are laid out in the tangent
    // plane at water_anchor (axes water_east, water_north), and water_phases carries what the anchor's own position
    // adds to the waves' phases (8 waves, then the 4 warp terms), reduced on the CPU in double precision.
    glm::vec4 water_sphere_centre{0.0F};
    glm::vec4 water_anchor{0.0F};
    glm::vec4 water_east{1.0F, 0.0F, 0.0F, 0.0F};
    glm::vec4 water_north{0.0F, 0.0F, 1.0F, 0.0F};
    std::array<glm::vec4, 3> water_phases{};

    static constexpr std::uint32_t no_shadow_cascade = ~0U;

    static constexpr std::uint32_t flag_double_sided = 1U << 0U;

    static constexpr std::uint32_t flag_alpha_to_coverage = 1U << 1U;

    static constexpr std::uint32_t flag_outlined = 1U << 2U;

    // Draws a procedural world-space grid over the surface (see grid_line in forward_geom.slang).
    static constexpr std::uint32_t flag_grid = 1U << 3U;

    // Shaded as a water surface in the blended pass (see water.slang and docs/water.md). Needs AlphaMode::blend.
    static constexpr std::uint32_t flag_water = 1U << 4U;

    // A planet's terrain: the fragment shader takes the base colour's equirectangular coordinates from the direction to
    // the planet's centre (water_sphere_centre; planet axes x, y, z in water_east, water_anchor, water_north).
    static constexpr std::uint32_t flag_planet = 1U << 5U;
};

static_assert(std::is_trivially_copyable_v<GpuMaterial>);
static_assert(sizeof(GpuMaterial) % 16 == 0);
static_assert(alignof(GpuMaterial) == 16);
// Mirrors Material in assets/shaders/scene_types.slang.
static_assert(sizeof(GpuMaterial) == 288);
static_assert(offsetof(GpuMaterial, water_absorption) == 112);

struct MaterialSlotData;

using MaterialHandle = Handle<MaterialSlotData, 0>;

struct MaterialSlotOverride {
    MaterialHandle source{};
    MaterialHandle material{};
};
