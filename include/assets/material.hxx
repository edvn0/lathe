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

    // Non-zero replaces the base colour with a per-meshlet hash colour; see debug_meshlet_colour() in
    // forward_geom.slang.
    std::uint32_t debug_meshlet_colours = 0;

    // material_flag_* bits. Mirrors Material::flags in scene_types.slang.
    std::uint32_t flags = 0;

    static constexpr std::uint32_t no_shadow_cascade = ~0U;

    // Drawn without back-face culling; back faces shade with the normal flipped.
    static constexpr std::uint32_t flag_double_sided = 1U << 0U;

    // Mask materials under MSAA: the depth prepass turns alpha into sample coverage instead of a hard cutoff, so
    // alpha-tested edges are antialiased.
    static constexpr std::uint32_t flag_alpha_to_coverage = 1U << 1U;

    // The forward pass marks the pixels this material covers in the outline mask (see Renderer::outline_settings).
    static constexpr std::uint32_t flag_outlined = 1U << 2U;
};

static_assert(std::is_trivially_copyable_v<GpuMaterial>);
static_assert(sizeof(GpuMaterial) % 16 == 0);
static_assert(alignof(GpuMaterial) == 16);

// Defined in material_storage.hxx.
//
// Sentinel = 0: slot 0 holds the default material, but MaterialHandle{} must read as "no override".
struct MaterialSlotData;

using MaterialHandle = Handle<MaterialSlotData, 0>;

struct MaterialSlotOverride {
    MaterialHandle source{};
    MaterialHandle material{};
};
