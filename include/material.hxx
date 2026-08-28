#pragma once

#include <cstdint>
#include <glm/glm.hpp>

#include "config.hxx"
#include "handle.hxx"

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
    float _pad1 = 0.0F;
    float _pad2 = 0.0F;

    static constexpr std::uint32_t no_shadow_cascade = ~0U;
};

static_assert(std::is_trivially_copyable_v<GpuMaterial>);
static_assert(sizeof(GpuMaterial) % 16 == 0);
static_assert(alignof(GpuMaterial) == 16);

// Full definition lives in material_storage.hxx, alongside the
// ObjectPool<MaterialSlotData, 0> it backs (see sampler.hxx's SamplerHandle
// for why an incomplete forward declaration is enough here).
//
// Sentinel = 0: slot 0 is MaterialStorage's permanently-reserved default
// material (see MaterialStorage::create()), so MaterialHandle{.index = 0}
// deliberately reads as invalid -- e.g. Renderer::submit_model/submit_mesh
// use .valid() to mean "was an explicit material_override given", and a
// bare MaterialHandle{} (index 0, generation 0) must mean "no override"
// even though slot 0 itself holds real, live default-material data.
struct MaterialSlotData;

using MaterialHandle = Handle<MaterialSlotData, 0>;
