#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>

#include "assets/material.hxx"
#include "assets/texture_pipeline.hxx"
#include "core/config.hxx"
#include "core/transform.hxx"
#include "gpu/sampler.hxx"
#include "physics/physics.hxx"
#include "physics/physics_components.hxx"
#include "scene/components.hxx"
#include "scene/environment.hxx"
#include "serialisation/asset_id.hxx"
#include "serialisation/lbf_error.hxx"

inline constexpr std::uint32_t scene_no_index = std::numeric_limits<std::uint32_t>::max();

struct SceneAssetRef {
    AssetId id{};
    std::string source;
};

struct SceneTextureRef {
    AssetId id{};
    std::string source;
    TextureRole role = TextureRole::colour;
};

struct SceneMaterial {
    std::string name;

    glm::vec4 base_colour_factor{1.0F};
    glm::vec3 emissive_factor{0.0F};
    float emissive_strength = 1.0F;
    float metallic_factor = 1.0F;
    float roughness_factor = 1.0F;
    float normal_scale = 1.0F;
    float occlusion_strength = 1.0F;
    float alpha_cutoff = 0.5F;
    float wind_strength = 0.0F;
    std::uint32_t max_shadow_cascade = shadow_cascade_count - 1;
    AlphaMode alpha_mode = AlphaMode::opaque;
    DefaultSampler sampler = DefaultSampler::linear_repeat;
    bool debug_meshlet_colours = false;
    bool double_sided = false;
    bool alpha_to_coverage = false;

    std::uint32_t far_material = scene_no_index;
    std::uint32_t far_material_lod = 2;

    std::array<std::uint32_t, 5> textures{scene_no_index, scene_no_index, scene_no_index, scene_no_index,
                                          scene_no_index};
};

namespace scene_material_texture {
    inline constexpr std::size_t base_colour = 0;
    inline constexpr std::size_t normal = 1;
    inline constexpr std::size_t metallic_roughness = 2;
    inline constexpr std::size_t occlusion = 3;
    inline constexpr std::size_t emissive = 4;
}

enum class SceneEntityFlags : std::uint32_t {
    none = 0,
    generated_name = 1U << 0U,
    player = 1U << 1U,
    bullet = 1U << 2U,
    streamed_model = 1U << 3U,
};

[[nodiscard]] constexpr auto operator|(SceneEntityFlags left, SceneEntityFlags right) noexcept -> SceneEntityFlags {
    return static_cast<SceneEntityFlags>(static_cast<std::uint32_t>(left) | static_cast<std::uint32_t>(right));
}

[[nodiscard]] constexpr auto has_flag(SceneEntityFlags flags, SceneEntityFlags flag) noexcept -> bool {
    return (static_cast<std::uint32_t>(flags) & static_cast<std::uint32_t>(flag)) != 0;
}

struct SceneEntity {
    std::string name;
    std::optional<Components::Transform> transform;
    std::uint32_t parent = scene_no_index;
    SceneEntityFlags flags = SceneEntityFlags::none;
};

struct SceneModelComponent {
    std::uint32_t entity = 0;
    std::uint32_t model = 0;
};

struct SceneMaterialOverrideComponent {
    struct Slot {
        std::uint32_t source_slot = 0;
        std::uint32_t material = scene_no_index;
    };

    std::uint32_t entity = 0;
    std::uint32_t material = scene_no_index;
    std::vector<Slot> slots;
};

struct SceneInstancedModelComponent {
    std::uint32_t entity = 0;
    std::uint32_t model = 0;
    std::uint32_t material = scene_no_index;
    std::vector<glm::mat4> transforms;
};

struct ScenePointLightComponent {
    std::uint32_t entity = 0;
    Components::PointLight light;
};

struct SceneSpotLightComponent {
    std::uint32_t entity = 0;
    Components::SpotLight light;
};

struct SceneRigidBodyComponent {
    std::uint32_t entity = 0;
    Components::RigidBody body;
};

struct SceneScriptComponent {
    std::uint32_t entity = 0;
    std::string script;
};

struct SceneLifetimeComponent {
    std::uint32_t entity = 0;
    float remaining_seconds = 0.0F;
};

struct SceneDescription {
    PhysicsWorldSettings physics_settings{};

    SceneEnvironment environment{};
    AssetId environment_id{};

    std::vector<SceneAssetRef> models;
    std::vector<SceneTextureRef> textures;
    std::vector<SceneMaterial> materials;
    std::vector<SceneEntity> entities;

    std::vector<SceneModelComponent> model_components;
    std::vector<SceneMaterialOverrideComponent> material_overrides;
    std::vector<SceneInstancedModelComponent> instanced_models;
    std::vector<ScenePointLightComponent> point_lights;
    std::vector<SceneSpotLightComponent> spot_lights;
    std::vector<SceneRigidBodyComponent> rigid_bodies;
    std::vector<SceneScriptComponent> scripts;
    std::vector<SceneLifetimeComponent> lifetimes;
};

inline constexpr std::uint16_t scene_chunk_version = 1;

namespace scene_section {
    inline constexpr std::uint32_t settings = 1;
    inline constexpr std::uint32_t models = 2;
    inline constexpr std::uint32_t textures = 3;
    inline constexpr std::uint32_t materials = 4;
    inline constexpr std::uint32_t entities = 5;
    inline constexpr std::uint32_t model_components = 6;
    inline constexpr std::uint32_t material_overrides = 7;
    inline constexpr std::uint32_t instanced_models = 8;
    inline constexpr std::uint32_t point_lights = 9;
    inline constexpr std::uint32_t spot_lights = 10;
    inline constexpr std::uint32_t rigid_bodies = 11;
    inline constexpr std::uint32_t scripts = 12;
    inline constexpr std::uint32_t lifetimes = 13;
    inline constexpr std::uint32_t environment = 14;
}

inline constexpr std::uint16_t scene_section_version = 1;

inline constexpr std::uint16_t instanced_models_section_version = 2;

inline constexpr std::uint16_t materials_section_version = 2;

inline constexpr std::uint16_t rigid_bodies_section_version = 2;

inline constexpr std::uint16_t environment_section_version = 1;

[[nodiscard]]
auto encode_scene(SceneDescription const &scene) -> std::vector<std::byte>;

struct SceneDecodeReport {
    std::uint32_t skipped_sections = 0;
};

[[nodiscard]]
auto decode_scene(std::span<std::byte const> payload, std::uint16_t chunk_version = scene_chunk_version,
                  SceneDecodeReport *report = nullptr) -> std::expected<SceneDescription, LbfError>;

[[nodiscard]]
auto validate_scene(SceneDescription const &scene) -> std::expected<void, LbfError>;
