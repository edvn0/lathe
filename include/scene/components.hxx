#pragma once

#include <atomic>
#include <cstdint>

#include <entt/entt.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "assets/material.hxx"
#include "assets/model.hxx"
#include "assets/script_handle.hxx"
#include "core/transform.hxx"
#include "physics/physics_components.hxx"
#include "scene/script_property.hxx"

namespace Components {
    struct PointLight {
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
        bool casts_shadows = false;
    };

    struct SpotLight {
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
        float inner_cone_degrees = 20.0F;
        float outer_cone_degrees = 30.0F;
        bool casts_shadows = false;
    };

    // Projects `material` (base colour, normal, metallic/roughness, occlusion and emissive, as on a mesh) onto the
    // geometry inside this entity's box. The box is the unit cube scaled by the entity's transform; the decal projects
    // along the entity's local -Y and its texture lies in local XZ.
    struct Decal {
        MaterialHandle material{};
        float opacity = 1.0F;
        // Surfaces tilted further than `end` from the decal's up axis get nothing; `start` and nearer get full strength.
        float angle_fade_start_degrees = 60.0F;
        float angle_fade_end_degrees = 80.0F;
        // Decals layer in ascending order, so a higher value draws over a lower one.
        std::int32_t sort_order = 0;
    };

    struct MaterialOverride {
        MaterialHandle material{};
        std::vector<MaterialSlotOverride> slots;

        [[nodiscard]]
        auto replacement_for(MaterialHandle source) const noexcept -> MaterialHandle {
            for (auto const &slot: slots) {
                if (slot.source == source) {
                    return slot.material;
                }
            }
            return material;
        }

        [[nodiscard]]
        auto empty() const noexcept -> bool {
            return !material.valid() && slots.empty();
        }
    };

    struct Model {
        ModelHandle model{};
    };

    // Draws every meshlet's normal cone as debug lines. extrude/shrink mirror the material's meshlet extrude/shrink
    // so the cones follow an exploded model.
    struct DebugMeshletCones {
        bool enabled = true;
        float extrude = 0.0F;
        float shrink = 0.0F;
    };

    [[nodiscard]] inline auto next_instanced_model_revision() noexcept -> std::uint64_t {
        static std::atomic<std::uint64_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    struct InstancedModel {
        ModelHandle model{};
        MaterialHandle material_override{};
        std::vector<glm::mat4> transforms;

        std::vector<std::uint32_t> palette_offsets;
        std::uint64_t revision = next_instanced_model_revision();

        auto touch() noexcept -> void { revision = next_instanced_model_revision(); }
    };

    enum class ParticleShape : std::uint8_t {
        // Every particle starts at the entity and flies off in a random direction.
        point,
        // Starts anywhere inside a ball of `shape_size` and flies outwards.
        sphere,
        // Starts anywhere inside a cube of half extent `shape_size` and flies along the entity's +Y axis (within
        // `cone_degrees`).
        box,
        // Starts at the entity and flies along its +Y axis, spread within `cone_degrees`.
        cone,
        // Starts on the outline of the entity's Text component, in the text's own space, and flies off in a random
        // direction. Without a Text component it behaves as `point`.
        text,
    };

    // A GPU particle emitter, simulated and drawn by the engine (rendering/particle_system.hxx). All data: the
    // entity's transform places it. A particle is a screen-facing square.
    struct ParticleEmitter {
        static constexpr std::uint32_t max_count = 1U << 20U;

        // Particle slots. When the rate outruns count / lifetime the oldest particles are overwritten.
        std::uint32_t count = 1024;
        // Particles spawned per second while emitting.
        float rate = 256.0F;
        float lifetime = 2.0F;
        glm::vec3 gravity{0.0F, -9.81F, 0.0F};
        ParticleShape shape = ParticleShape::cone;
        float shape_size = 0.5F;
        float cone_degrees = 25.0F;
        float speed = 3.0F;
        // Fraction of `speed` a particle's speed may differ by, either way.
        float speed_variance = 0.25F;
        // World units, from spawn to death.
        float size_start = 0.1F;
        float size_end = 0.02F;
        glm::vec4 colour_start{1.0F, 0.6F, 0.2F, 1.0F};
        // Alpha-blended over the scene (depth tested, no depth writes), so alpha fades the particle out over its life.
        glm::vec4 colour_end{1.0F, 0.1F, 0.0F, 0.0F};
        // Tints the particles (base colour times, emissive plus). Not reference counted: a released material is white.
        MaterialHandle material{};
        bool emitting = true;
    };

    enum class WaterBody : std::uint8_t {
        // An open sea: a surface that follows the camera out to the horizon, at the entity's height, clipped to the
        // `extent` rectangle about the entity (the water zone: where the sea may appear, so it stays out of the land).
        ocean,
        // A finite rectangle of `extent` metres (x, z) about the entity, at its height and turned by its yaw.
        lake,
    };

    // A body of water, drawn and animated by the engine (rendering/water_system.hxx, docs/water.md). All data: the
    // entity's transform places it (position and yaw; scale is ignored, `extent` sizes it). The system makes and owns
    // its mesh and material. Use ocean() / pool() for sensible starting values.
    struct Water {
        WaterBody body = WaterBody::lake;
        // Lake: the size (m) along x and z. Ocean: the water zone, the rectangle it is clipped to.
        glm::vec2 extent{10.0F, 10.0F};
        bool visible = true;

        // The colour light scatters back out of deep water, and how blurred the reflection is.
        glm::vec3 colour{0.08F, 0.35F, 0.4F};
        float roughness = 0.05F;

        // Beer-Lambert absorption per metre of water: red goes first, so deep water turns blue-green.
        glm::vec3 absorption{0.35F, 0.07F, 0.05F};
        // The water depth (m) over which the surface fades in from the shoreline.
        float shore_fade = 0.25F;

        // Waves: pattern frequency (1/m, larger is smaller waves), speed, how far (m) the surface moves up and down,
        // and how strongly the shading normal follows the short waves on top.
        float wave_scale = 0.45F;
        float wave_speed = 0.8F;
        float wave_height = 0.03F;
        float normal_strength = 0.5F;

        // Screen fraction (at 1 m) the refraction is offset by the waves.
        float refraction_strength = 0.03F;
        float foam = 0.8F;
        // Foam on the steepest crests (whitecaps); 0 on calm water.
        float whitecaps = 0.0F;
        float caustics = 1.0F;

        auto operator==(Water const &) const -> bool = default;

        // A calm, clear pool or lake of `size` metres.
        [[nodiscard]] static auto pool(glm::vec2 size = {10.0F, 10.0F}) -> Water {
            return Water{.body = WaterBody::lake, .extent = size};
        }

        // A rough, deep sea with whitecaps, out to the horizon within a zone of `zone` metres.
        [[nodiscard]] static auto ocean(glm::vec2 zone = {20000.0F, 20000.0F}) -> Water {
            return Water{
                    .body = WaterBody::ocean,
                    .extent = zone,
                    .colour = {0.02F, 0.16F, 0.22F},
                    .roughness = 0.08F,
                    .absorption = {0.45F, 0.12F, 0.09F},
                    .shore_fade = 0.4F,
                    .wave_scale = 0.13F,
                    .wave_speed = 1.0F,
                    .wave_height = 0.6F,
                    .normal_strength = 1.3F,
                    .refraction_strength = 0.03F,
                    .foam = 1.0F,
                    .whitecaps = 1.0F,
                    .caustics = 0.8F,
            };
        }
    };

    enum class PlanetColour : std::uint8_t {
        // The colour image (`colour_path`, an equirectangular RGB image such as NASA's Blue Marble).
        image,
        // Made from elevation and latitude: a depth gradient under the sea, beach, grass, rock, snow and polar ice.
        elevation,
    };

    // A planet at real scale (docs/globe.md), drawn by rendering/planet_system.hxx: a cube-sphere quadtree of terrain
    // chunks displaced by an equirectangular elevation map, an ocean at sea level shaded as water, and an optional
    // atmosphere. The entity's position is the planet's centre and its rotation turns it; scale is ignored. Real
    // scale needs a camera that sets CameraParams::origin (a floating origin): the system draws everything relative to
    // it, so no planet-sized number reaches the GPU.
    struct Planet {
        double radius = 6'371'000.0;
        // The two grey images of NASA's Blue Marble topography/bathymetry set (see planet::HeightmapSources). Until
        // both exist the planet is a smooth sphere.
        std::string elevation_path;
        std::string bathymetry_path;
        PlanetColour colour = PlanetColour::elevation;
        std::string colour_path;
        // Multiplies the relief (real relief is tiny next to the radius); 0 to about 20.
        float exaggeration = 1.0F;
        // Metres above the radius the ocean surface stands at.
        float sea_level = 0.0F;
        // The deepest quadtree level drawn for the terrain.
        std::uint8_t max_level = 11;
        bool ocean = true;
        // The ocean's look: the waves, colour and foam of a Water body (its `body` and `extent` are not used).
        Water ocean_water = deep_ocean();
        bool atmosphere = true;
        // Height (m) of the top of the atmosphere above sea level, and how bright its scattering is.
        float atmosphere_height = 100'000.0F;
        float atmosphere_intensity = 1.0F;
        // Tints each terrain chunk by its quadtree level.
        bool debug_lod = false;
        bool visible = true;

        auto operator==(Planet const &) const -> bool = default;

        // A sea seen from orbit: darker than Water::ocean, with the same waves.
        [[nodiscard]] static auto deep_ocean() -> Water {
            auto water = Water::ocean();
            water.colour = {0.01F, 0.05F, 0.09F};
            water.absorption = {0.45F, 0.09F, 0.05F};
            // Swells long enough to read from a few kilometres up.
            water.wave_scale = 0.03F;
            water.wave_height = 1.5F;
            water.normal_strength = 1.5F;
            return water;
        }
    };

    // Text drawn in the world (rendering/text_renderer.hxx): the entity's transform places it and it is depth-tested and
    // alpha-blended like any other transparent object. Before the transform, it reads left to right along +x with its
    // up on +y, for a viewer looking along +z. `emissive` multiplies the colour into HDR, so a value above 1 blooms like a
    // neon sign. A ParticleEmitter on the same entity with ParticleShape::text sheds particles from the letters.
    struct Text {
        std::string text;
        // The font to draw with: 0 is the engine's own, others are what the text renderer loaded.
        std::uint32_t font = 0;
        // The em size, in world units.
        float size = 1.0F;
        // Linear RGBA.
        glm::vec4 colour{1.0F};
        float emissive = 1.0F;
        // The entity's origin is the middle of the text, instead of the start of its first baseline.
        bool centred = true;
        bool visible = true;
    };

    struct Script {
        ScriptHandle script{};
    };

    // A behaviour (assets/scripts/behaviours/<script>.lua) on this entity. `values` are the properties the
    // entity sets itself; the rest take the script's defaults. LuaGame runs it while the game plays.
    struct Behaviour {
        std::string script;
        std::vector<ScriptPropertyValue> values;

        [[nodiscard]] auto find(std::string_view property) noexcept -> ScriptPropertyValue * {
            for (auto &entry: values) {
                if (entry.name == property) {
                    return &entry;
                }
            }

            return nullptr;
        }

        [[nodiscard]] auto find(std::string_view property) const noexcept -> ScriptPropertyValue const * {
            return const_cast<Behaviour *>(this)->find(property);
        }

        auto set(std::string_view property, ScriptValue value) -> void {
            if (auto *existing = find(property)) {
                existing->value = std::move(value);
            } else {
                values.push_back(ScriptPropertyValue{.name = std::string{property}, .value = std::move(value)});
            }
        }

        auto reset(std::string_view property) -> void {
            std::erase_if(values, [&](ScriptPropertyValue const &entry) { return entry.name == property; });
        }
    };

    struct PlayerTag {};

    struct Outlined {};

    // Marks an entity, and everything parented under it, as never moving once the scene runs. A static Model is baked
    // into a batch the renderer keeps on the GPU between frames (see StaticModelCache), so it costs nothing per frame
    // until something changes. The bake is rebuilt when a Transform is patched (registry.patch, as the editor does),
    // or a Model, MaterialOverride, Outlined, Parent or Static is added, changed or removed. Writing a static
    // entity's Transform directly, as physics and animation do, is not noticed: do not mark entities they move.
    struct Static {};

    struct BulletTag {};

    struct StreamedModelTag {};

    [[nodiscard]]
    inline auto model_released_by_model_destroy(entt::registry const &registry, entt::entity entity)
            -> std::optional<ModelHandle> {
        if (!registry.all_of<StreamedModelTag>(entity)) {
            return std::nullopt;
        }

        return registry.get<Model>(entity).model;
    }

    [[nodiscard]]
    inline auto model_released_by_tag_destroy(entt::registry const &registry, entt::entity entity)
            -> std::optional<ModelHandle> {
        auto const *model = registry.try_get<Model>(entity);

        if (model == nullptr) {
            return std::nullopt;
        }

        return model->model;
    }
}
