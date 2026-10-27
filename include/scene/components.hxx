#pragma once

#include <atomic>
#include <cstdint>

#include <entt/entt.hpp>
#include <glm/mat4x4.hpp>
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
