#pragma once

#include <atomic>
#include <cstdint>

#include <entt/entt.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <optional>
#include <vector>

#include "assets/material.hxx"
#include "assets/model.hxx"
#include "core/transform.hxx"
#include "physics/physics_components.hxx"
#include "scene/script_handle.hxx"

namespace Components {
    struct PointLight {
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
    };

    struct SpotLight {
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
        float inner_cone_degrees = 20.0F;
        float outer_cone_degrees = 30.0F;
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

    struct Script {
        ScriptHandle script{};
    };

    struct PlayerTag {};

    struct Outlined {};

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
