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
