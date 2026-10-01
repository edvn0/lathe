#pragma once

#include <entt/entt.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <optional>
#include <vector>

#include "assets/material.hxx"
#include "assets/model.hxx"
#include "physics/physics_components.hxx"
#include "core/transform.hxx"
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

    // The Scene holds a reference to every material here. Change an existing override through
    // Scene::set_material_override() so those references stay balanced.
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

    // Many instances of one model with one material_override, owned by a single entity. Transforms are world-space
    // and fixed at spawn; Parent doesn't apply to individual instances.
    struct InstancedModel {
        ModelHandle model{};
        MaterialHandle material_override{};
        std::vector<glm::mat4> transforms;
    };

    struct Script {
        ScriptHandle script{};
    };

    struct PlayerTag {};

    // Bullets spawned by BasicGame::shoot_bullet(). Their names aren't unique, so they need a tag.
    struct BulletTag {};

    // The entity owns one reference on its Model handle (ref-counted through the model cache), which Scene releases
    // when the entity, the Model or the tag goes. Not for entities sharing an unretained handle.
    struct StreamedModelTag {};

    // For on_destroy<Model> and on_destroy<StreamedModelTag>: the model reference to release, if this removal is the
    // one that ends the entity's ownership. entt raises on_destroy before erasing, one pool at a time, so whichever
    // of the pair goes first finds its partner still present and returns the handle; the second finds it gone.
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
} // namespace Components
