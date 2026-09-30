#pragma once

#include <entt/entt.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
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

    // The entity's Model handle is ref-counted (loaded through a model cache), so removing the entity may call
    // destroy_model(). Not for entities sharing an unretained handle.
    struct StreamedModelTag {};
} // namespace Components
