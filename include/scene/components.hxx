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

    // Every handle here is retained by the Scene while the component exists, so a material made just for this
    // entity needs no name and is freed with the last entity using it. Change an existing override through
    // Scene::set_material_override(), which keeps those references balanced; emplace and remove work directly.
    struct MaterialOverride {
        // Replaces every submesh material when valid.
        MaterialHandle material{};

        // Replaces single model materials, winning over `material`. At most one entry per source.
        std::vector<MaterialSlotOverride> slots;

        // What `source` draws with on this entity, or an invalid handle if nothing replaces it.
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
