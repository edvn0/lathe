#pragma once

#include <entt/entt.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vector>

#include "assets/material.hxx" // MaterialHandle
#include "assets/model.hxx" // ModelHandle
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


    struct MaterialOverride {
        MaterialHandle material{};
    };

    struct Model {
        ModelHandle model{};
    };

    // Many instances of one model sharing one material_override, owned by a
    // single entity instead of one entity per instance (e.g. a grass field's
    // thousands of blades). Transforms are world-space and computed once at
    // spawn time -- unlike Model, there's no per-instance Transform, so
    // Parent/systems::get_world_transform don't apply to individual
    // instances, only (if ever needed) to the owning entity as a whole.
    struct InstancedModel {
        ModelHandle model{};
        MaterialHandle material_override{};
        std::vector<glm::mat4> transforms;
    };

    struct Script {
        ScriptHandle script{};
    };

    struct PlayerTag {};

    // Marks entities spawned by BasicGame::shoot_bullet() -- lets UI (the
    // Hierarchy widget) and any future bullet-specific systems identify them
    // by a typed marker instead of a name-string convention (bullet names
    // aren't even unique -- shoot_bullet() restarts its index at 0 every
    // call, so multiple live bullets can share the same GeneratedMeta name).
    struct BulletTag {};
} // namespace Components
