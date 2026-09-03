#pragma once

#include <entt/entt.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

// Pure-data ECS components with no dependency on any other subsystem --
// usable from the lowest layers of the engine (physics needs Transform for
// its rigid-body sync, for instance). See components.hxx for the rest of
// the Components namespace, including the physics-specific components in
// physics_components.hxx.
namespace Components {
    struct Lifetime {
        float remaining_seconds{0.0F};
    };

    struct Transform {
        glm::vec3 position{0.0F};
        glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
        glm::vec3 scale{1.0F};

        [[nodiscard]] auto matrix() const -> glm::mat4 {
            return glm::translate(glm::mat4{1.0F}, position) * glm::mat4_cast(rotation) *
                   glm::scale(glm::mat4{1.0F}, scale);
        }
    };

    struct Parent {
        entt::entity entity{entt::null};
    };
} // namespace Components
