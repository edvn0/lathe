#pragma once

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

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

} // namespace Components
