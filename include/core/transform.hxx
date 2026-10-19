#pragma once

#include <entt/entt.hpp>
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

        // translate * rotate * scale, written out: the columns of the rotation scaled one by one, and the position in
        // the last. Composing the three matrices costs two full 4x4 products, and this runs for every entity every
        // frame.
        [[nodiscard]] auto matrix() const -> glm::mat4 {
            auto const axes = glm::mat3_cast(rotation);
            return glm::mat4{glm::vec4{axes[0] * scale.x, 0.0F}, glm::vec4{axes[1] * scale.y, 0.0F},
                             glm::vec4{axes[2] * scale.z, 0.0F}, glm::vec4{position, 1.0F}};
        }
    };

    struct Parent {
        entt::entity entity{entt::null};
    };
}
