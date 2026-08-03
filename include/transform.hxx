#pragma once

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

struct Transform {
    glm::vec3 position{0.0F};
    glm::quat rotation{};
    glm::vec3 scale{1.0F};

    [[nodiscard]] auto matrix() const -> glm::mat4 {
        return glm::translate(glm::mat4{1.0F}, position) * glm::mat4_cast(rotation) *
               glm::scale(glm::mat4{1.0F}, scale);
    }
};
