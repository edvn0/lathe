#pragma once

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

struct Transform {
    glm::vec3 position{0.0F};
    // Explicit identity (w, x, y, z) -- value-initializing with {} would
    // zero every component instead, which is a degenerate (zero-length)
    // quaternion rather than identity. glm::mat4_cast tolerates that by
    // coincidence of its formula, but anything that treats the quaternion
    // as a real rotation (e.g. Bullet's btMatrix3x3::setRotation, which
    // divides by its squared length) turns it into NaN/Inf.
    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    glm::vec3 scale{1.0F};

    [[nodiscard]] auto matrix() const -> glm::mat4 {
        return glm::translate(glm::mat4{1.0F}, position) * glm::mat4_cast(rotation) *
               glm::scale(glm::mat4{1.0F}, scale);
    }
};
