#pragma once

#include <glm/vec3.hpp>


struct PhysicsWorldSettings {
    glm::vec3 gravity{0.0F, -9.81F, 0.0F};
    float ground_y = 0.0F;
};
