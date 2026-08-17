#pragma once

#include <glm/vec3.hpp>

// A dynamic (or static, if is_static) box body. This is a plain descriptor,
// not the simulated object itself -- PhysicsWorld builds a real btRigidBody
// from it. Kept as a value type (no pointers/handles) because Scene::play()
// deep-copies it via clone_registry(), which round-trips components through
// std::any.
struct RigidBody {
    glm::vec3 velocity{0.0F}; // initial velocity, applied when the body is created
    glm::vec3 half_extents{0.5F}; // btBoxShape half-extents
    float restitution = 0.4F;
    float mass = 1.0F; // ignored (treated as immovable) when is_static
    bool is_static = false;

    static auto from_model_bounds(auto &&bounds) -> RigidBody {
        auto &&[min, max] = std::tuple(std::get<0>(bounds), std::get<1>(bounds));
        return RigidBody{.half_extents = (max - min) * 0.5F};
    }
};

struct PhysicsWorldSettings {
    glm::vec3 gravity{0.0F, -9.81F, 0.0F};
    float ground_y = 0.0F;
};
