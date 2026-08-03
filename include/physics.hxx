#pragma once

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

// A dynamic (or static, if is_static) axis-aligned box body. Rotation is not
// simulated -- collision response only ever moves position/velocity, never
// the paired Transform's rotation.
struct RigidBody {
    glm::vec3 velocity{0.0F};
    glm::vec3 half_extents{0.5F};
    float restitution = 0.4F;
    bool is_static = false;
};

// Tags an entity (which must also have a Transform) as a point source of
// gravitational attraction: every non-static RigidBody accelerates toward
// it each step, inverse-square with distance, softened near the source so
// it doesn't blow up as bodies pass close by.
struct Attractor {
    float strength = 30.0F;
};

struct PhysicsWorldSettings {
    glm::vec3 gravity{0.0F, -9.81F, 0.0F};
    float ground_y = 0.0F;
    float attraction_softening = 0.75F;
    // Backstop against a dense, mutually-attracting cluster diverging into
    // Inf/NaN velocities under naive sequential impulse resolution. High
    // enough that it never engages during normal falling/bouncing.
    float max_speed = 60.0F;
};

// Advances every entity with a Transform+RigidBody by one fixed step:
// attraction toward any Attractor entities, gravity integration, a
// ground-plane clamp, then brute-force AABB-vs-AABB collision resolution
// between all dynamic body pairs.
auto simulate_physics(entt::registry &registry, PhysicsWorldSettings const &settings, float fixed_dt) -> void;
