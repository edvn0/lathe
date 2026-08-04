#pragma once

#include <entt/entt.hpp>
#include <memory>
#include <unordered_map>

#include "physics.hxx"

class btBroadphaseInterface;
class btCollisionConfiguration;
class btCollisionShape;
class btConstraintSolver;
class btDefaultCollisionConfiguration;
class btDiscreteDynamicsWorld;
class btCollisionDispatcher;
class btRigidBody;

struct Transform;

// Owns a Bullet dynamics world and every btRigidBody/btCollisionShape
// created from it. Deliberately separate from the (copyable, cloned)
// RigidBody component -- Scene::play() deep-copies components via
// clone_registry(), which can't carry Bullet's non-copyable objects, so
// those live here instead, keyed by the entity they were built from.
class PhysicsWorld {
public:
    explicit PhysicsWorld(PhysicsWorldSettings const &settings);
    ~PhysicsWorld();

    PhysicsWorld(PhysicsWorld const &) = delete;
    auto operator=(PhysicsWorld const &) -> PhysicsWorld & = delete;
    PhysicsWorld(PhysicsWorld &&) = delete;
    auto operator=(PhysicsWorld &&) -> PhysicsWorld & = delete;

    // Creates a btRigidBody for every entity with a Transform + RigidBody in
    // the registry. Call once, after construction.
    auto populate_from(entt::registry &registry) -> void;

    auto add_body(entt::entity entity, Transform const &transform, RigidBody const &body) -> void;
    auto remove_body(entt::entity entity) -> void;

    // Advances the simulation by delta_time (internally substepped at a
    // fixed 1/120s step, capped at 8 substeps/frame) and writes the result
    // back into each simulated entity's Transform (position + rotation
    // only -- Transform::scale is renderer-only and is left untouched).
    auto step(entt::registry &registry, float delta_time) -> void;

private:
    struct Body {
        std::unique_ptr<btRigidBody> rigid_body;
        std::unique_ptr<btCollisionShape> shape;
    };

    std::unique_ptr<btCollisionConfiguration> collision_configuration_;
    std::unique_ptr<btCollisionDispatcher> dispatcher_;
    std::unique_ptr<btBroadphaseInterface> broadphase_;
    std::unique_ptr<btConstraintSolver> solver_;
    std::unique_ptr<btDiscreteDynamicsWorld> world_;

    std::unordered_map<entt::entity, Body> bodies_;
};
