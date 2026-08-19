#include "physics_world.hxx"

#include <btBulletDynamicsCommon.h>
#include <glm/gtc/quaternion.hpp>

#include "components.hxx"

namespace {
    auto to_bt(glm::vec3 const &v) -> btVector3 { return btVector3{v.x, v.y, v.z}; }
    auto to_glm(btVector3 const &v) -> glm::vec3 { return glm::vec3{v.x(), v.y(), v.z()}; }
    auto to_bt(glm::quat const &q) -> btQuaternion { return btQuaternion{q.x, q.y, q.z, q.w}; }
    auto to_glm(btQuaternion const &q) -> glm::quat { return glm::quat{q.w(), q.x(), q.y(), q.z()}; }
} // namespace

PhysicsWorld::PhysicsWorld(PhysicsWorldSettings const &settings) {
    collision_configuration_ = arena_.construct<btDefaultCollisionConfiguration>();
    dispatcher_ = arena_.construct<btCollisionDispatcher>(collision_configuration_);
    broadphase_ = arena_.construct<btDbvtBroadphase>();
    solver_ = arena_.construct<btSequentialImpulseConstraintSolver>();
    world_ = arena_.construct<btDiscreteDynamicsWorld>(dispatcher_, broadphase_, solver_, collision_configuration_);

    world_->setGravity(to_bt(settings.gravity));
    world_->getSolverInfo().m_numIterations = 6;
    world_->getSolverInfo().m_solverMode |= SOLVER_USE_WARMSTARTING | SOLVER_SIMD | SOLVER_CACHE_FRIENDLY;
}

PhysicsWorld::~PhysicsWorld() {
    for (auto &[entity, body]: bodies_) {
        world_->removeRigidBody(body.rigid_body);
        body.rigid_body->~btRigidBody();
        body.shape->~btCollisionShape();
    }

    world_->~btDiscreteDynamicsWorld();

    solver_->~btConstraintSolver();
    broadphase_->~btBroadphaseInterface();
    dispatcher_->~btCollisionDispatcher();
    collision_configuration_->~btDefaultCollisionConfiguration();
}

auto PhysicsWorld::populate_from(entt::registry &registry) -> void {
    auto view = registry.view<Components::Transform const, Components::RigidBody const>();

    for (auto entity: view) {
        add_body(entity, view.get<Components::Transform const>(entity), view.get<Components::RigidBody const>(entity));
    }
}

auto PhysicsWorld::add_body(entt::entity entity, Components::Transform const &transform,
                            Components::RigidBody const &body) -> void {
    auto *shape = body.shape == Components::BodyShape::capsule
                          ? static_cast<btCollisionShape *>(
                                    arena_.construct<btCapsuleShape>(body.capsule_radius, body.capsule_height))
                          : static_cast<btCollisionShape *>(arena_.construct<btBoxShape>(to_bt(body.half_extents)));

    btTransform start_transform;
    start_transform.setIdentity();
    start_transform.setOrigin(to_bt(transform.position));
    start_transform.setRotation(to_bt(transform.rotation));

    auto const mass = body.is_static ? 0.0F : body.mass;

    btVector3 local_inertia{0.0F, 0.0F, 0.0F};
    if (mass != 0.0F) {
        shape->calculateLocalInertia(mass, local_inertia);
    }

    btRigidBody::btRigidBodyConstructionInfo construction_info{mass, nullptr, shape, local_inertia};
    construction_info.m_startWorldTransform = start_transform;
    construction_info.m_restitution = body.restitution;

    auto *rigid_body = arena_.construct<btRigidBody>(construction_info);
    rigid_body->setUserPointer(std::bit_cast<void *>(static_cast<std::uintptr_t>(entity)));

    if (!body.is_static) {
        rigid_body->setLinearVelocity(to_bt(body.velocity));
    }

    if (body.lock_rotation) {
        // Prevents collision torque from tipping a capsule controller over --
        // yaw is driven visually by PlayerController, not by Bullet.
        rigid_body->setAngularFactor(btVector3{0.0F, 0.0F, 0.0F});
        rigid_body->setActivationState(DISABLE_DEACTIVATION);
    }

    rigid_body->setSleepingThresholds(/*linear=*/0.8f, /*angular=*/1.0f);
    rigid_body->setDeactivationTime(0.8f); // Sleep faster (default is 2.0s)

    world_->addRigidBody(rigid_body);

    bodies_.emplace(entity, Body{rigid_body, shape});
}

auto PhysicsWorld::set_velocity(entt::entity entity, glm::vec3 const &linear_velocity) -> void {
    auto it = bodies_.find(entity);

    if (it == bodies_.end()) {
        return;
    }

    auto current = it->second.rigid_body->getLinearVelocity();
    it->second.rigid_body->setLinearVelocity(btVector3{linear_velocity.x, current.y(), linear_velocity.z});
    it->second.rigid_body->activate(true);
}

auto PhysicsWorld::jump(entt::entity entity, float jump_velocity) -> void {
    auto it = bodies_.find(entity);
    if (it == bodies_.end()) {
        return;
    }

    auto *body = it->second.rigid_body;
    auto velocity = body->getLinearVelocity();
    velocity.setY(jump_velocity);

    body->setLinearVelocity(velocity);
    body->activate(true);
}

auto PhysicsWorld::is_grounded(entt::entity entity, float capsule_half_height, float capsule_radius) const -> bool {
    auto it = bodies_.find(entity);
    if (it == bodies_.end()) {
        return false;
    }

    auto const &origin = it->second.rigid_body->getWorldTransform().getOrigin();
    glm::vec3 const start = to_glm(origin);

    float const ray_length = capsule_half_height + capsule_radius + 0.1f;
    glm::vec3 const end = start - glm::vec3{0.0f, ray_length, 0.0f};

    auto hit = raycast(start, end);
    return hit.has_value() && hit->entity != entity;
}

auto PhysicsWorld::remove_body(entt::entity entity) -> void {
    auto it = bodies_.find(entity);

    if (it == bodies_.end()) {
        return;
    }

    world_->removeRigidBody(it->second.rigid_body);

    it->second.rigid_body->~btRigidBody();
    it->second.shape->~btCollisionShape();

    bodies_.erase(it);
}

auto PhysicsWorld::step(entt::registry &registry, float delta_time) -> void {
    constexpr float fixed_dt = 1.0F / 60.0F;
    constexpr int max_substeps = 2;
    world_->stepSimulation(delta_time, max_substeps, fixed_dt);

    for (auto &[entity, body]: bodies_) {
        if (!registry.valid(entity) || !registry.all_of<Components::Transform>(entity)) {
            continue;
        }

        auto const &world_transform = body.rigid_body->getWorldTransform();
        auto &transform = registry.get<Components::Transform>(entity);

        transform.position = to_glm(world_transform.getOrigin());
        transform.rotation = to_glm(world_transform.getRotation());
    }
}

auto PhysicsWorld::raycast(glm::vec3 const &from, glm::vec3 const &to) const -> std::optional<RaycastHit> {
    btVector3 const bt_from = to_bt(from);
    btVector3 const bt_to = to_bt(to);

    btCollisionWorld::ClosestRayResultCallback ray_callback(bt_from, bt_to);
    world_->rayTest(bt_from, bt_to, ray_callback);

    if (!ray_callback.hasHit()) {
        return std::nullopt;
    }

    auto const *hit_body = btRigidBody::upcast(ray_callback.m_collisionObject);
    entt::entity hit_entity = entt::null;

    if (hit_body && hit_body->getUserPointer()) {
        hit_entity = static_cast<entt::entity>(reinterpret_cast<std::uintptr_t>(hit_body->getUserPointer()));
    }

    return RaycastHit{
            .entity = hit_entity,
            .point = to_glm(ray_callback.m_hitPointWorld),
            .normal = to_glm(ray_callback.m_hitNormalWorld),
            .distance = glm::distance(from, to_glm(ray_callback.m_hitPointWorld)),
    };
}

auto PhysicsWorld::raycast(glm::vec3 const &origin, glm::vec3 const &direction, float max_distance) const
        -> std::optional<RaycastHit> {
    glm::vec3 const to = origin + (glm::normalize(direction) * max_distance);

    btVector3 const bt_from = to_bt(origin);
    btVector3 const bt_to = to_bt(to);

    btCollisionWorld::ClosestRayResultCallback ray_callback(bt_from, bt_to);
    world_->rayTest(bt_from, bt_to, ray_callback);

    if (!ray_callback.hasHit()) {
        return std::nullopt;
    }

    auto const *hit_body = btRigidBody::upcast(ray_callback.m_collisionObject);
    entt::entity hit_entity = entt::null;

    if (hit_body && hit_body->getUserPointer()) {
        hit_entity = static_cast<entt::entity>(reinterpret_cast<std::uintptr_t>(hit_body->getUserPointer()));
    }

    auto const hit_point = to_glm(ray_callback.m_hitPointWorld);

    return RaycastHit{
            .entity = hit_entity,
            .point = hit_point,
            .normal = to_glm(ray_callback.m_hitNormalWorld),
            // Bullet's m_closestHitFraction represents [0.0, 1.0] along the segment (from -> to)
            .distance = ray_callback.m_closestHitFraction * max_distance,
    };
}
