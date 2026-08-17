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
    auto view = registry.view<Components::Transform const, RigidBody const>();

    for (auto entity: view) {
        add_body(entity, view.get<Components::Transform const>(entity), view.get<RigidBody const>(entity));
    }
}

auto PhysicsWorld::add_body(entt::entity entity, Components::Transform const &transform, RigidBody const &body)
        -> void {
    auto *shape = arena_.construct<btBoxShape>(to_bt(body.half_extents));

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

    rigid_body->setSleepingThresholds(/*linear=*/0.8f, /*angular=*/1.0f);
    rigid_body->setDeactivationTime(0.8f); // Sleep faster (default is 2.0s)

    world_->addRigidBody(rigid_body);

    bodies_.emplace(entity, Body{rigid_body, shape});
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
