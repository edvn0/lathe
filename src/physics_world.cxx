#include "physics_world.hxx"

#include <btBulletDynamicsCommon.h>
#include <glm/gtc/quaternion.hpp>

#include "transform.hxx"

namespace {
    auto to_bt(glm::vec3 const &v) -> btVector3 { return btVector3{v.x, v.y, v.z}; }

    auto to_glm(btVector3 const &v) -> glm::vec3 { return glm::vec3{v.x(), v.y(), v.z()}; }

    auto to_bt(glm::quat const &q) -> btQuaternion { return btQuaternion{q.x, q.y, q.z, q.w}; }

    auto to_glm(btQuaternion const &q) -> glm::quat { return glm::quat{q.w(), q.x(), q.y(), q.z()}; }
} // namespace

PhysicsWorld::PhysicsWorld(PhysicsWorldSettings const &settings) {
    collision_configuration_ = std::make_unique<btDefaultCollisionConfiguration>();
    dispatcher_ = std::make_unique<btCollisionDispatcher>(collision_configuration_.get());
    broadphase_ = std::make_unique<btDbvtBroadphase>();
    solver_ = std::make_unique<btSequentialImpulseConstraintSolver>();
    world_ = std::make_unique<btDiscreteDynamicsWorld>(dispatcher_.get(), broadphase_.get(), solver_.get(),
                                                       collision_configuration_.get());

    world_->setGravity(to_bt(settings.gravity));
}

PhysicsWorld::~PhysicsWorld() {
    // Bullet requires every body to be removed from the world before either
    // is destroyed -- do that explicitly rather than relying on member
    // destruction order.
    for (auto &[entity, body]: bodies_) {
        world_->removeRigidBody(body.rigid_body.get());
    }

    bodies_.clear();
}

auto PhysicsWorld::populate_from(entt::registry &registry) -> void {
    auto view = registry.view<Transform const, RigidBody const>();

    for (auto entity: view) {
        add_body(entity, view.get<Transform const>(entity), view.get<RigidBody const>(entity));
    }
}

auto PhysicsWorld::add_body(entt::entity entity, Transform const &transform, RigidBody const &body) -> void {
    auto shape = std::make_unique<btBoxShape>(to_bt(body.half_extents));

    btTransform start_transform;
    start_transform.setIdentity();
    start_transform.setOrigin(to_bt(transform.position));
    start_transform.setRotation(to_bt(transform.rotation));

    auto const mass = body.is_static ? 0.0F : body.mass;

    btVector3 local_inertia{0.0F, 0.0F, 0.0F};
    if (mass != 0.0F) {
        shape->calculateLocalInertia(mass, local_inertia);
    }

    btRigidBody::btRigidBodyConstructionInfo construction_info{mass, nullptr, shape.get(), local_inertia};
    construction_info.m_startWorldTransform = start_transform;
    construction_info.m_restitution = body.restitution;

    auto rigid_body = std::make_unique<btRigidBody>(construction_info);

    if (!body.is_static) {
        rigid_body->setLinearVelocity(to_bt(body.velocity));
    }

    world_->addRigidBody(rigid_body.get());

    bodies_.emplace(entity, Body{std::move(rigid_body), std::move(shape)});
}

auto PhysicsWorld::remove_body(entt::entity entity) -> void {
    auto it = bodies_.find(entity);

    if (it == bodies_.end()) {
        return;
    }

    world_->removeRigidBody(it->second.rigid_body.get());
    bodies_.erase(it);
}

auto PhysicsWorld::step(entt::registry &registry, float delta_time) -> void {
    constexpr float fixed_dt = 1.0F / 120.0F;
    constexpr int max_substeps = 8;

    world_->stepSimulation(delta_time, max_substeps, fixed_dt);

    for (auto &[entity, body]: bodies_) {
        if (!registry.valid(entity) || !registry.all_of<Transform>(entity)) {
            continue;
        }

        auto const &world_transform = body.rigid_body->getWorldTransform();
        auto &transform = registry.get<Transform>(entity);

        transform.position = to_glm(world_transform.getOrigin());
        transform.rotation = to_glm(world_transform.getRotation());
        // transform.scale is deliberately left untouched -- it's
        // renderer-only, not part of the physics state.
    }
}
