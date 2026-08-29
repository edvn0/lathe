// physics_world.cxx
#include "physics_world.hxx"

#include <btBulletDynamicsCommon.h>
#include <glm/gtc/quaternion.hpp>

#include <BS_thread_pool.hpp>
#include <BulletCollision/CollisionDispatch/btCollisionDispatcherMt.h>
#include <BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolverMt.h>
#include <BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h>
#include <LinearMath/btThreads.h>

#include "arena_allocator.hxx"
#include "components.hxx"
#include "debug_renderer.hxx"
#include "logger.hxx"

namespace {
    auto to_bt(glm::vec3 const &v) -> btVector3 { return btVector3{v.x, v.y, v.z}; }
    auto to_glm(btVector3 const &v) -> glm::vec3 { return glm::vec3{v.x(), v.y(), v.z()}; }
    auto to_bt(glm::quat const &q) -> btQuaternion { return btQuaternion{q.x, q.y, q.z, q.w}; }
    auto to_glm(btQuaternion const &q) -> glm::quat { return glm::quat{q.w(), q.x(), q.y(), q.z()}; }

    class ThreadPoolTaskScheduler final : public btITaskScheduler {
    public:
        explicit ThreadPoolTaskScheduler(BS::priority_thread_pool &pool) :
            btITaskScheduler{"bs_thread_pool"}, pool_{pool} {}

        auto getMaxNumThreads() const -> int override { return static_cast<int>(pool_.get_thread_count()); }
        auto getNumThreads() const -> int override { return static_cast<int>(pool_.get_thread_count()); }
        auto setNumThreads(int /*num_threads*/) -> void override {} // pool size is fixed at construction

        auto parallelFor(int i_begin, int i_end, int grain_size, btIParallelForBody const &body) -> void override {
            if (i_end - i_begin <= grain_size) {
                body.forLoop(i_begin, i_end);
                return;
            }

            auto const num_chunks = std::max(1, (i_end - i_begin) / grain_size);

            auto future = pool_.submit_blocks(
                    i_begin, i_end,
                    [&body](int const chunk_begin, int const chunk_end) { body.forLoop(chunk_begin, chunk_end); },
                    num_chunks);

            future.wait();
        }

        auto parallelSum(int i_begin, int i_end, int /*grain_size*/, btIParallelSumBody const &body)
                -> btScalar override {
            return body.sumLoop(i_begin, i_end); // Bullet rarely hits this path — serial fallback is fine
        }

    private:
        BS::priority_thread_pool &pool_;
    };
} // namespace

struct PhysicsWorld::Impl {
    struct Body {
        btRigidBody *rigid_body;
        btCollisionShape *shape;
    };

    Impl(PhysicsWorldSettings const &settings, BS::priority_thread_pool &thread_pool) : task_scheduler{thread_pool} {
        btSetTaskScheduler(&task_scheduler);

        collision_configuration = arena.construct<btDefaultCollisionConfiguration>();
        dispatcher = arena.construct<btCollisionDispatcherMt>(collision_configuration);
        broadphase = arena.construct<btDbvtBroadphase>();

        auto const solver_count = static_cast<int>(thread_pool.get_thread_count());
        solver_pool = arena.construct<btConstraintSolverPoolMt>(solver_count);
        solver_mt = arena.construct<btSequentialImpulseConstraintSolverMt>();

        world = arena.construct<btDiscreteDynamicsWorldMt>(dispatcher, broadphase, solver_pool, solver_mt,
                                                           collision_configuration);

        world->setGravity(to_bt(settings.gravity));
        world->getSolverInfo().m_numIterations = 6;
        world->getSolverInfo().m_solverMode |= SOLVER_USE_WARMSTARTING | SOLVER_SIMD | SOLVER_CACHE_FRIENDLY;
    }

    ~Impl() {
        for (auto &[entity, body]: bodies) {
            world->removeRigidBody(body.rigid_body);
            body.rigid_body->~btRigidBody();
            body.shape->~btCollisionShape();
        }

        world->~btDiscreteDynamicsWorldMt();

        solver_pool->~btConstraintSolverPoolMt();
        solver_mt->~btSequentialImpulseConstraintSolverMt();
        broadphase->~btBroadphaseInterface();
        dispatcher->~btCollisionDispatcherMt();
        collision_configuration->~btDefaultCollisionConfiguration();
    }

    ArenaAllocator arena{512 * 1024};

    debug_draw::DebugRenderer *debug_renderer = nullptr;

    ThreadPoolTaskScheduler task_scheduler; // must outlive world; declared before it, constructed first

    btDefaultCollisionConfiguration *collision_configuration{nullptr};
    btCollisionDispatcherMt *dispatcher{nullptr};
    btBroadphaseInterface *broadphase{nullptr};
    btConstraintSolverPoolMt *solver_pool{nullptr};
    btSequentialImpulseConstraintSolverMt *solver_mt{nullptr};
    btDiscreteDynamicsWorldMt *world{nullptr};

    std::unordered_map<entt::entity, Body> bodies;
};

PhysicsWorld::PhysicsWorld(PhysicsWorldSettings const &settings, BS::priority_thread_pool &thread_pool) :
    impl_{std::make_unique<Impl>(settings, thread_pool)} {}

PhysicsWorld::~PhysicsWorld() = default; // Impl is complete here, so the default destructor is fine

auto PhysicsWorld::populate_from(entt::registry &registry) -> void {
    auto view = registry.view<Components::Transform const, Components::RigidBody const>();

    for (auto &&[entity, transform, rigid]: view.each()) {
        add_body(entity, transform, rigid);
    }
}

auto PhysicsWorld::add_body(entt::entity entity, Components::Transform const &transform,
                            Components::RigidBody const &body) -> void {
    auto *shape = body.shape == Components::BodyShape::capsule
                          ? impl_->arena.construct_with_base<btCapsuleShape, btCollisionShape>(body.capsule_radius,
                                                                                               body.capsule_height)
                          : impl_->arena.construct_with_base<btBoxShape, btCollisionShape>(to_bt(body.half_extents));

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

    auto *rigid_body = impl_->arena.construct<btRigidBody>(construction_info);
    rigid_body->setUserPointer(std::bit_cast<void *>(static_cast<std::uintptr_t>(entity)));

    if (!body.is_static) {
        rigid_body->setLinearVelocity(to_bt(body.velocity));
    }

    if (body.lock_rotation) {
        rigid_body->setAngularFactor(btVector3{0.0F, 0.0F, 0.0F});
        rigid_body->setActivationState(DISABLE_DEACTIVATION);
    }

    rigid_body->setSleepingThresholds(/*linear=*/0.8f, /*angular=*/1.0f);
    rigid_body->setDeactivationTime(0.8f);

    impl_->world->addRigidBody(rigid_body);

    impl_->bodies.emplace(entity, Impl::Body{rigid_body, shape});
}

auto PhysicsWorld::set_velocity(entt::entity entity, glm::vec3 const &linear_velocity) -> void {
    auto it = impl_->bodies.find(entity);
    if (it == impl_->bodies.end()) {
        return;
    }

    auto current = it->second.rigid_body->getLinearVelocity();
    it->second.rigid_body->setLinearVelocity(btVector3{linear_velocity.x, current.y(), linear_velocity.z});
    it->second.rigid_body->activate(true);
}

auto PhysicsWorld::jump(entt::entity entity, float jump_velocity) -> void {
    auto it = impl_->bodies.find(entity);
    if (it == impl_->bodies.end()) {
        return;
    }

    auto *body = it->second.rigid_body;
    auto velocity = body->getLinearVelocity();
    velocity.setY(jump_velocity);

    body->setLinearVelocity(velocity);
    body->activate(true);
}

auto PhysicsWorld::is_grounded(entt::entity entity, float capsule_half_height, float capsule_radius) const -> bool {
    auto it = impl_->bodies.find(entity);
    if (it == impl_->bodies.end()) {
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
    auto it = impl_->bodies.find(entity);
    if (it == impl_->bodies.end()) {
        return;
    }

    impl_->world->removeRigidBody(it->second.rigid_body);

    it->second.rigid_body->~btRigidBody();
    it->second.shape->~btCollisionShape();

    impl_->bodies.erase(it);
}

auto PhysicsWorld::attach_debug_drawer(debug_draw::DebugRenderer &renderer) -> void {
    impl_->debug_renderer = &renderer;
    impl_->world->setDebugDrawer(renderer.bullet_debug_draw());
}

auto PhysicsWorld::detach_debug_drawer() -> void {
    impl_->world->setDebugDrawer(nullptr);
    impl_->debug_renderer = nullptr;
}

auto PhysicsWorld::step(entt::registry &registry, float delta_time) -> void {
    ZoneScopedNC("PhysicsStep", tracy::Color::Firebrick);

    constexpr float fixed_dt = 1.0F / 60.0F;
    constexpr int max_substeps = 2;
    impl_->world->stepSimulation(delta_time, max_substeps, fixed_dt);

    for (auto &[entity, body]: impl_->bodies) {
        if (!registry.valid(entity) || !registry.all_of<Components::Transform>(entity)) {
            continue;
        }

        auto const &world_transform = body.rigid_body->getWorldTransform();
        auto &transform = registry.get<Components::Transform>(entity);

        transform.position = to_glm(world_transform.getOrigin());
        transform.rotation = to_glm(world_transform.getRotation());
    }

    if (impl_->debug_renderer != nullptr) {
        impl_->debug_renderer->begin_frame();
        impl_->world->debugDrawWorld();
    }
}

auto PhysicsWorld::raycast(glm::vec3 const &from, glm::vec3 const &to) const -> std::optional<RaycastHit> {
    btVector3 const bt_from = to_bt(from);
    btVector3 const bt_to = to_bt(to);

    btCollisionWorld::ClosestRayResultCallback ray_callback(bt_from, bt_to);
    impl_->world->rayTest(bt_from, bt_to, ray_callback);

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
    impl_->world->rayTest(bt_from, bt_to, ray_callback);

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
            .distance = ray_callback.m_closestHitFraction * max_distance,
    };
}
