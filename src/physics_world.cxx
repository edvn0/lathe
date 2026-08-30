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
#include "memory_tracker.hxx"

namespace {
    auto to_bt(glm::vec3 const &v) -> btVector3 { return btVector3{v.x, v.y, v.z}; }
    auto to_glm(btVector3 const &v) -> glm::vec3 { return glm::vec3{v.x(), v.y(), v.z()}; }
    auto to_bt(glm::quat const &q) -> btQuaternion { return btQuaternion{q.x, q.y, q.z, q.w}; }
    auto to_glm(btQuaternion const &q) -> glm::quat { return glm::quat{q.w(), q.x(), q.y(), q.z()}; }

    class ThreadPoolTaskScheduler final : public btITaskScheduler {
    public:
        explicit ThreadPoolTaskScheduler(BS::priority_thread_pool &pool) :
            btITaskScheduler{"bs_thread_pool"}, pool_{pool} {}

        // Bullet indexes per-thread scratch arrays (e.g. btCollisionDispatcherMt's
        // m_batchManifoldsPtr) by btGetCurrentThreadIndex(), which reserves index 0 for
        // the calling thread and assigns worker threads 1..N. getNumThreads() must report
        // that total (workers + caller), or Bullet undersizes those arrays and a worker
        // thread writes one past the end -- silent heap corruption that crashes later,
        // somewhere unrelated.
        auto getMaxNumThreads() const -> int override { return static_cast<int>(pool_.get_thread_count()) + 1; }
        auto getNumThreads() const -> int override { return static_cast<int>(pool_.get_thread_count()) + 1; }
        auto setNumThreads(int /*num_threads*/) -> void override {} // pool size is fixed at construction

        auto parallelFor(int i_begin, int i_end, int grain_size, btIParallelForBody const &body) -> void override {
            if (i_end - i_begin <= grain_size) {
                body.forLoop(i_begin, i_end);
                return;
            }

            auto const num_chunks = std::max(1, (i_end - i_begin) / grain_size);

            // submit_blocks() itself heap-allocates task/promise bookkeeping on this
            // thread for every call -- real memory traffic, but it's thread-pool
            // plumbing, not engine state, and it fires every substep regardless of
            // what body actually does. Left tracked, it swamps the "this frame"
            // allocation count with noise unrelated to what the game is doing. Only
            // the submission itself is untracked; body.forLoop() below still runs
            // under the worker threads' own (tracked) state.
            auto const untracked = MemoryTracker::UntrackedScope{};

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
    Impl(PhysicsWorldSettings const &settings, BS::priority_thread_pool &thread_pool, entt::registry &reg) :
        task_scheduler{thread_pool}, registry{reg} {
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
        // No entity-keyed container of our own to walk here -- world's own
        // collision object array already lists every body add_body() put
        // in, so teardown reads that instead of duplicating it. Removing
        // the current last element each iteration (by always indexing from
        // the current end and counting down) stays valid across whatever
        // removeRigidBody()'s internal swap-and-pop does to the array.
        auto &collision_objects = world->getCollisionObjectArray();

        for (auto i = collision_objects.size() - 1; i >= 0; --i) {
            auto *rigid_body = btRigidBody::upcast(collision_objects[i]);

            if (rigid_body == nullptr) {
                continue;
            }

            auto *shape = rigid_body->getCollisionShape();

            // Every body still in the world at this point only ever got
            // here via add_body(), which always paired it with a
            // Components::PhysicsBody on this same entity -- remove that
            // now too, or it's left pointing at memory this destructor is
            // about to free.
            auto const entity = static_cast<entt::entity>(reinterpret_cast<std::uintptr_t>(rigid_body->getUserPointer()));

            if (registry.valid(entity)) {
                registry.remove<Components::PhysicsBody>(entity);
            }

            world->removeRigidBody(rigid_body);

            rigid_body->~btRigidBody();
            shape->~btCollisionShape();
        }

        world->~btDiscreteDynamicsWorldMt();

        solver_pool->~btConstraintSolverPoolMt();
        solver_mt->~btSequentialImpulseConstraintSolverMt();
        broadphase->~btBroadphaseInterface();
        dispatcher->~btCollisionDispatcherMt();
        collision_configuration->~btDefaultCollisionConfiguration();

        // btSetTaskScheduler(other) calls the *previous* scheduler's deactivate() before
        // switching. task_scheduler is about to be destroyed along with this Impl, so the
        // global must be cleared now -- otherwise the next PhysicsWorld's constructor calls
        // btSetTaskScheduler() and dereferences this now-dangling pointer.
        if (btGetTaskScheduler() == &task_scheduler) {
            btSetTaskScheduler(nullptr);
        }
    }

    ArenaAllocator arena{512 * 1024};

    debug_draw::DebugRenderer *debug_renderer = nullptr;

    ThreadPoolTaskScheduler task_scheduler; // must outlive world; declared before it, constructed first

    entt::registry &registry; // outlives this PhysicsWorld -- see the constructor's doc comment

    btDefaultCollisionConfiguration *collision_configuration{nullptr};
    btCollisionDispatcherMt *dispatcher{nullptr};
    btBroadphaseInterface *broadphase{nullptr};
    btConstraintSolverPoolMt *solver_pool{nullptr};
    btSequentialImpulseConstraintSolverMt *solver_mt{nullptr};
    btDiscreteDynamicsWorldMt *world{nullptr};
};

PhysicsWorld::PhysicsWorld(PhysicsWorldSettings const &settings, BS::priority_thread_pool &thread_pool,
                           entt::registry &registry) :
    impl_{std::make_unique<Impl>(settings, thread_pool, registry)} {}

PhysicsWorld::~PhysicsWorld() = default; // Impl is complete here, so the default destructor is fine

auto PhysicsWorld::populate_from(entt::registry &registry) -> void {
    auto view = registry.view<Components::Transform const, Components::RigidBody const>();

    for (auto &&[entity, transform, rigid]: view.each()) {
        add_body(registry, entity, transform, rigid);
    }
}

auto PhysicsWorld::add_body(entt::registry &registry, entt::entity entity, Components::Transform const &transform,
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

    registry.emplace<Components::PhysicsBody>(entity, Components::PhysicsBody{
                                                               .rigid_body = rigid_body,
                                                               .shape = shape,
                                                       });
}

auto PhysicsWorld::set_velocity(entt::registry const &registry, entt::entity entity,
                                glm::vec3 const &linear_velocity) -> void {
    auto const *physics_body = registry.try_get<Components::PhysicsBody const>(entity);
    if (physics_body == nullptr) {
        return;
    }

    auto current = physics_body->rigid_body->getLinearVelocity();
    physics_body->rigid_body->setLinearVelocity(btVector3{linear_velocity.x, current.y(), linear_velocity.z});
    physics_body->rigid_body->activate(true);
}

auto PhysicsWorld::jump(entt::registry const &registry, entt::entity entity, float jump_velocity) -> void {
    auto const *physics_body = registry.try_get<Components::PhysicsBody const>(entity);
    if (physics_body == nullptr) {
        return;
    }

    auto *body = physics_body->rigid_body;
    auto velocity = body->getLinearVelocity();
    velocity.setY(jump_velocity);

    body->setLinearVelocity(velocity);
    body->activate(true);
}

auto PhysicsWorld::is_grounded(entt::registry const &registry, entt::entity entity, float capsule_half_height,
                               float capsule_radius) const -> bool {
    auto const *physics_body = registry.try_get<Components::PhysicsBody const>(entity);
    if (physics_body == nullptr) {
        return false;
    }

    auto const &origin = physics_body->rigid_body->getWorldTransform().getOrigin();
    glm::vec3 const start = to_glm(origin);

    float const ray_length = capsule_half_height + capsule_radius + 0.1f;
    glm::vec3 const end = start - glm::vec3{0.0f, ray_length, 0.0f};

    auto hit = raycast(start, end);
    return hit.has_value() && hit->entity != entity;
}

auto PhysicsWorld::remove_body(entt::registry &registry, entt::entity entity) -> void {
    auto const *physics_body = registry.try_get<Components::PhysicsBody const>(entity);
    if (physics_body == nullptr) {
        return;
    }

    impl_->world->removeRigidBody(physics_body->rigid_body);

    physics_body->rigid_body->~btRigidBody();
    physics_body->shape->~btCollisionShape();

    registry.remove<Components::PhysicsBody>(entity);
}

auto PhysicsWorld::attach_debug_drawer(debug_draw::DebugRenderer &renderer) -> void {
    impl_->debug_renderer = &renderer;
    impl_->world->setDebugDrawer(renderer.bullet_debug_draw());
}

auto PhysicsWorld::detach_debug_drawer() -> void {
    impl_->world->setDebugDrawer(nullptr);
    impl_->debug_renderer->clear_lines();
    impl_->debug_renderer = nullptr;
}

auto PhysicsWorld::step(entt::registry &registry, float delta_time) -> void {
    ZoneScopedNC("PhysicsStep", tracy::Color::Firebrick);

    constexpr float fixed_dt = 1.0F / 60.0F;
    constexpr int max_substeps = 2;
    impl_->world->stepSimulation(delta_time, max_substeps, fixed_dt);

    // A view instead of walking our own entity-keyed container: entt
    // already guarantees every iterated entity is alive and has both
    // components, and packs Transform/PhysicsBody contiguously per its own
    // storage layout, so this no longer pays a hash lookup per body per
    // step just to find what a view already hands over directly.
    auto view = registry.view<Components::Transform, Components::PhysicsBody const>();

    for (auto &&[entity, transform, physics_body]: view.each()) {
        auto const &world_transform = physics_body.rigid_body->getWorldTransform();

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
