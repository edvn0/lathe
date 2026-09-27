#include "physics/physics_world.hxx"

#include <btBulletDynamicsCommon.h>
#include <glm/gtc/quaternion.hpp>

#include <BS_thread_pool.hpp>
#include <BulletCollision/CollisionDispatch/btCollisionDispatcherMt.h>
#include <BulletCollision/CollisionShapes/btCompoundShape.h>
#include <BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h>
#include <BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolverMt.h>
#include <BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h>
#include <LinearMath/btThreads.h>

#include "core/arena_allocator.hxx"
#include "physics/debug_lines.hxx"
#include "core/logger.hxx"
#include "core/memory_tracker.hxx"
#include "physics/physics_components.hxx"
#include "core/transform.hxx"

#include <algorithm>
#include <vector>

namespace {
    auto to_bt(glm::vec3 const &v) -> btVector3 { return btVector3{v.x, v.y, v.z}; }
    auto to_glm(btVector3 const &v) -> glm::vec3 { return glm::vec3{v.x(), v.y(), v.z()}; }
    auto to_bt(glm::quat const &q) -> btQuaternion { return btQuaternion{q.x, q.y, q.z, q.w}; }
    auto to_glm(btQuaternion const &q) -> glm::quat { return glm::quat{q.w(), q.x(), q.y(), q.z()}; }

    // Shapes live in the arena, so only destructors run. A compound doesn't destroy its children, so recurse.
    auto destroy_shape(btCollisionShape *shape) -> void {
        if (shape->getShapeType() == COMPOUND_SHAPE_PROXYTYPE) {
            auto *compound = static_cast<btCompoundShape *>(shape);

            for (int i = 0; i < compound->getNumChildShapes(); ++i) {
                destroy_shape(compound->getChildShape(i));
            }
        }

        shape->~btCollisionShape();
    }

    class ThreadPoolTaskScheduler final : public btITaskScheduler {
    public:
        explicit ThreadPoolTaskScheduler(BS::priority_thread_pool &pool) :
            btITaskScheduler{"bs_thread_pool"}, pool_{pool} {}

        // Bullet reserves thread index 0 for the caller and gives workers 1..N, and sizes per-thread scratch arrays
        // from this. Returning only the worker count makes a worker write past the end.
        auto getMaxNumThreads() const -> int override { return static_cast<int>(pool_.get_thread_count()) + 1; }
        auto getNumThreads() const -> int override { return static_cast<int>(pool_.get_thread_count()) + 1; }
        auto setNumThreads(int /*num_threads*/) -> void override {} // pool size is fixed

        auto parallelFor(int i_begin, int i_end, int grain_size, btIParallelForBody const &body) -> void override {
            if (i_end - i_begin <= grain_size) {
                body.forLoop(i_begin, i_end);
                return;
            }

            auto const num_chunks = std::max(1, (i_end - i_begin) / grain_size);

            // Keep the pool's per-call bookkeeping allocations out of the per-frame allocation stats.
            auto const untracked = MemoryTracker::UntrackedScope{};

            auto future = pool_.submit_blocks(
                    i_begin, i_end,
                    [&body](int const chunk_begin, int const chunk_end) { body.forLoop(chunk_begin, chunk_end); },
                    num_chunks);

            future.wait();
        }

        auto parallelSum(int i_begin, int i_end, int /*grain_size*/, btIParallelSumBody const &body)
                -> btScalar override {
            return body.sumLoop(i_begin, i_end); // Bullet rarely uses this; serial is fine.
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
        // Terrain colliders have no entity, so remove them before the entity loop below.
        for (auto &slot: terrain_colliders) {
            if (slot.active) {
                world->removeRigidBody(slot.body);
                slot.active = false;
            }

            slot.body->~btRigidBody();
            slot.shape->~btCollisionShape();
        }

        // Walk the world's own object array, always removing the last element so removeRigidBody()'s swap-and-pop
        // doesn't skip anything.
        auto &collision_objects = world->getCollisionObjectArray();

        for (auto i = collision_objects.size() - 1; i >= 0; --i) {
            auto *rigid_body = btRigidBody::upcast(collision_objects[i]);

            if (rigid_body == nullptr) {
                continue;
            }

            auto *shape = rigid_body->getCollisionShape();

            // Every remaining body came from add_body() and has a PhysicsBody to remove. No null check on the user
            // pointer: entity 0 bit-casts to null.
            auto const entity = static_cast<entt::entity>(reinterpret_cast<std::uintptr_t>(rigid_body->getUserPointer()));

            if (registry.valid(entity)) {
                registry.remove<Components::PhysicsBody>(entity);
            }

            world->removeRigidBody(rigid_body);

            rigid_body->~btRigidBody();
            destroy_shape(shape);
        }

        world->~btDiscreteDynamicsWorldMt();

        solver_pool->~btConstraintSolverPoolMt();
        solver_mt->~btSequentialImpulseConstraintSolverMt();
        broadphase->~btBroadphaseInterface();
        dispatcher->~btCollisionDispatcherMt();
        collision_configuration->~btDefaultCollisionConfiguration();

        // Setting a new scheduler calls the previous one's deactivate(), so clear the global before ours dies.
        if (btGetTaskScheduler() == &task_scheduler) {
            btSetTaskScheduler(nullptr);
        }
    }

    struct TerrainColliderSlot {
        // Bullet's heightfield keeps a raw pointer into this, so it is sized once and only overwritten in place.
        std::vector<float> heights;

        btHeightfieldTerrainShape *shape = nullptr;
        btRigidBody *body = nullptr;
        bool active = false; // in `world`
    };

    ArenaAllocator arena{512 * 1024};

    std::vector<TerrainColliderSlot> terrain_colliders;

    IDebugLines *debug_lines = nullptr;

    ThreadPoolTaskScheduler task_scheduler; // must outlive world, so declared first

    entt::registry &registry;

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

PhysicsWorld::~PhysicsWorld() = default;

auto PhysicsWorld::populate_from(entt::registry &registry) -> void {
    auto view = registry.view<Components::Transform const, Components::RigidBody const>();

    for (auto &&[entity, transform, rigid]: view.each()) {
        add_body(registry, entity, transform, rigid);
    }
}

auto PhysicsWorld::add_body(entt::registry &registry, entt::entity entity, Components::Transform const &transform,
                            Components::RigidBody const &body) -> void {
    btCollisionShape *shape = nullptr;

    switch (body.shape) {
        case Components::BodyShape::capsule:
            shape = impl_->arena.construct_with_base<btCapsuleShape, btCollisionShape>(body.capsule_radius,
                                                                                        body.capsule_height);
            break;
        case Components::BodyShape::heightfield: {
            auto const &heightfield = *body.heightfield;

            // Bullet keeps a raw pointer into heightfield.heights; the RigidBody owning it outlives the shape.
            shape = impl_->arena.construct_with_base<btHeightfieldTerrainShape, btCollisionShape>(
                    static_cast<int>(heightfield.width), static_cast<int>(heightfield.length),
                    heightfield.heights->data(), heightfield.min_height, heightfield.max_height,
                    /*upAxis=*/1, /*flipQuadEdges=*/false);
            shape->setLocalScaling(btVector3{heightfield.cell_size_x, 1.0F, heightfield.cell_size_z});
            break;
        }
        case Components::BodyShape::compound: {
            // Children are already axis-aligned in compound space, so their transforms are translation-only.
            auto *compound = impl_->arena.construct<btCompoundShape>();

            if (body.compound_boxes) {
                for (auto const &child: *body.compound_boxes) {
                    auto *box =
                            impl_->arena.construct_with_base<btBoxShape, btCollisionShape>(to_bt(child.half_extents));

                    btTransform child_transform;
                    child_transform.setIdentity();
                    child_transform.setOrigin(to_bt(child.local_centre));

                    compound->addChildShape(child_transform, box);
                }
            }

            shape = compound;
            break;
        }
        case Components::BodyShape::box:
        default:
            shape = impl_->arena.construct_with_base<btBoxShape, btCollisionShape>(to_bt(body.half_extents));
            break;
    }

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

    rigid_body->setSleepingThresholds(/*linear=*/0.8F, /*angular=*/1.0F);
    rigid_body->setDeactivationTime(0.8F);

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

    float const ray_length = capsule_half_height + capsule_radius + 0.1F;
    glm::vec3 const end = start - glm::vec3{0.0F, ray_length, 0.0F};

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
    destroy_shape(physics_body->shape);

    registry.remove<Components::PhysicsBody>(entity);
}

auto PhysicsWorld::reserve_terrain_colliders(std::uint32_t count) -> void {
    impl_->terrain_colliders.reserve(count);
}

auto PhysicsWorld::reserve_terrain_collider(TerrainColliderDesc const &desc) -> TerrainColliderHandle {
    Impl::TerrainColliderSlot slot;
    slot.heights.assign(static_cast<std::size_t>(desc.samples_x) * desc.samples_z, 0.0F);

    slot.shape = impl_->arena.construct<btHeightfieldTerrainShape>(
            static_cast<int>(desc.samples_x), static_cast<int>(desc.samples_z), slot.heights.data(),
            desc.min_height, desc.max_height, /*upAxis=*/1, /*flipQuadEdges=*/false);
    slot.shape->setLocalScaling(btVector3{desc.cell_size_x, 1.0F, desc.cell_size_z});

    btTransform start_transform;
    start_transform.setIdentity();

    btRigidBody::btRigidBodyConstructionInfo construction_info{0.0F, nullptr, slot.shape, btVector3{0.0F, 0.0F, 0.0F}};
    construction_info.m_startWorldTransform = start_transform;

    slot.body = impl_->arena.construct<btRigidBody>(construction_info);
    slot.body->setUserPointer(nullptr);

    auto const index = static_cast<std::uint32_t>(impl_->terrain_colliders.size());
    impl_->terrain_colliders.push_back(std::move(slot));

    return TerrainColliderHandle{.index = index};
}

auto PhysicsWorld::bind_terrain_collider(TerrainColliderHandle handle, glm::vec3 const &centre,
                                         std::span<float const> heights) -> void {
    if (!handle.valid() || handle.index >= impl_->terrain_colliders.size()) {
        return;
    }

    auto &slot = impl_->terrain_colliders[handle.index];

    if (heights.size() != slot.heights.size()) {
        return; // must match the slot's reserved sample count
    }

    std::ranges::copy(heights, slot.heights.begin());

    // Rebinding an active slot moves the same surface, so bodies resting on it move by the same delta.
    if (slot.active) {
        glm::vec3 const delta = centre - to_glm(slot.body->getWorldTransform().getOrigin());

        struct RestingBodyCollector final : public btCollisionWorld::ContactResultCallback {
            btCollisionObject const *self = nullptr;
            std::vector<btRigidBody *> bodies;

            auto addSingleResult(btManifoldPoint & /*contact_point*/, btCollisionObjectWrapper const *col_obj_0_wrap,
                                 int /*part_id_0*/, int /*index_0*/, btCollisionObjectWrapper const *col_obj_1_wrap,
                                 int /*part_id_1*/, int /*index_1*/) -> btScalar override {
                auto const *other = col_obj_0_wrap->getCollisionObject() == self ? col_obj_1_wrap->getCollisionObject()
                                                                                  : col_obj_0_wrap->getCollisionObject();

                if (auto *rigid_body = btRigidBody::upcast(other);
                    rigid_body != nullptr && !rigid_body->isStaticObject() &&
                    std::ranges::find(bodies, rigid_body) == bodies.end()) {
                    bodies.push_back(const_cast<btRigidBody *>(rigid_body));
                }

                return 0.0F;
            }
        } collector;
        collector.self = slot.body;

        impl_->world->contactTest(slot.body, collector);

        for (auto *rigid_body: collector.bodies) {
            auto body_transform = rigid_body->getWorldTransform();
            body_transform.setOrigin(body_transform.getOrigin() + to_bt(delta));
            rigid_body->setWorldTransform(body_transform);
            rigid_body->activate(true);
        }
    }

    btTransform transform;
    transform.setIdentity();
    transform.setOrigin(to_bt(centre));
    slot.body->setWorldTransform(transform);

    // Remove and re-add rather than teleport, so Bullet drops stale contacts and broadphase pairs.
    if (slot.active) {
        impl_->world->removeRigidBody(slot.body);
    }

    impl_->world->addRigidBody(slot.body);
    slot.active = true;
}

auto PhysicsWorld::unbind_terrain_collider(TerrainColliderHandle handle) -> void {
    if (!handle.valid() || handle.index >= impl_->terrain_colliders.size()) {
        return;
    }

    auto &slot = impl_->terrain_colliders[handle.index];

    if (!slot.active) {
        return;
    }

    impl_->world->removeRigidBody(slot.body);
    slot.active = false;
}

auto PhysicsWorld::attach_debug_drawer(IDebugLines &debug_lines) -> void {
    impl_->debug_lines = &debug_lines;
    impl_->world->setDebugDrawer(debug_lines.bullet_debug_draw());
}

auto PhysicsWorld::detach_debug_drawer() -> void {
    impl_->world->setDebugDrawer(nullptr);
    impl_->debug_lines->clear_lines();
    impl_->debug_lines = nullptr;
}

auto PhysicsWorld::step(entt::registry &registry, float delta_time) -> void {
    ZoneScopedNC("PhysicsStep", tracy::Color::Firebrick);

    constexpr float fixed_dt = 1.0F / 60.0F;
    constexpr int max_substeps = 2;
    impl_->world->stepSimulation(delta_time, max_substeps, fixed_dt);

    auto view = registry.view<Components::Transform, Components::PhysicsBody const>();

    for (auto &&[entity, transform, physics_body]: view.each()) {
        auto const &world_transform = physics_body.rigid_body->getWorldTransform();

        transform.position = to_glm(world_transform.getOrigin());
        transform.rotation = to_glm(world_transform.getRotation());
    }

    if (impl_->debug_lines != nullptr) {
        impl_->debug_lines->begin_frame();
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
