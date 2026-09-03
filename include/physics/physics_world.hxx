#pragma once

#include <entt/entt.hpp>
#include <memory>
#include <optional>

#include <BS_thread_pool.hpp>

#include "physics/debug_lines.hxx"
#include "core/forward.hxx"
#include "physics/physics.hxx"
#include "physics/physics_components.hxx"
#include "core/transform.hxx"

#include <cstdint>
#include <limits>
#include <span>

struct RaycastHit {
    entt::entity entity{entt::null};
    glm::vec3 point{0.0f};
    glm::vec3 normal{0.0f};
    float distance{0.0f};
};

// Identifies one slot in PhysicsWorld's fixed terrain-collider pool -- see
// PhysicsWorld::reserve_terrain_collider. Not an entt::entity handle: these
// bodies are never entity-backed (no Components::RigidBody, no
// Components::PhysicsBody, no participation in step()'s Transform
// writeback), which is what lets a bounded, one-time set of slots absorb
// unlimited terrain chunk streaming without growing PhysicsWorld's arena
// per chunk load -- see the terrain streaming plan.
struct TerrainColliderHandle {
    std::uint32_t index = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] auto valid() const noexcept -> bool { return index != std::numeric_limits<std::uint32_t>::max(); }
};

struct TerrainColliderDesc {
    std::uint32_t samples_x = 65;
    std::uint32_t samples_z = 65;
    float cell_size_x = 1.0F;
    float cell_size_z = 1.0F;

    // Global vertical bounds -- must bound every chunk this slot will ever
    // hold, not just the first one, since btHeightfieldTerrainShape derives
    // its cached local AABB from these at construction and never inspects
    // the height data again (verified against Bullet's source -- see the
    // terrain streaming plan). This is what makes bind_terrain_collider's
    // in-place height rewrite safe with no AABB/accelerator invalidation.
    float min_height = -2.0F;
    float max_height = 2.0F;
};

class PhysicsWorld {
public:
    // registry must outlive this PhysicsWorld: the destructor removes each
    // remaining live body's Components::PhysicsBody from it, so that a body
    // never left dangling in the ECS after the btRigidBody/btCollisionShape
    // it points to have been destroyed (see ~Impl()).
    PhysicsWorld(PhysicsWorldSettings const &settings, BS::priority_thread_pool &thread_pool,
                entt::registry &registry);
    ~PhysicsWorld();

    PhysicsWorld(PhysicsWorld const &) = delete;
    auto operator=(PhysicsWorld const &) -> PhysicsWorld & = delete;
    PhysicsWorld(PhysicsWorld &&) = delete;
    auto operator=(PhysicsWorld &&) -> PhysicsWorld & = delete;

    auto populate_from(entt::registry &registry) -> void;
    auto add_body(entt::registry &registry, entt::entity entity, Components::Transform const &transform,
                  Components::RigidBody const &body) -> void;
    auto remove_body(entt::registry &registry, entt::entity entity) -> void;
    auto step(entt::registry &registry, float delta_time) -> void;

    // Pre-sizes the terrain-collider slot vector so reserve_terrain_collider
    // never triggers a reallocation mid-stream; purely a performance/
    // predictability hint, not required for correctness (moving a
    // std::vector<float> preserves its heap buffer's address, so the raw
    // pointer a slot's btHeightfieldTerrainShape holds into its heights
    // stays valid even if this vector does reallocate).
    auto reserve_terrain_colliders(std::uint32_t count) -> void;

    // Constructs one static heightfield body with `desc.samples_x *
    // samples_z` height samples, initially all zero and not yet added to
    // the world. Call bind_terrain_collider to make it live. The returned
    // handle is permanent for this PhysicsWorld's lifetime -- there is no
    // per-slot destroy, only unbind_terrain_collider (removes from the
    // world) and rebinding to new heights/centre (see the terrain streaming
    // plan for why a bounded, reusable pool avoids ArenaAllocator's lack of
    // a free-list).
    [[nodiscard]] auto reserve_terrain_collider(TerrainColliderDesc const &desc) -> TerrainColliderHandle;

    // Copies `heights` (must match the sample count `handle` was reserved
    // with) into the slot's permanently-owned buffer, moves it to `centre`,
    // and ensures it's in the world. Safe to call repeatedly on an already-
    // bound handle to move/rewrite it in place -- uses remove+add rather
    // than an in-place AABB update so Bullet's broadphase pair cache can't
    // retain stale contacts from the slot's previous position. Main-thread
    // only, like every other PhysicsWorld entry point.
    auto bind_terrain_collider(TerrainColliderHandle handle, glm::vec3 const &centre, std::span<float const> heights)
            -> void;

    // Removes `handle` from the world without destroying it -- it can be
    // bind_terrain_collider'd again later. No-op if not currently bound.
    auto unbind_terrain_collider(TerrainColliderHandle handle) -> void;

    auto attach_debug_drawer(IDebugLines &debug_lines) -> void;
    auto detach_debug_drawer() -> void;

    auto set_velocity(entt::registry const &registry, entt::entity entity, glm::vec3 const &linear_velocity) -> void;
    auto jump(entt::registry const &registry, entt::entity entity, float jump_velocity) -> void;
    auto is_grounded(entt::registry const &registry, entt::entity entity, float capsule_half_height,
                     float capsule_radius) const -> bool;

    [[nodiscard]] auto raycast(glm::vec3 const &from, glm::vec3 const &to) const -> std::optional<RaycastHit>;
    [[nodiscard]] auto raycast(glm::vec3 const &origin, glm::vec3 const &direction, float max_distance) const
            -> std::optional<RaycastHit>;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
