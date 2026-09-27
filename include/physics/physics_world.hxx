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
    glm::vec3 point{0.0F};
    glm::vec3 normal{0.0F};
    float distance{0.0F};
};

// A slot in PhysicsWorld's fixed terrain-collider pool. These bodies have no entity, so a bounded set of
// slots serves all terrain streaming without growing the arena.
struct TerrainColliderHandle {
    std::uint32_t index = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] auto valid() const noexcept -> bool { return index != std::numeric_limits<std::uint32_t>::max(); }
};

struct TerrainColliderDesc {
    std::uint32_t samples_x = 65;
    std::uint32_t samples_z = 65;
    float cell_size_x = 1.0F;
    float cell_size_z = 1.0F;

    // Must bound every chunk the slot will ever hold: Bullet computes the heightfield's AABB from these once, which
    // is what makes rewriting heights in place safe.
    float min_height = -2.0F;
    float max_height = 2.0F;
};

class PhysicsWorld {
public:
    // `registry` must outlive this PhysicsWorld; the destructor removes remaining PhysicsBody components.
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

    // Pre-sizes the slot vector. Only a hint: reallocation keeps each heights buffer's address.
    auto reserve_terrain_colliders(std::uint32_t count) -> void;

    // Creates a static heightfield body of samples_x * samples_z zero heights, not yet in the world. The handle
    // is permanent; use bind/unbind to add, move or remove it.
    [[nodiscard]] auto reserve_terrain_collider(TerrainColliderDesc const &desc) -> TerrainColliderHandle;

    // Copies `heights` (same sample count as reserved) into the slot, moves it to `centre` and adds it to the
    // world. Rebinding re-adds the body so no stale contacts survive. Main thread only.
    auto bind_terrain_collider(TerrainColliderHandle handle, glm::vec3 const &centre, std::span<float const> heights)
            -> void;

    // Removes `handle` from the world so it can be rebound later. No-op if unbound.
    auto unbind_terrain_collider(TerrainColliderHandle handle) -> void;

    auto attach_debug_drawer(IDebugLines &debug_lines) -> void;
    auto detach_debug_drawer() -> void;

    auto set_velocity(entt::registry const &registry, entt::entity entity, glm::vec3 const &linear_velocity) -> void;
    auto jump(entt::registry const &registry, entt::entity entity, float jump_velocity) -> void;
    [[nodiscard]] auto is_grounded(entt::registry const &registry, entt::entity entity, float capsule_half_height,
                                   float capsule_radius) const -> bool;

    [[nodiscard]] auto raycast(glm::vec3 const &from, glm::vec3 const &to) const -> std::optional<RaycastHit>;
    [[nodiscard]] auto raycast(glm::vec3 const &origin, glm::vec3 const &direction, float max_distance) const
            -> std::optional<RaycastHit>;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
