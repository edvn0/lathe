#pragma once

#include <entt/entt.hpp>
#include <memory>
#include <optional>

#include <BS_thread_pool.hpp>

#include "core/forward.hxx"
#include "core/transform.hxx"
#include "physics/capsule_sweep.hxx"
#include "physics/mesh_collider.hxx"
#include "physics/debug_lines.hxx"
#include "physics/physics.hxx"
#include "physics/physics_components.hxx"

#include <cstdint>
#include <limits>
#include <span>

struct RaycastHit {
    entt::entity entity{entt::null};
    glm::vec3 point{0.0F};
    glm::vec3 normal{0.0F};
    float distance{0.0F};
};

struct TerrainColliderHandle {
    std::uint32_t index = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] auto valid() const noexcept -> bool { return index != std::numeric_limits<std::uint32_t>::max(); }
};

struct TerrainColliderDesc {
    std::uint32_t samples_x = 65;
    std::uint32_t samples_z = 65;
    float cell_size_x = 1.0F;
    float cell_size_z = 1.0F;

    float min_height = -2.0F;
    float max_height = 2.0F;
};

class PhysicsWorld final : public CapsuleSweep {
public:
    PhysicsWorld(PhysicsWorldSettings const &settings, BS::priority_thread_pool &thread_pool, entt::registry &registry);
    ~PhysicsWorld() override;

    PhysicsWorld(PhysicsWorld const &) = delete;
    auto operator=(PhysicsWorld const &) -> PhysicsWorld & = delete;
    PhysicsWorld(PhysicsWorld &&) = delete;
    auto operator=(PhysicsWorld &&) -> PhysicsWorld & = delete;

    // Distinct for every world ever made, unlike its address, which a later world can reuse.
    [[nodiscard]] auto id() const noexcept -> std::uint64_t { return id_; }

    auto populate_from(entt::registry &registry) -> void;
    auto add_body(entt::registry &registry, entt::entity entity, Components::Transform const &transform,
                  Components::RigidBody const &body) -> void;
    auto remove_body(entt::registry &registry, entt::entity entity) -> void;

    // A static body that collides with `mesh`'s triangles, placed by `transform`. Reports `entity` on hits. The world
    // keeps the mesh alive and removes the body when it is destroyed.
    auto add_static_mesh(entt::entity entity, Components::Transform const &transform,
                         std::shared_ptr<MeshCollider const> mesh) -> void;
    auto step(entt::registry &registry, float delta_time) -> void;

    auto reserve_terrain_colliders(std::uint32_t count) -> void;

    [[nodiscard]] auto reserve_terrain_collider(TerrainColliderDesc const &desc) -> TerrainColliderHandle;

    auto bind_terrain_collider(TerrainColliderHandle handle, glm::vec3 const &centre, std::span<float const> heights)
            -> void;

    auto unbind_terrain_collider(TerrainColliderHandle handle) -> void;

    auto attach_debug_drawer(IDebugLines &debug_lines) -> void;
    auto detach_debug_drawer() -> void;

    auto set_velocity(entt::registry const &registry, entt::entity entity, glm::vec3 const &linear_velocity) -> void;

    auto apply_impulse(entt::registry const &registry, entt::entity entity, glm::vec3 const &impulse) -> void;

    auto set_transform(entt::registry const &registry, entt::entity entity, Components::Transform const &transform)
            -> void;
    auto jump(entt::registry const &registry, entt::entity entity, float jump_velocity) -> void;
    [[nodiscard]] auto is_grounded(entt::registry const &registry, entt::entity entity, float capsule_half_height,
                                   float capsule_radius) const -> bool;

    [[nodiscard]] auto raycast(glm::vec3 const &from, glm::vec3 const &to) const -> std::optional<RaycastHit>;
    [[nodiscard]] auto raycast(glm::vec3 const &origin, glm::vec3 const &direction, float max_distance) const
            -> std::optional<RaycastHit>;

    [[nodiscard]] auto sweep_capsule(glm::vec3 const &from, glm::vec3 const &to, float radius, float height,
                                     entt::entity ignore) const -> std::optional<SweepHit> override;

private:
    std::uint64_t id_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
