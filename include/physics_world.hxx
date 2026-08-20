#pragma once

#include <entt/entt.hpp>
#include <memory>
#include <optional>

#include <BS_thread_pool.hpp>

#include "components.hxx"
#include "forward.hxx"
#include "physics.hxx"

struct RaycastHit {
    entt::entity entity{entt::null};
    glm::vec3 point{0.0f};
    glm::vec3 normal{0.0f};
    float distance{0.0f};
};

class PhysicsWorld {
public:
    PhysicsWorld(PhysicsWorldSettings const &settings, BS::priority_thread_pool &thread_pool);
    ~PhysicsWorld();

    PhysicsWorld(PhysicsWorld const &) = delete;
    auto operator=(PhysicsWorld const &) -> PhysicsWorld & = delete;
    PhysicsWorld(PhysicsWorld &&) = delete;
    auto operator=(PhysicsWorld &&) -> PhysicsWorld & = delete;

    auto populate_from(entt::registry &registry) -> void;
    auto add_body(entt::entity entity, Components::Transform const &transform, Components::RigidBody const &body)
            -> void;
    auto remove_body(entt::entity entity) -> void;
    auto step(entt::registry &registry, float delta_time) -> void;

    auto set_debug_drawer(debug_draw::DebugRenderer *) -> void;

    auto set_velocity(entt::entity entity, glm::vec3 const &linear_velocity) -> void;
    auto jump(entt::entity entity, float jump_velocity) -> void;
    auto is_grounded(entt::entity entity, float capsule_half_height, float capsule_radius) const -> bool;

    [[nodiscard]] auto raycast(glm::vec3 const &from, glm::vec3 const &to) const -> std::optional<RaycastHit>;
    [[nodiscard]] auto raycast(glm::vec3 const &origin, glm::vec3 const &direction, float max_distance) const
            -> std::optional<RaycastHit>;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
