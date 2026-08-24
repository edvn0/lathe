#pragma once

#include "forward.hxx"
#include "physics.hxx"

#include <entt/entity/snapshot.hpp>
#include <entt/entt.hpp>

#include <glm/mat4x4.hpp>

#include <any>
#include <memory>
#include <utility>
#include <vector>

class PhysicsWorld;

namespace detail {
    template<typename NameType>
    class Entity;

    class ReadOnlyEntity;
} // namespace detail

class Scene {
public:
    PhysicsWorldSettings physics_settings{};
    std::unique_ptr<PhysicsWorld> physics_world;

    explicit Scene(Renderer &);
    ~Scene();

    auto on_scene_start() -> void;
    auto on_scene_stop() -> void;

    auto step(float delta_time) -> void;

    auto get_registry() noexcept -> entt::registry & { return registry; }
    auto get_registry() const noexcept -> entt::registry const & { return registry; }

    auto attach_debug_renderer(debug_draw::DebugRenderer &renderer) -> void;
    auto detach_debug_renderer() -> void;

private:
    entt::registry registry;
    entt::sigh<void()> lights_changed_signal_;

    auto mark_lights_dirty(entt::registry &, entt::entity) -> void;
    auto on_transform_changed(entt::registry &reg, entt::entity entity) -> void;
    auto connect_light_signals() -> void;

    template<typename T>
    friend class detail::Entity;
    friend class detail::ReadOnlyEntity;
};

namespace detail {

    // entt::basic_snapshot and entt::basic_snapshot_loader call their archive
    // with the exact same sequence of types (both are templated on the same
    // registry type), so a type-erased queue is enough to move data between them
    // without clone_registry having to hand-roll per-component copy code.
    struct SnapshotOutputArchive {
        std::vector<std::any> *values;

        template<typename T>
        auto operator()(T &&value) -> void {
            values->emplace_back(std::in_place_type<std::decay_t<T>>, std::forward<T>(value));
        }
    };

    struct SnapshotInputArchive {
        std::vector<std::any> const *values;
        std::size_t read_pos = 0;

        template<typename T>
        auto operator()(T &value) -> void {
            value = std::any_cast<T>((*values)[read_pos++]);
        }
    };

} // namespace detail

// Deep-copies every entity and its Components... from src into dst,
// preserving entity identity. dst must be an empty registry (a requirement
// of entt::basic_snapshot_loader).
template<typename... Components>
auto clone_registry(entt::registry const &src, entt::registry &dst) -> void {
    std::vector<std::any> values;

    detail::SnapshotOutputArchive out{&values};
    entt::snapshot snapshot{src};
    snapshot.get<entt::entity>(out);
    (snapshot.get<Components>(out), ...);

    detail::SnapshotInputArchive in{&values};
    entt::snapshot_loader loader{dst};
    loader.get<entt::entity>(in);
    (loader.get<Components>(in), ...);
}


namespace systems {
    [[nodiscard]] auto get_world_transform(entt::registry const &registry, entt::entity, const Components::Transform &)
            -> glm::mat4;
    auto lifetime(entt::registry &registry, PhysicsWorld &physics, float dt) -> void;
    auto player_movement(entt::registry &registry, PhysicsWorld &physics_world, entt::entity player_entity,
                         PlayerController &controller, PlayerCamera &camera, float delta_time) -> void;
} // namespace systems
