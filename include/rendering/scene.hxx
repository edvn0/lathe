#pragma once

#include "core/forward.hxx"
#include "physics/physics.hxx"

#include <entt/entt.hpp>
#include <entt/entity/snapshot.hpp>

#include <glm/mat4x4.hpp>

#include <any>
#include <memory>
#include <utility>
#include <vector>

class PhysicsWorld;
class ScriptStorage;

namespace detail {
    template<typename NameType>
    class Entity;

    class ReadOnlyEntity;
    class ScriptEntity;
    class AttachedEntity;
} // namespace detail

class Scene {
public:
    PhysicsWorldSettings physics_settings{};

    explicit Scene(Renderer &);
    ~Scene();

    auto on_scene_start() -> void;
    auto on_scene_stop() -> void;

    auto step(float delta_time) -> void;

    auto get_registry() noexcept -> entt::registry & { return registry; }
    [[nodiscard]] auto get_registry() const noexcept -> entt::registry const & { return registry; }

    auto attach_debug_renderer(debug_draw::DebugRenderer &renderer) -> void;
    auto detach_debug_renderer() -> void;

    auto get_scripts() noexcept -> ScriptStorage &;
    [[nodiscard]] auto get_scripts() const noexcept -> ScriptStorage const &;

    // Removes the component if `material_override` is empty.
    auto set_material_override(entt::entity entity, Components::MaterialOverride material_override) -> void;

private:
    entt::registry registry;
    entt::sigh<void()> lights_changed_signal_;

public:
    // Declared after `registry` so it is destroyed first; its destructor uses the registry.
    std::unique_ptr<PhysicsWorld> physics_world;

private:
    // Scripts live on the Renderer, which editor_scene and runtime_scene share, so script handles survive the
    // play() clone.
    Renderer &renderer_;

    auto mark_lights_dirty(entt::registry &, entt::entity) -> void;
    auto on_transform_changed(entt::registry &reg, entt::entity entity) -> void;
    auto on_rigid_body_destroyed(entt::registry &, entt::entity entity) -> void;
    auto on_script_attached(entt::registry &, entt::entity) -> void;
    auto on_script_detached(entt::registry &, entt::entity) -> void;
    auto on_material_override_attached(entt::registry &, entt::entity) -> void;
    auto on_material_override_detached(entt::registry &, entt::entity) -> void;
    auto connect_light_signals() -> void;

    template<typename T>
    friend class detail::Entity;
    friend class detail::ReadOnlyEntity;
    friend class detail::ScriptEntity;
    friend class detail::AttachedEntity;
};

namespace detail {

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

template<typename... Components>
auto clone_registry(entt::registry const &src, entt::registry &dst) -> void {
    std::vector<std::any> values;

    detail::SnapshotOutputArchive out{&values};
    entt::snapshot snapshot{src};
    snapshot.get<entt::entity>(out);
    (snapshot.get<Components>(out), ...);

    detail::SnapshotInputArchive in{.values = &values};
    entt::snapshot_loader loader{dst};
    loader.get<entt::entity>(in);
    (loader.get<Components>(in), ...);
}

namespace systems {
    [[nodiscard]] auto get_world_transform(entt::registry const &registry, entt::entity, Components::Transform const &)
            -> glm::mat4;
    auto lifetime(entt::registry &registry, PhysicsWorld &physics, float dt) -> void;
    auto script_update(entt::registry &registry, Scene &scene, float dt) -> void;
} // namespace systems
