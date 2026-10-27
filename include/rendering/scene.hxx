#pragma once

#include "core/forward.hxx"
#include "physics/physics.hxx"
#include "scene/environment.hxx"

#include <entt/entt.hpp>

#include <glm/mat4x4.hpp>

#include <any>
#include <cstdint>
#include <memory>
#include <unordered_map>
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
}

class Scene {
public:
    PhysicsWorldSettings physics_settings{};

    SceneEnvironment environment = new_scene_environment();

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

    auto set_material_override(entt::entity entity, Components::MaterialOverride material_override) -> void;

    [[nodiscard]] auto hierarchy_revision() const noexcept -> std::uint64_t { return hierarchy_revision_; }

    // Changes whenever something a static batch is baked from changes (see Components::Static). Unique across scenes.
    [[nodiscard]] auto static_revision() const noexcept -> std::uint64_t { return static_revision_; }
    auto mark_static_dirty() noexcept -> void;

    [[nodiscard]] auto find_entity(std::string_view) const noexcept -> entt::entity;

private:
    entt::registry registry;
    entt::sigh<void()> lights_changed_signal_;
    std::uint64_t hierarchy_revision_ = 0;
    std::uint64_t static_revision_ = 0;

public:
    std::unique_ptr<PhysicsWorld> physics_world;

private:
    Renderer &renderer_;

    auto mark_lights_dirty(entt::registry &, entt::entity) -> void;
    auto on_transform_changed(entt::registry &reg, entt::entity entity) -> void;
    auto on_rigid_body_destroyed(entt::registry &, entt::entity entity) -> void;
    auto on_script_attached(entt::registry &, entt::entity) -> void;
    auto on_script_detached(entt::registry &, entt::entity) -> void;
    auto on_material_override_attached(entt::registry &, entt::entity) -> void;
    auto on_material_override_detached(entt::registry &, entt::entity) -> void;

    auto on_model_destroyed(entt::registry &, entt::entity) -> void;
    auto on_streamed_model_tag_destroyed(entt::registry &, entt::entity) -> void;

    auto connect_light_signals() -> void;

    auto mark_static_changed(entt::registry &, entt::entity) -> void;
    auto connect_static_signals() -> void;

    auto mark_hierarchy_changed(entt::registry &, entt::entity) -> void;
    auto connect_hierarchy_signals() -> void;

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

}

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

    // World transforms for many entities in one frame. get_world_transform walks the whole chain of parents for every
    // entity it is asked about; this remembers the product of each parent chain, so entities under the same parents
    // share one walk. It is not thread safe: threads that split the entities use one each (a parent chain is short, so
    // they repeat little). Call begin_frame when the registry's transforms may have changed.
    class WorldTransformCache {
    public:
        auto begin_frame() -> void { ancestors_.clear(); }

        // The same matrix get_world_transform returns for `entity`.
        [[nodiscard]] auto get(entt::registry const &registry, entt::entity entity,
                               Components::Transform const &transform) -> glm::mat4;

        // The same, for an entity the caller knows has a Parent.
        [[nodiscard]] auto get_with_parent(entt::registry const &registry, entt::entity entity,
                                           Components::Transform const &transform) -> glm::mat4;

    private:
        enum class State : std::uint8_t { computing, known };

        struct Ancestors {
            glm::mat4 matrix{1.0F};
            State state = State::computing;
        };

        // The product of the transforms above `entity`, from the cache or computed (and kept) on the way up.
        [[nodiscard]] auto ancestors_of(entt::registry const &registry, entt::entity entity, int depth) -> glm::mat4;

        std::unordered_map<entt::entity, Ancestors> ancestors_;
    };

    auto lifetime(entt::registry &registry, PhysicsWorld &physics, float dt) -> void;
    auto script_update(entt::registry &registry, Scene &scene, float dt) -> void;
}
