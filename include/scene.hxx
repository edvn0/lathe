#pragma once

#include "physics.hxx"

#include <entt/entt.hpp>
#include <entt/entity/snapshot.hpp>

#include <any>
#include <utility>
#include <vector>

// Owns everything simulate/render iterate over. The editor keeps one Scene
// alive for the whole app lifetime; play/simulate just points Application's
// active_scene at whichever Scene should be ticked this frame.
struct Scene {
    entt::registry registry;
    PhysicsWorldSettings physics_settings{};
    float physics_accumulator = 0.0F;

    auto on_scene_start() -> void { physics_accumulator = 0.0F; }

    auto on_scene_stop() -> void { physics_accumulator = 0.0F; }
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
