#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <future>
#include <memory>
#include <vector>

#include <glm/vec3.hpp>

#include "assets/load_model.hxx"

class btBvhTriangleMeshShape;
class btTriangleIndexVertexArray;

// A static triangle-mesh collision shape (a Bullet BVH over the triangles). It owns its triangles, takes no world, and
// can be built on any thread and shared by several PhysicsWorlds, so a model's collider can be prepared while the model
// streams in.
class MeshCollider {
public:
    [[nodiscard]] static auto build(std::vector<glm::vec3> positions, std::vector<std::uint32_t> indices)
            -> std::shared_ptr<MeshCollider const>;

    ~MeshCollider();
    MeshCollider(MeshCollider const &) = delete;
    auto operator=(MeshCollider const &) -> MeshCollider & = delete;

    [[nodiscard]] auto shape() const noexcept -> btBvhTriangleMeshShape * { return shape_.get(); }
    [[nodiscard]] auto triangle_count() const noexcept -> std::size_t { return indices_.size() / 3; }

private:
    MeshCollider() = default;

    std::vector<glm::vec3> positions_;
    std::vector<std::uint32_t> indices_;
    std::unique_ptr<btTriangleIndexVertexArray> mesh_;
    std::unique_ptr<btBvhTriangleMeshShape> shape_;
};

// Filled in the background while a model loads (see collect_model_collider). `ready` is set once `mesh` is built.
struct MeshColliderSlot {
    std::atomic<bool> ready{false};
    std::atomic<bool> failed{false};
    std::shared_ptr<MeshCollider const> mesh;
    std::future<void> build;
};

// An observer for ModelStreamer::request that, when the model has loaded, collects the triangles of every mesh instance
// (node transforms applied) and builds a MeshCollider from them on another thread, without delaying the model.
// Primitives whose bounding-box diagonal is below `min_extent` (wires, string lights, clutter a character should walk
// through) are left out. The slot reports the result.
struct ColliderObserver {
    std::function<void(ModelCpuData const &)> observer;
    std::shared_ptr<MeshColliderSlot> slot;
};

[[nodiscard]] auto make_collider_observer(float min_extent) -> ColliderObserver;

// The triangles of `model` as one world-space soup; exposed for tests.
auto collect_model_triangles(ModelCpuData const &model, float min_extent, std::vector<glm::vec3> &positions,
                             std::vector<std::uint32_t> &indices) -> void;
