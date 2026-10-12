#include "physics/mesh_collider.hxx"

#include <btBulletCollisionCommon.h>
#include <BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h>
#include <BulletCollision/CollisionShapes/btTriangleIndexVertexArray.h>

#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>

#include <algorithm>
#include <utility>

#include "core/logger.hxx"

namespace {

    auto collect_node(ModelCpuData const &model, std::uint32_t index, glm::mat4 const &parent, float min_extent,
                      std::vector<glm::vec3> &positions, std::vector<std::uint32_t> &indices, int depth) -> void {
        if (index >= model.nodes.size() || depth > 256) {
            return;
        }

        auto const &node = model.nodes[index];
        auto const world = parent * node.local_transform;

        if (node.mesh_index < model.meshes.size()) {
            for (auto const &primitive: model.meshes[node.mesh_index].primitives) {
                if (primitive.vertices.empty() || primitive.indices.size() < 3) {
                    continue;
                }

                if (primitive.bounds && glm::length(primitive.bounds->second - primitive.bounds->first) < min_extent) {
                    continue;
                }

                auto const base = static_cast<std::uint32_t>(positions.size());

                for (auto const &vertex: primitive.vertices) {
                    positions.emplace_back(world * glm::vec4{vertex.position, 1.0F});
                }

                for (std::size_t i = 0; i + 2 < primitive.indices.size(); i += 3) {
                    indices.push_back(base + primitive.indices[i]);
                    indices.push_back(base + primitive.indices[i + 1]);
                    indices.push_back(base + primitive.indices[i + 2]);
                }
            }
        }

        for (auto const child: node.children) {
            collect_node(model, child, world, min_extent, positions, indices, depth + 1);
        }
    }

}

auto collect_model_triangles(ModelCpuData const &model, float min_extent, std::vector<glm::vec3> &positions,
                             std::vector<std::uint32_t> &indices) -> void {
    for (auto const root: model.scene_roots) {
        collect_node(model, root, glm::mat4{1.0F}, min_extent, positions, indices, 0);
    }
}

MeshCollider::~MeshCollider() = default;

auto MeshCollider::build(std::vector<glm::vec3> positions, std::vector<std::uint32_t> indices)
        -> std::shared_ptr<MeshCollider const> {
    auto collider = std::shared_ptr<MeshCollider>{new MeshCollider};

    collider->positions_ = std::move(positions);
    collider->indices_ = std::move(indices);

    // Bullet's quantised BVH numbers triangles within a part with 21 bits, so large meshes are split into parts that
    // share the vertex array.
    constexpr std::size_t triangles_per_part = 1U << 20U;
    auto mesh = std::make_unique<btTriangleIndexVertexArray>();
    auto const triangles = collider->indices_.size() / 3;

    for (std::size_t first = 0; first < triangles; first += triangles_per_part) {
        btIndexedMesh part;
        part.m_numTriangles = static_cast<int>(std::min(triangles_per_part, triangles - first));
        part.m_triangleIndexBase = reinterpret_cast<unsigned char const *>(collider->indices_.data() + first * 3);
        part.m_triangleIndexStride = 3 * sizeof(std::uint32_t);
        part.m_numVertices = static_cast<int>(collider->positions_.size());
        part.m_vertexBase = reinterpret_cast<unsigned char const *>(collider->positions_.data());
        part.m_vertexStride = sizeof(glm::vec3);
        mesh->addIndexedMesh(part, PHY_INTEGER);
    }

    collider->mesh_ = std::move(mesh);
    collider->shape_ = std::make_unique<btBvhTriangleMeshShape>(collider->mesh_.get(), true, true);

    return collider;
}

auto make_collider_observer(float min_extent) -> ColliderObserver {
    auto slot = std::make_shared<MeshColliderSlot>();

    auto observer = [slot, min_extent](ModelCpuData const &model) {
        std::vector<glm::vec3> positions;
        std::vector<std::uint32_t> indices;
        collect_model_triangles(model, min_extent, positions, indices);

        if (indices.empty()) {
            slot->failed.store(true, std::memory_order_release);
            return;
        }

        slot->build = std::async(std::launch::async,
                                 [slot, positions = std::move(positions), indices = std::move(indices)]() mutable {
                                     auto const triangles = indices.size() / 3;
                                     slot->mesh = MeshCollider::build(std::move(positions), std::move(indices));
                                     slot->ready.store(true, std::memory_order_release);
                                     info("Built a mesh collider of {} triangles", triangles);
                                 });
    };

    return {.observer = std::move(observer), .slot = std::move(slot)};
}
