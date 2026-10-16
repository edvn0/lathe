#include "assets/model_pick_geometry.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include <glm/geometric.hpp>

#include "assets/load_model.hxx"

namespace {

    constexpr std::uint32_t leaf_size = 4;
    constexpr int max_depth = 256;

    auto collect_node(ModelCpuData const &model, std::uint32_t index, glm::mat4 const &parent,
                      std::vector<glm::vec3> &positions, std::vector<std::uint32_t> &indices, int depth) -> void {
        if (index >= model.nodes.size() || depth > max_depth) {
            return;
        }

        auto const &node = model.nodes[index];
        auto const world = parent * node.local_transform;

        if (node.mesh_index < model.meshes.size()) {
            for (auto const &primitive: model.meshes[node.mesh_index].primitives) {
                if (primitive.vertices.empty() || primitive.indices.size() < 3) {
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
            collect_node(model, child, world, positions, indices, depth + 1);
        }
    }

    // Slab test against a box; the entry parameter, or nothing when the ray misses or the box lies past `limit`.
    auto ray_box_entry(glm::vec3 const &origin, glm::vec3 const &inverse_direction, glm::vec3 const &low,
                       glm::vec3 const &high, float limit) -> std::optional<float> {
        auto t_near = 0.0F;
        auto t_far = limit;

        for (auto axis = 0; axis < 3; ++axis) {
            auto t0 = (low[axis] - origin[axis]) * inverse_direction[axis];
            auto t1 = (high[axis] - origin[axis]) * inverse_direction[axis];

            if (t0 > t1) {
                std::swap(t0, t1);
            }

            // NaN (0 * inf, the ray running in the slab's plane) compares false and is ignored; the slab's other
            // bound then decides.
            t_near = std::max(t_near, t0);
            t_far = std::min(t_far, t1);

            if (t_near > t_far) {
                return std::nullopt;
            }
        }

        return t_near;
    }

    // Moller-Trumbore, two-sided.
    auto ray_triangle(glm::vec3 const &origin, glm::vec3 const &direction, glm::vec3 const &a, glm::vec3 const &b,
                      glm::vec3 const &c) -> std::optional<float> {
        constexpr auto epsilon = 1e-9F;

        auto const edge_ab = b - a;
        auto const edge_ac = c - a;
        auto const p = glm::cross(direction, edge_ac);
        auto const determinant = glm::dot(edge_ab, p);

        if (std::abs(determinant) < epsilon) {
            return std::nullopt;
        }

        auto const inverse = 1.0F / determinant;
        auto const to_origin = origin - a;
        auto const u = glm::dot(to_origin, p) * inverse;

        if (u < 0.0F || u > 1.0F) {
            return std::nullopt;
        }

        auto const q = glm::cross(to_origin, edge_ab);
        auto const v = glm::dot(direction, q) * inverse;

        if (v < 0.0F || u + v > 1.0F) {
            return std::nullopt;
        }

        auto const t = glm::dot(edge_ac, q) * inverse;

        return t >= 0.0F ? std::optional{t} : std::nullopt;
    }

}

auto ModelPickGeometry::build(ModelCpuData const &model) -> std::shared_ptr<ModelPickGeometry const> {
    std::vector<glm::vec3> positions;
    std::vector<std::uint32_t> indices;

    for (auto const root: model.scene_roots) {
        collect_node(model, root, glm::mat4{1.0F}, positions, indices, 0);
    }

    return build(std::move(positions), std::move(indices));
}

auto ModelPickGeometry::build(std::vector<glm::vec3> positions, std::vector<std::uint32_t> indices)
        -> std::shared_ptr<ModelPickGeometry const> {
    auto geometry = std::shared_ptr<ModelPickGeometry>{new ModelPickGeometry};
    geometry->positions_ = std::move(positions);
    geometry->indices_ = std::move(indices);

    // Drop a trailing partial triangle and any that index past the vertices.
    auto const vertex_count = static_cast<std::uint32_t>(geometry->positions_.size());
    std::vector<std::uint32_t> valid;
    valid.reserve(geometry->indices_.size());

    for (std::size_t i = 0; i + 2 < geometry->indices_.size(); i += 3) {
        if (geometry->indices_[i] < vertex_count && geometry->indices_[i + 1] < vertex_count &&
            geometry->indices_[i + 2] < vertex_count) {
            valid.insert(valid.end(), geometry->indices_.begin() + static_cast<std::ptrdiff_t>(i),
                         geometry->indices_.begin() + static_cast<std::ptrdiff_t>(i + 3));
        }
    }

    geometry->indices_ = std::move(valid);

    auto const triangles = static_cast<std::uint32_t>(geometry->indices_.size() / 3);

    if (triangles == 0) {
        return geometry;
    }

    std::vector<glm::vec3> centroids(triangles);
    std::vector<std::uint32_t> order(triangles);

    for (std::uint32_t t = 0; t < triangles; ++t) {
        auto const &a = geometry->positions_[geometry->indices_[(t * 3) + 0]];
        auto const &b = geometry->positions_[geometry->indices_[(t * 3) + 1]];
        auto const &c = geometry->positions_[geometry->indices_[(t * 3) + 2]];
        centroids[t] = (a + b + c) / 3.0F;
        order[t] = t;
    }

    geometry->nodes_.reserve(static_cast<std::size_t>(triangles) / 2 + 1);
    geometry->build_node(0, triangles, centroids, order);

    // Reorder the triangles to the leaves' order so a leaf is a contiguous range.
    std::vector<std::uint32_t> sorted(geometry->indices_.size());

    for (std::uint32_t t = 0; t < triangles; ++t) {
        for (auto corner = 0U; corner < 3; ++corner) {
            sorted[(t * 3) + corner] = geometry->indices_[(order[t] * 3) + corner];
        }
    }

    geometry->indices_ = std::move(sorted);

    return geometry;
}

auto ModelPickGeometry::build_node(std::uint32_t first, std::uint32_t count, std::vector<glm::vec3> const &centroids,
                                   std::vector<std::uint32_t> &order) -> std::uint32_t {
    auto const node_index = static_cast<std::uint32_t>(nodes_.size());
    nodes_.emplace_back();

    auto low = glm::vec3{std::numeric_limits<float>::max()};
    auto high = glm::vec3{std::numeric_limits<float>::lowest()};
    auto centroid_low = low;
    auto centroid_high = high;

    for (auto i = first; i < first + count; ++i) {
        auto const triangle = order[i];

        for (auto corner = 0U; corner < 3; ++corner) {
            auto const &point = positions_[indices_[(triangle * 3) + corner]];
            low = glm::min(low, point);
            high = glm::max(high, point);
        }

        centroid_low = glm::min(centroid_low, centroids[triangle]);
        centroid_high = glm::max(centroid_high, centroids[triangle]);
    }

    nodes_[node_index].low = low;
    nodes_[node_index].high = high;

    auto const extent = centroid_high - centroid_low;
    auto const axis = extent.x >= extent.y && extent.x >= extent.z ? 0 : (extent.y >= extent.z ? 1 : 2);

    if (count <= leaf_size || extent[axis] <= 0.0F) {
        nodes_[node_index].first = first;
        nodes_[node_index].count = count;
        return node_index;
    }

    auto const middle = first + (count / 2);
    std::nth_element(order.begin() + static_cast<std::ptrdiff_t>(first),
                     order.begin() + static_cast<std::ptrdiff_t>(middle),
                     order.begin() + static_cast<std::ptrdiff_t>(first + count),
                     [&](std::uint32_t left, std::uint32_t right) { return centroids[left][axis] < centroids[right][axis]; });

    auto const left = build_node(first, middle - first, centroids, order);
    auto const right = build_node(middle, first + count - middle, centroids, order);

    nodes_[node_index].first = left;
    nodes_[node_index].right = right;
    nodes_[node_index].count = 0;

    return node_index;
}

auto ModelPickGeometry::raycast(glm::vec3 const &origin, glm::vec3 const &direction, float max_distance) const
        -> std::optional<ModelPickHit> {
    if (nodes_.empty()) {
        return std::nullopt;
    }

    auto const inverse_direction = glm::vec3{1.0F} / direction;
    auto best = max_distance;
    std::optional<std::uint32_t> best_triangle;

    std::array<std::uint32_t, 128> stack{};
    auto top = 0U;
    stack[top++] = 0;

    while (top > 0) {
        auto const &node = nodes_[stack[--top]];

        if (!ray_box_entry(origin, inverse_direction, node.low, node.high, best)) {
            continue;
        }

        if (node.count > 0) {
            for (auto triangle = node.first; triangle < node.first + node.count; ++triangle) {
                auto const &a = positions_[indices_[(triangle * 3) + 0]];
                auto const &b = positions_[indices_[(triangle * 3) + 1]];
                auto const &c = positions_[indices_[(triangle * 3) + 2]];

                if (auto const t = ray_triangle(origin, direction, a, b, c); t && *t < best) {
                    best = *t;
                    best_triangle = triangle;
                }
            }
        } else if (top + 2 <= stack.size()) {
            stack[top++] = node.first;
            stack[top++] = node.right;
        }
    }

    if (!best_triangle) {
        return std::nullopt;
    }

    auto const triangle = *best_triangle;
    ModelPickHit hit;
    hit.distance = best;
    hit.position = origin + (direction * best);
    hit.triangle = triangle;
    hit.corners = {positions_[indices_[(triangle * 3) + 0]], positions_[indices_[(triangle * 3) + 1]],
                   positions_[indices_[(triangle * 3) + 2]]};

    auto normal = glm::cross(hit.corners[1] - hit.corners[0], hit.corners[2] - hit.corners[0]);
    auto const length = glm::length(normal);
    normal = length > 0.0F ? normal / length : glm::vec3{0.0F, 1.0F, 0.0F};
    hit.normal = glm::dot(normal, direction) > 0.0F ? -normal : normal;

    return hit;
}
