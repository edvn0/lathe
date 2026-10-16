#include "app/surface_pick.hxx"

#include <algorithm>
#include <cmath>
#include <limits>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {

    // Slab test in the box's local space; returns the ray parameter of the entry point (0 if the origin is inside).
    auto ray_box_hit(glm::vec3 const &origin, glm::vec3 const &direction, glm::vec3 const &low, glm::vec3 const &high)
            -> std::optional<float> {
        auto t_near = 0.0F;
        auto t_far = std::numeric_limits<float>::max();

        for (auto axis = 0; axis < 3; ++axis) {
            if (std::abs(direction[axis]) < 1e-8F) {
                if (origin[axis] < low[axis] || origin[axis] > high[axis]) {
                    return std::nullopt;
                }
                continue;
            }

            auto t0 = (low[axis] - origin[axis]) / direction[axis];
            auto t1 = (high[axis] - origin[axis]) / direction[axis];
            if (t0 > t1) {
                std::swap(t0, t1);
            }
            t_near = std::max(t_near, t0);
            t_far = std::min(t_far, t1);
            if (t_near > t_far) {
                return std::nullopt;
            }
        }

        return t_near;
    }

    auto ray_sphere_hit(glm::vec3 const &origin, glm::vec3 const &direction, glm::vec3 const &centre, float radius)
            -> std::optional<float> {
        auto const to_centre = centre - origin;
        auto const along = glm::dot(to_centre, direction);
        auto const closest_sq = glm::dot(to_centre, to_centre) - (along * along);
        if (along < 0.0F || closest_sq > radius * radius) {
            return std::nullopt;
        }
        return along - std::sqrt(radius * radius - closest_sq);
    }

    auto box_face_normal(glm::vec3 const &point, glm::vec3 const &low, glm::vec3 const &high) -> glm::vec3 {
        auto const centre = (low + high) * 0.5F;
        auto const half = glm::max((high - low) * 0.5F, glm::vec3{1e-6F});
        auto const scaled = (point - centre) / half;
        auto const axis = std::abs(scaled.x) >= std::abs(scaled.y) && std::abs(scaled.x) >= std::abs(scaled.z)
                                  ? 0
                                  : (std::abs(scaled.y) >= std::abs(scaled.z) ? 1 : 2);
        auto normal = glm::vec3{0.0F};
        normal[axis] = scaled[axis] >= 0.0F ? 1.0F : -1.0F;
        return normal;
    }

}

// Nearest surface under the ray (world space, unit direction). Models are tested against their submesh bounds and,
// when their triangles are on the CPU, against those; lights and other model-less transforms against a small sphere
// so they stay clickable.
auto pick_surface(Scene &scene, Renderer &renderer, glm::vec3 const &origin, glm::vec3 const &direction,
                  PickTargets targets) -> std::optional<SurfaceHit> {
    constexpr auto marker_radius = 0.3F;

    auto &registry = scene.get_registry();
    std::optional<SurfaceHit> best;

    for (auto const [entity, transform]: registry.view<Components::Transform const>().each()) {
        auto const world = systems::get_world_transform(registry, entity, transform);
        std::optional<SurfaceHit> candidate;

        auto const *model = registry.try_get<Components::Model const>(entity);
        auto const bounds = model != nullptr ? renderer.model_submesh_bounds(model->model) : std::nullopt;

        if (bounds && !bounds->empty()) {
            auto low = bounds->front().first;
            auto high = bounds->front().second;
            for (auto const &[lo, hi]: *bounds) {
                low = glm::min(low, lo);
                high = glm::max(high, hi);
            }

            auto const to_local = glm::inverse(world);
            auto const local_origin = glm::vec3{to_local * glm::vec4{origin, 1.0F}};
            auto const local_direction = glm::vec3{to_local * glm::vec4{direction, 0.0F}};
            auto const normal_matrix = glm::transpose(glm::mat3{to_local});

            auto const to_world_normal = [&](glm::vec3 const &local_normal) {
                auto normal = glm::normalize(normal_matrix * local_normal);
                return glm::dot(normal, direction) > 0.0F ? -normal : normal;
            };

            if (auto const box_t = ray_box_hit(local_origin, local_direction, low, high)) {
                auto const geometry = renderer.model_pick_geometry(model->model);
                auto const surface = geometry ? geometry->raycast(local_origin, local_direction,
                                                                  std::numeric_limits<float>::max())
                                              : std::nullopt;

                if (surface) {
                    auto const position = glm::vec3{world * glm::vec4{surface->position, 1.0F}};
                    auto world_corners = std::array<glm::vec3, 3>{};
                    for (auto corner = 0U; corner < 3; ++corner) {
                        world_corners[corner] = glm::vec3{world * glm::vec4{surface->corners[corner], 1.0F}};
                    }
                    candidate = SurfaceHit{.entity = entity,
                                           .distance = glm::dot(position - origin, direction),
                                           .position = position,
                                           .normal = to_world_normal(surface->normal),
                                           .triangle = world_corners};
                } else if (!geometry && targets == PickTargets::everything) {
                    auto const local_hit = local_origin + (local_direction * *box_t);
                    auto const position = glm::vec3{world * glm::vec4{local_hit, 1.0F}};
                    candidate = SurfaceHit{.entity = entity,
                                           .distance = glm::dot(position - origin, direction),
                                           .position = position,
                                           .normal = to_world_normal(box_face_normal(local_hit, low, high))};
                }
            }
        } else if (model == nullptr && targets == PickTargets::everything) {
            if (auto const distance = ray_sphere_hit(origin, direction, glm::vec3{world[3]}, marker_radius)) {
                auto const position = origin + (direction * *distance);
                candidate = SurfaceHit{.entity = entity,
                                       .distance = *distance,
                                       .position = position,
                                       .normal = -direction};
            }
        }

        if (candidate && (!best || candidate->distance < best->distance)) {
            best = candidate;
        }
    }

    return best;
}

auto pick_entity(Scene &scene, Renderer &renderer, glm::vec3 const &origin, glm::vec3 const &direction)
        -> entt::entity {
    auto const hit = pick_surface(scene, renderer, origin, direction);
    return hit ? hit->entity : entt::entity{entt::null};
}

