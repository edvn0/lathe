#include "physics.hxx"

#include <cstddef>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <vector>

#include "transform.hxx"

namespace {
    // Every non-static body accelerates toward every Attractor entity
    // (other than itself). O(bodies * attractors); fine as long as
    // attractors stay a small subset of the scene.
    auto apply_attraction(entt::registry &registry, PhysicsWorldSettings const &settings, float fixed_dt) -> void {
        auto attractors = registry.view<Transform const, Attractor const>();

        if (attractors.begin() == attractors.end()) {
            return;
        }

        auto bodies = registry.view<Transform, RigidBody>();

        for (auto body_entity: bodies) {
            auto &body = bodies.get<RigidBody>(body_entity);

            if (body.is_static) {
                continue;
            }

            auto &transform = bodies.get<Transform>(body_entity);

            for (auto [attractor_entity, attractor_transform, attractor]: attractors.each()) {
                if (attractor_entity == body_entity) {
                    continue;
                }

                auto const delta = attractor_transform.position - transform.position;
                auto const distance = glm::length(delta);

                // A body exactly at an attractor's position has no defined
                // pull direction; normalize() would divide by zero and
                // inject NaN into velocity (and from there, position).
                if (distance <= 1e-6F) {
                    continue;
                }

                auto const distance_squared =
                        glm::dot(delta, delta) + settings.attraction_softening * settings.attraction_softening;
                auto const acceleration = attractor.strength / distance_squared;

                body.velocity += (delta / distance) * acceleration * fixed_dt;
            }
        }
    }

    auto integrate(entt::registry &registry, PhysicsWorldSettings const &settings, float fixed_dt) -> void {
        auto view = registry.view<Transform, RigidBody>();

        for (auto entity: view) {
            auto &body = view.get<RigidBody>(entity);

            if (body.is_static) {
                continue;
            }

            auto &transform = view.get<Transform>(entity);

            body.velocity += settings.gravity * fixed_dt;
            transform.position += body.velocity * fixed_dt;

            auto const floor = settings.ground_y + body.half_extents.y;

            if (transform.position.y < floor) {
                transform.position.y = floor;

                if (body.velocity.y < 0.0F) {
                    body.velocity.y = -body.velocity.y * body.restitution;
                }
            }
        }
    }

    // Separates a pair of overlapping AABBs along their axis of least
    // overlap and cancels the closing component of relative velocity along
    // that axis (with a bit of restitution), so bodies settle instead of
    // interpenetrating or jittering indefinitely.
    auto resolve_pair(Transform &transform_a, RigidBody &body_a, Transform &transform_b, RigidBody &body_b) -> void {
        auto const delta = transform_b.position - transform_a.position;
        auto const overlap = (body_a.half_extents + body_b.half_extents) - glm::abs(delta);

        if (overlap.x <= 0.0F || overlap.y <= 0.0F || overlap.z <= 0.0F) {
            return;
        }

        auto axis = 0;

        if (overlap.y < overlap.x && overlap.y < overlap.z) {
            axis = 1;
        } else if (overlap.z < overlap.x && overlap.z < overlap.y) {
            axis = 2;
        }

        auto const sign = delta[axis] >= 0.0F ? 1.0F : -1.0F;
        auto const push = overlap[axis] * sign;

        if (body_b.is_static) {
            transform_a.position[axis] -= push;
        } else if (body_a.is_static) {
            transform_b.position[axis] += push;
        } else {
            transform_a.position[axis] -= push * 0.5F;
            transform_b.position[axis] += push * 0.5F;
        }

        auto const relative_velocity = body_b.velocity[axis] - body_a.velocity[axis];

        if (relative_velocity * sign >= 0.0F) {
            return;
        }

        auto const restitution = glm::min(body_a.restitution, body_b.restitution);
        auto const impulse = -(1.0F + restitution) * relative_velocity * 0.5F;

        if (!body_a.is_static) {
            body_a.velocity[axis] -= impulse;
        }

        if (!body_b.is_static) {
            body_b.velocity[axis] += impulse;
        }
    }

    // A per-step backstop, not a physical force: naive sequential impulse
    // resolution over a dense, mutually-attracting cluster has no
    // convergence guarantee and can compound velocity indefinitely. This
    // only clips the rare runaway case -- normal falling/bouncing never
    // approaches max_speed.
    auto clamp_velocities(entt::registry &registry, float max_speed) -> void {
        auto view = registry.view<RigidBody>();

        for (auto entity: view) {
            auto &body = view.get<RigidBody>(entity);
            auto const speed = glm::length(body.velocity);

            if (speed > max_speed) {
                body.velocity *= max_speed / speed;
            }
        }
    }

    auto resolve_collisions(entt::registry &registry) -> void {
        auto view = registry.view<Transform, RigidBody>();
        auto const entities = std::vector<entt::entity>{view.begin(), view.end()};

        for (std::size_t i = 0; i < entities.size(); ++i) {
            auto &body_a = view.get<RigidBody>(entities[i]);

            if (body_a.is_static) {
                continue;
            }

            auto &transform_a = view.get<Transform>(entities[i]);

            for (std::size_t j = i + 1; j < entities.size(); ++j) {
                resolve_pair(transform_a, body_a, view.get<Transform>(entities[j]), view.get<RigidBody>(entities[j]));
            }
        }
    }
} // namespace

auto simulate_physics(entt::registry &registry, PhysicsWorldSettings const &settings, float fixed_dt) -> void {
    apply_attraction(registry, settings, fixed_dt);
    integrate(registry, settings, fixed_dt);
    resolve_collisions(registry);
    clamp_velocities(registry, settings.max_speed);
}
