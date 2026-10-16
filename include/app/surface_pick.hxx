#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>

#include <entt/entt.hpp>
#include <glm/glm.hpp>

class Renderer;
class Scene;

// Where a ray first meets an entity's surface, in world space.
struct SurfaceHit {
    entt::entity entity = entt::null;
    float distance = std::numeric_limits<float>::max();
    glm::vec3 position{0.0F};
    // Unit, facing the ray's origin. For a model without triangles on the CPU it is the bounding box's face.
    glm::vec3 normal{0.0F, 1.0F, 0.0F};
    // The hit triangle's corners in world space, when the model has triangles on the CPU.
    std::optional<std::array<glm::vec3, 3>> triangle;
};

enum class PickTargets : std::uint8_t {
    everything,
    // Only models whose triangles are known, so lights, markers and a model's bounding box are not surfaces: what the
    // ray finds is what is drawn.
    models,
};

// The nearest surface under the ray (world space, unit direction). Models are tested against their submesh bounds and,
// when their triangles are on the CPU, against those; lights and other model-less transforms against a small sphere so
// they stay clickable.
[[nodiscard]] auto pick_surface(Scene &scene, Renderer &renderer, glm::vec3 const &origin, glm::vec3 const &direction,
                               PickTargets targets = PickTargets::everything) -> std::optional<SurfaceHit>;

[[nodiscard]] auto pick_entity(Scene &scene, Renderer &renderer, glm::vec3 const &origin, glm::vec3 const &direction)
        -> entt::entity;
