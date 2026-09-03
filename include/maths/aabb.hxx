#pragma once

#include <utility>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace maths {

    // Transforms an axis-aligned min/max box through `matrix` and returns the
    // tightest axis-aligned box around the result, without assuming anything
    // about how the matrix's rows/columns map to glm's mul convention: each
    // local axis, scaled by its half-extent, is pushed through the matrix and
    // the transformed axes' absolute values are summed. Mirrors
    // transform_aabb in assets/shaders/frustum_cull.slang and is used
    // everywhere else in the engine that needs to fold a local-space AABB
    // through a node/world transform (Renderer::model_submesh_bounds, debug
    // AABB visualization) -- kept as one shared implementation so those
    // stay consistent with what the GPU culling pass actually tests against.
    [[nodiscard]]
    inline auto transform_aabb(glm::mat4 const &matrix, glm::vec3 const &min, glm::vec3 const &max)
            -> std::pair<glm::vec3, glm::vec3> {
        auto const centre = (min + max) * 0.5F;
        auto const extent = (max - min) * 0.5F;

        auto const world_centre = glm::vec3(matrix * glm::vec4(centre, 1.0F));

        auto const axis_x = glm::vec3(matrix * glm::vec4(extent.x, 0.0F, 0.0F, 0.0F));
        auto const axis_y = glm::vec3(matrix * glm::vec4(0.0F, extent.y, 0.0F, 0.0F));
        auto const axis_z = glm::vec3(matrix * glm::vec4(0.0F, 0.0F, extent.z, 0.0F));

        auto const world_extent = glm::abs(axis_x) + glm::abs(axis_y) + glm::abs(axis_z);

        return {world_centre - world_extent, world_centre + world_extent};
    }

} // namespace maths
