#pragma once

#include <expected>

#include <glm/mat4x4.hpp>

#include "assets/geometry_arena.hxx"
#include "assets/mesh_create_info.hxx"
#include "assets/model.hxx" // MeshHandle
#include "core/renderer_error.hxx"

// Narrow surface of Renderer that terrain streaming needs to create and
// submit GPU meshes. terrain_streamer.hxx/terrain_slot_pool.cxx/
// terrain_world.cxx depend on this instead of the full Renderer -- Renderer
// consumes terrain's output (it sits above terrain in the module layering),
// so terrain calling back into it directly would be a cycle.
struct IMeshSink {
    [[nodiscard]]
    virtual auto create_mesh(MeshCreateInfo const &create_info) -> std::expected<MeshHandle, RendererError> = 0;

    [[nodiscard]]
    virtual auto submit_mesh(MeshHandle mesh, glm::mat4 const &transform, MaterialHandle material_override = {})
            -> std::expected<void, RendererError> = 0;

    [[nodiscard]]
    virtual auto geometry_arena() noexcept -> GeometryArena & = 0;

protected:
    ~IMeshSink() = default;
};
