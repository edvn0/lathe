#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "load_model.hxx"

struct PrimitiveMeshData {
    std::vector<ModelVertex> vertices;
    std::vector<std::uint32_t> indices;
};

// Procedurally generated engine primitives. These share ModelVertex/
// generate_tangents with the glTF importer, so they go through the exact
// same GPU upload and shading path as imported meshes (see engine_models.hxx).

[[nodiscard]] auto make_cube_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError>;

[[nodiscard]] auto make_sphere_mesh(std::uint32_t rings = 16, std::uint32_t segments = 32)
        -> std::expected<PrimitiveMeshData, ModelLoadError>;

// A single grass "clump": three crossed blade quads fanned around the
// vertical axis, base at local y=0, tip at local y=1. Meant to be scaled/
// rotated/scattered per-instance and drawn with a material whose
// wind_strength > 0 -- see wind.slang. Both winding orders are emitted per
// quad so blades stay visible under the forward pass's back-face culling
// from any viewing angle, without needing a dedicated no-cull pipeline.
[[nodiscard]] auto make_grass_clump_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError>;

[[nodiscard]] auto make_capsule_mesh(std::uint32_t segments = 16, std::uint32_t rings = 8)
        -> std::expected<PrimitiveMeshData, ModelLoadError>;
