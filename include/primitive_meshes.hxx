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
