#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "assets/load_model.hxx"

struct PrimitiveMeshData {
    std::vector<ModelVertex> vertices;
    std::vector<std::uint32_t> indices;
};

// Wraps a generated mesh as single-mesh ModelCpuData for Renderer::create_model_from_cpu_data.
[[nodiscard]] auto to_model_cpu_data(PrimitiveMeshData mesh) -> ModelCpuData;

// Procedural primitives, sharing the import path's vertex format and tangent generation.

[[nodiscard]] auto make_cube_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError>;

[[nodiscard]] auto make_sphere_mesh(std::uint32_t rings = 16, std::uint32_t segments = 32)
        -> std::expected<PrimitiveMeshData, ModelLoadError>;

// A grass clump: three crossed blades around the vertical axis, base at y=0, tip at y=1. Meant for instancing
// with a wind material. Both windings are emitted so blades show from either side with back-face culling on.
[[nodiscard]] auto make_grass_clump_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError>;

[[nodiscard]] auto make_capsule_mesh(std::uint32_t segments = 16, std::uint32_t rings = 8)
        -> std::expected<PrimitiveMeshData, ModelLoadError>;
