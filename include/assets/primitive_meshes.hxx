#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include <glm/vec3.hpp>

#include "assets/load_model.hxx"

struct PrimitiveMeshData {
    std::vector<ModelVertex> vertices;
    std::vector<std::uint32_t> indices;

    // Hand-made index buffers for LOD1..LOD(lod_count-1) into the same `vertices`; nullopt reuses the level before.
    // Primitives get no simplified LODs otherwise.
    std::array<std::optional<std::vector<std::uint32_t>>, lod_count - 1> lod_indices{};
};

// Wraps a generated mesh as single-mesh ModelCpuData for Renderer::create_model_from_cpu_data.
[[nodiscard]] auto to_model_cpu_data(PrimitiveMeshData mesh) -> ModelCpuData;

// Procedural primitives, sharing the import path's vertex format and tangent generation.

[[nodiscard]] auto make_cube_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError>;

[[nodiscard]] auto make_sphere_mesh(std::uint32_t rings = 16, std::uint32_t segments = 32)
        -> std::expected<PrimitiveMeshData, ModelLoadError>;

// A grass clump about 0.6 m across and up to 0.9 m tall, base at y=0, for instancing with a wind material. Blades
// are single-sided, so its materials must be double-sided. Hand-made LODs (lod_distances): LOD0 is 12 curved blades
// (60 triangles), LOD1 6 wider, straighter ones (18), LOD2 and LOD3 three crossed cards (6) textured with
// make_grass_card_texture(), which need an alpha-tested far material (MaterialCreateInfo::far_material).
[[nodiscard]] auto make_grass_clump_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError>;

// The LOD the grass clump switches from blades to cards at.
inline constexpr std::uint32_t grass_clump_card_lod = 2;

[[nodiscard]] auto make_capsule_mesh(std::uint32_t segments = 16, std::uint32_t rings = 8)
        -> std::expected<PrimitiveMeshData, ModelLoadError>;

// A strip of quads through a row-major grid of world-space points: `columns` points across (first to last),
// the rest along. Roads, rivers and paths. Normals come from the grid, so pass points already draped over any
// terrain. Facing is up when, looking along the strip, the last column is to the right of the first (+Z across
// for +X along). UVs are `uv_scale` per metre, U across and V along. Needs columns >= 2 and >= 2 rows.
[[nodiscard]] auto make_ribbon_mesh(std::span<glm::vec3 const> grid, std::uint32_t columns, float uv_scale = 0.25F)
        -> std::expected<PrimitiveMeshData, ModelLoadError>;
