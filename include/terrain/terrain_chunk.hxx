#pragma once

#include <cstdint>
#include <vector>

#include "assets/load_model.hxx"
#include "terrain/terrain_mesh.hxx"

// Every chunk at every LOD is a 65x65 vertex grid plus a one-vertex skirt ring, so TerrainSlotPool can reuse a
// fixed set of GPU slots.
inline constexpr std::uint32_t terrain_chunk_samples = 65; // interior vertices per side
inline constexpr std::uint32_t terrain_chunk_cells = terrain_chunk_samples - 1; // 64

inline constexpr std::uint32_t terrain_chunk_interior_vertex_count = terrain_chunk_samples * terrain_chunk_samples; // 4225
inline constexpr std::uint32_t terrain_chunk_skirt_vertex_count = 4U * terrain_chunk_samples; // 260
inline constexpr std::uint32_t terrain_chunk_vertex_count =
        terrain_chunk_interior_vertex_count + terrain_chunk_skirt_vertex_count; // 4485

inline constexpr std::uint32_t terrain_chunk_interior_index_count = terrain_chunk_cells * terrain_chunk_cells * 6U; // 24576
inline constexpr std::uint32_t terrain_chunk_skirt_index_count = 4U * terrain_chunk_cells * 6U; // 1536
inline constexpr std::uint32_t terrain_chunk_index_count =
        terrain_chunk_interior_index_count + terrain_chunk_skirt_index_count; // 26112

// Vertex layout, as indices into TerrainChunkResult::vertices:
//   [0, 4225)     interior grid, row-major: index = row * 65 + column
//   [4225, 4290)  south skirt  (row 0),  one entry per column
//   [4290, 4355)  north skirt  (row 64), one entry per column
//   [4355, 4420)  west  skirt  (col 0),  one entry per row
//   [4420, 4485)  east  skirt  (col 64), one entry per row
[[nodiscard]] constexpr auto terrain_chunk_interior_index(std::uint32_t column, std::uint32_t row) -> std::uint32_t {
    return row * terrain_chunk_samples + column;
}

struct TerrainChunkRequest {
    // World-space centre of the chunk.
    float world_origin_x = 0.0F;
    float world_origin_z = 0.0F;

    // World units per cell; span = 64 * cell_size.
    float cell_size = 1.0F;
};

struct TerrainChunkResult {
    // terrain_chunk_vertex_count entries in the layout above, uploaded as-is alongside terrain_chunk_indices().
    std::vector<CompressedModelVertex> vertices;

    // Interior heights, row-major, for LOD0 colliders.
    std::vector<float> heights;

    float min_height = 0.0F;
    float max_height = 0.0F;
};

// Builds one chunk on the CPU. Skips the weld/optimize path, which would break the fixed vertex layout;
// tangents are analytic instead (u along +X, v along +Z, so handedness is always -1).
//
// `field.params()` needs a real height_range_min/max so every chunk agrees on mid_height and skirt depth.
//
// Thread-safe: only reads `field` and `request`.
[[nodiscard]] auto make_terrain_chunk(TerrainField const &field, TerrainChunkRequest const &request)
        -> TerrainChunkResult;

// The index buffer shared by every chunk, built once. Optimized for vertex cache only, so the vertex layout is
// kept.
[[nodiscard]] auto terrain_chunk_indices() -> std::vector<std::uint32_t> const &;
