#pragma once

#include <cstdint>
#include <vector>

#include "load_model.hxx"
#include "terrain_mesh.hxx"

// Every terrain chunk, at every LOD, is a fixed 65x65 interior vertex grid
// plus a 1-vertex skirt ring -- this is what lets TerrainSlotPool recycle a
// fixed set of GPU slots instead of allocating/freeing per chunk (see the
// terrain streaming plan). These constants describe that fixed layout.
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

// Fixed vertex layout every chunk uses, indices into TerrainChunkResult::vertices:
//   [0, 4225)            interior grid, row-major: index = row * 65 + column
//   [4225, 4290)  south skirt  (row 0),  one entry per column
//   [4290, 4355)  north skirt  (row 64), one entry per column
//   [4355, 4420)  west  skirt  (col 0),  one entry per row
//   [4420, 4485)  east  skirt  (col 64), one entry per row
[[nodiscard]] constexpr auto terrain_chunk_interior_index(std::uint32_t column, std::uint32_t row) -> std::uint32_t {
    return row * terrain_chunk_samples + column;
}

struct TerrainChunkRequest {
    // World-space centre of this chunk.
    float world_origin_x = 0.0F;
    float world_origin_z = 0.0F;

    // World units per grid cell at this chunk's LOD -- span = 64 * cell_size.
    float cell_size = 1.0F;
};

struct TerrainChunkResult {
    // Always exactly terrain_chunk_vertex_count entries, in the fixed
    // layout above -- suitable for direct upload into a TerrainSlotPool
    // slot alongside the single shared index buffer from
    // terrain_chunk_indices().
    std::vector<CompressedModelVertex> vertices;

    // Interior-only, row-major, terrain_chunk_interior_vertex_count entries
    // -- the same heightfield-ready format TerrainMeshResult::heights uses,
    // for LOD0 chunks that get a physics collider.
    std::vector<float> heights;

    float min_height = 0.0F;
    float max_height = 0.0F;
};

// Pure CPU generation of one fixed-layout terrain chunk against `field`'s
// noise. Deliberately bypasses make_terrain_mesh's weld/optimize path
// (generate_tangents() in load_model.cxx runs mikktspace + a meshopt vertex
// remap, which would destroy the fixed vertex count and the
// heights<->vertex correspondence) -- tangents are analytic here instead:
// with u along +X and v along +Z, the shader computes
// `bitangent = cross(normal, tangent) * w`, and cross(+Y, +X) = -Z, so the
// tangent handedness is always -1.
//
// `field.params()` must have a real height_range_min/max set (max > min) --
// see TerrainParams -- since every chunk needs to agree on the same
// mid_height and skirt_depth for chunk boundaries to align exactly.
//
// Safe to call from any thread: touches only `field` (a thread-safe const
// noise field, see TerrainField) and `request`, and allocates its own
// output -- this is what makes it directly submit_task-able by
// TerrainStreamer without also needing to run compress_vertices()'s
// thread-pool-internal blocking path (see compress_vertex(), used per
// vertex here instead).
[[nodiscard]] auto make_terrain_chunk(TerrainField const &field, TerrainChunkRequest const &request)
        -> TerrainChunkResult;

// The canonical index buffer shared by every chunk at every LOD -- built
// once (meshopt-optimized for vertex cache only, never vertex fetch, so the
// row-major vertex layout above is preserved) and cached. Identical
// terrain_chunk_index_count uint32 indices regardless of chunk origin/LOD,
// which is what lets every chunk share one GeometryArena index slice.
[[nodiscard]] auto terrain_chunk_indices() -> std::vector<std::uint32_t> const &;
