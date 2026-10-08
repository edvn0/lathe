#pragma once

#include <cstdint>
#include <vector>

#include "assets/load_model.hxx"
#include "assets/meshlet.hxx"
#include "terrain/terrain_mesh.hxx"

inline constexpr std::uint32_t terrain_chunk_samples = 65;
inline constexpr std::uint32_t terrain_chunk_cells = terrain_chunk_samples - 1;

inline constexpr std::uint32_t terrain_chunk_interior_vertex_count =
        terrain_chunk_samples * terrain_chunk_samples;
inline constexpr std::uint32_t terrain_chunk_skirt_vertex_count = 4U * terrain_chunk_samples;
inline constexpr std::uint32_t terrain_chunk_vertex_count =
        terrain_chunk_interior_vertex_count + terrain_chunk_skirt_vertex_count;

inline constexpr std::uint32_t terrain_chunk_interior_index_count =
        terrain_chunk_cells * terrain_chunk_cells * 6U;
inline constexpr std::uint32_t terrain_chunk_skirt_index_count = 4U * terrain_chunk_cells * 6U;
inline constexpr std::uint32_t terrain_chunk_index_count =
        terrain_chunk_interior_index_count + terrain_chunk_skirt_index_count;

[[nodiscard]] constexpr auto terrain_chunk_interior_index(std::uint32_t column, std::uint32_t row) -> std::uint32_t {
    return row * terrain_chunk_samples + column;
}

struct TerrainChunkRequest {
    float world_origin_x = 0.0F;
    float world_origin_z = 0.0F;

    float cell_size = 1.0F;
};

struct TerrainChunkResult {
    std::vector<CompressedModelVertex> vertices;

    std::vector<std::uint32_t> indices;

    MeshletBuild meshlets;

    std::vector<float> heights;

    float min_height = 0.0F;
    float max_height = 0.0F;
};

[[nodiscard]] auto make_terrain_chunk(TerrainField const &field, TerrainChunkRequest const &request)
        -> TerrainChunkResult;

[[nodiscard]] auto terrain_chunk_indices() -> std::vector<std::uint32_t> const &;
