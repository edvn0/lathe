#pragma once

#include <cstdint>
#include <span>
#include <vector>

struct TerrainGreedyQuad {
    std::uint32_t column = 0;
    std::uint32_t row = 0;
    std::uint32_t width = 1;
    std::uint32_t height = 1;
};

[[nodiscard]] auto greedy_merge_terrain_cells(std::span<float const> heights, std::uint32_t samples_x,
                                              std::uint32_t samples_z, float tolerance)
        -> std::vector<TerrainGreedyQuad>;

[[nodiscard]] auto triangulate_terrain_quads(std::span<TerrainGreedyQuad const> quads, std::uint32_t samples_x,
                                             std::uint32_t samples_z) -> std::vector<std::uint32_t>;
