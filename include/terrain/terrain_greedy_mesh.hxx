#pragma once

#include <cstdint>
#include <span>
#include <vector>

// A rectangle of heightfield cells, in cell units: covers columns [column, column + width) and rows
// [row, row + height).
struct TerrainGreedyQuad {
    std::uint32_t column = 0;
    std::uint32_t row = 0;
    std::uint32_t width = 1;
    std::uint32_t height = 1;
};

// Greedy-merges the cells of a `samples_x` x `samples_z` row-major height grid into rectangles. Each rectangle is
// grown along +X, then along +Z, while every sample it covers stays within `tolerance` (world units) of the plane
// through its corners. Single cells always qualify, so the result tiles the whole grid.
[[nodiscard]] auto greedy_merge_terrain_cells(std::span<float const> heights, std::uint32_t samples_x,
                                              std::uint32_t samples_z, float tolerance)
        -> std::vector<TerrainGreedyQuad>;

// Triangle indices (row * samples_x + column) for `quads`, wound like a single unmerged cell (front face +Y).
//
// Every quad corner that lands on another quad's edge is stitched into that edge, so there are no T-junctions.
// Every grid-boundary vertex is stitched in too, so the edges still line up with skirts and same-LOD neighbours.
// Quads with nothing on their edges stay two triangles.
[[nodiscard]] auto triangulate_terrain_quads(std::span<TerrainGreedyQuad const> quads, std::uint32_t samples_x,
                                             std::uint32_t samples_z) -> std::vector<std::uint32_t>;
