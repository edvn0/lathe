#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <glm/vec3.hpp>

// The uniform grid the probe GI's trace looks its explicit emitters up in (docs/probe-gi.md): a hit reads the emitters of
// its own cell only. An emitter is a sphere (a bulb) that lights things up to `emitter_reach(radius)` metres away.
struct EmitterSphere {
    glm::vec3 centre{0.0F};
    float radius = 0.1F;
};

// How far an emitter of this radius lights (mirrors the trace's falloff range).
[[nodiscard]] inline auto emitter_reach(float radius) noexcept -> float {
    return radius * 80.0F > 1.0F ? radius * 80.0F : 1.0F;
}

struct EmitterGrid {
    // False when no grid fits the limits: nothing may be looked up.
    bool valid = false;
    glm::vec3 origin{0.0F};
    float cell = 1.0F;
    glm::uvec3 dims{1};
    // Per cell: the first index into `indices` and how many emitters follow, x fastest.
    std::vector<std::uint32_t> first;
    std::vector<std::uint32_t> count;
    // Emitter numbers (positions in the span given), the emitters of each cell one after the other.
    std::vector<std::uint32_t> indices;
};

// Cells are twice the largest reach, so an emitter touches at most two cells per axis, eight in all; the cell grows until
// the grid has at most `max_cells` cells and `max_indices` entries (valid = false if it never does).
[[nodiscard]] auto build_emitter_grid(std::span<EmitterSphere const> emitters, std::uint32_t max_cells,
                                      std::uint32_t max_indices) -> EmitterGrid;
