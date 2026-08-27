#pragma once

#include <array>

static constexpr auto frames_in_flight = 2U;
static constexpr auto shadow_cascade_count = 4U;

static constexpr auto lod_count = 4U; // LOD0 (full detail) + 3 generated

// Ascending world-space distances (from camera) at which an instance steps
// down to the next-coarser LOD.
static constexpr std::array<float, lod_count - 1> lod_distances{15.0F, 35.0F, 75.0F};

// Target index-count fraction of LOD0, for LOD1..LOD(lod_count-1).
static constexpr std::array<float, lod_count - 1> lod_simplification_ratios{0.5F, 0.25F, 0.1F};

template<typename T>
using FIFArray = std::array<T, frames_in_flight>;

using FrameIndex = decltype(frames_in_flight);
