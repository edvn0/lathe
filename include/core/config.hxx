#pragma once

#include <array>

static constexpr auto frames_in_flight = 2U;
static constexpr auto shadow_cascade_count = 4U;

static constexpr auto lod_count = 4U;

// An instance is drawn at the first LOD whose threshold its projected bounding radius, in pixels, still reaches, so the
// detail follows how big the object looks on screen: its size, its distance, the field of view and the render height.
static constexpr std::array<float, lod_count - 1> lod_radius_pixels{200.0F, 90.0F, 40.0F};

static constexpr std::array<float, lod_count - 1> lod_simplification_ratios{0.5F, 0.25F, 0.1F};

template<typename T>
using FIFArray = std::array<T, frames_in_flight>;

using FrameIndex = decltype(frames_in_flight);
