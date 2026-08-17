#pragma once

#include <array>

static constexpr auto frames_in_flight = 2U;

static constexpr std::uint32_t shadow_cascade_count = 4;

template<typename T>
using FIFArray = std::array<T, frames_in_flight>;

using FrameIndex = decltype(frames_in_flight);
