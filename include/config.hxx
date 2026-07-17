#pragma once

#include <array>

static constexpr auto frames_in_flight = 3U;

template<typename T>
using FIFArray = std::array<T, frames_in_flight>;

using FrameIndex = decltype(frames_in_flight);
