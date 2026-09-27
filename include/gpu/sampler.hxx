#pragma once

#include <cstdint>
#include <limits>

#include "core/handle.hxx"

inline constexpr auto invalid_sampler_index = std::numeric_limits<std::uint32_t>::max();

// Defined in sampler_storage.hxx. Handle<T> doesn't need a complete T.
struct SamplerSlotData;

using SamplerHandle = Handle<SamplerSlotData>;

enum class DefaultSampler : std::uint32_t {
    linear_repeat = 0,
    linear_clamp = 1,
    nearest_repeat = 2,
    nearest_clamp = 3,
    shadow_compare = 4,
};

enum class SamplerClass : std::uint8_t {
    regular,
    comparison,
};
