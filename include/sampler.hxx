#pragma once

#include <cstdint>
#include <limits>

inline constexpr auto invalid_sampler_index = std::numeric_limits<std::uint32_t>::max();

struct SamplerHandle {
    std::uint32_t index = invalid_sampler_index;

    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return generation != 0 && index != invalid_sampler_index;
    }

    auto operator==(SamplerHandle const &) const -> bool = default;
};

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
