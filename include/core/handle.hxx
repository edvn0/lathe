#pragma once

#include <cstdint>
#include <limits>

template<typename T, std::uint32_t Sentinel = std::numeric_limits<std::uint32_t>::max()>
struct Handle {
    static constexpr std::uint32_t sentinel = Sentinel;

    std::uint32_t index = Sentinel;
    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return generation != 0 && index != Sentinel;
    }

    auto operator==(Handle const &) const -> bool = default;
};
