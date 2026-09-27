#pragma once

#include <cstdint>
#include <limits>

// Generational handle for ObjectPool<T, Sentinel>; each payload type gets its own handle type.
//
// Sentinel is the index of a default-constructed handle and is never valid(). Most pools use the maximum
// uint32_t; pools that reserve slot 0 use 0.
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
