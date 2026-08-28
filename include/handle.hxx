#pragma once

#include <cstdint>
#include <limits>

/*
 * Generic generational handle used by every ObjectPool<T, Sentinel> (see
 * object_pool.hxx). Parameterized directly on the pool's payload type T so
 * each domain (ImageSlotData, GpuMaterial, Pipeline, ...) automatically gets
 * its own distinct handle type without a separate phantom tag.
 *
 * Sentinel is the index value a default-constructed handle carries, and is
 * also excluded by valid() -- most pools default it to the maximum uint32_t
 * (an index no real pool capacity can reach), but a pool may permanently
 * reserve index 0 as non-addressable storage (see MaterialStorage's default
 * material slot) by picking Sentinel = 0 instead.
 */
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
