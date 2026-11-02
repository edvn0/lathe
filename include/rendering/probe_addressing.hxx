#pragma once

#include <array>
#include <cstdint>

// The probe volume's toroidal addressing, mirrored from gi_probes.slang (gi_wrap, gi_slot_of, gi_slot3, gi_world_index) so
// the maths can be tested on the CPU. A probe keeps its slot while the window scrolls over it: slot = world index modulo
// the counts, laid out x fastest.
namespace probe_addressing {

using Index3 = std::array<std::int32_t, 3>;

[[nodiscard]] constexpr auto wrap(std::int32_t index, std::int32_t count) -> std::int32_t {
    return ((index % count) + count) % count;
}

[[nodiscard]] constexpr auto slot_of(Index3 const &index, Index3 const &counts) -> std::uint32_t {
    auto const x = wrap(index[0], counts[0]);
    auto const y = wrap(index[1], counts[1]);
    auto const z = wrap(index[2], counts[2]);
    return static_cast<std::uint32_t>(x + counts[0] * (y + counts[1] * z));
}

[[nodiscard]] constexpr auto slot3(std::uint32_t slot, Index3 const &counts) -> Index3 {
    auto const s = static_cast<std::int32_t>(slot);
    return {s % counts[0], (s / counts[0]) % counts[1], s / (counts[0] * counts[1])};
}

// The world index of the probe a slot holds while the window starts at `base`.
[[nodiscard]] constexpr auto world_index(std::uint32_t slot, Index3 const &base, Index3 const &counts) -> Index3 {
    auto const cell = slot3(slot, counts);
    return {base[0] + wrap(cell[0] - base[0], counts[0]), base[1] + wrap(cell[1] - base[1], counts[1]),
            base[2] + wrap(cell[2] - base[2], counts[2])};
}

} // namespace probe_addressing
