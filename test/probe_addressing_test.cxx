#include <doctest/doctest.h>

#include <set>

#include "rendering/probe_addressing.hxx"

using namespace probe_addressing;

TEST_CASE("wrap keeps negative indices in range") {
    CHECK(wrap(0, 8) == 0);
    CHECK(wrap(7, 8) == 7);
    CHECK(wrap(8, 8) == 0);
    CHECK(wrap(-1, 8) == 7);
    CHECK(wrap(-8, 8) == 0);
    CHECK(wrap(-9, 8) == 7);
}

TEST_CASE("every probe of the window has its own slot, also when the window sits on negative indices") {
    constexpr Index3 counts{6, 4, 5};
    for (auto const base: {Index3{0, 0, 0}, Index3{-3, 2, -7}, Index3{13, -1, 4}}) {
        std::set<std::uint32_t> slots;
        for (auto z = 0; z < counts[2]; ++z) {
            for (auto y = 0; y < counts[1]; ++y) {
                for (auto x = 0; x < counts[0]; ++x) {
                    slots.insert(slot_of({base[0] + x, base[1] + y, base[2] + z}, counts));
                }
            }
        }
        CHECK(slots.size() == 6U * 4U * 5U);
        CHECK(*slots.rbegin() == 6U * 4U * 5U - 1U);
    }
}

TEST_CASE("world_index inverts slot_of for the window") {
    constexpr Index3 counts{8, 3, 8};
    for (auto const base: {Index3{0, 0, 0}, Index3{-5, -1, 9}, Index3{100, 2, -100}}) {
        for (auto z = 0; z < counts[2]; ++z) {
            for (auto y = 0; y < counts[1]; ++y) {
                for (auto x = 0; x < counts[0]; ++x) {
                    Index3 const index{base[0] + x, base[1] + y, base[2] + z};
                    CHECK(world_index(slot_of(index, counts), base, counts) == index);
                }
            }
        }
    }
}

TEST_CASE("scrolling the window by one probe only reassigns the probes that left") {
    constexpr Index3 counts{8, 4, 8};
    constexpr Index3 before{10, 0, -4};
    constexpr Index3 after{11, 0, -4};
    auto changed = 0;
    for (std::uint32_t slot = 0; slot < 8U * 4U * 8U; ++slot) {
        if (world_index(slot, before, counts) != world_index(slot, after, counts)) {
            ++changed;
        }
    }
    // One x layer: counts.y * counts.z slots take a new world index.
    CHECK(changed == 4 * 8);

    // A probe inside both windows keeps its slot.
    Index3 const kept{15, 2, 0};
    CHECK(world_index(slot_of(kept, counts), before, counts) == kept);
    CHECK(world_index(slot_of(kept, counts), after, counts) == kept);
}
