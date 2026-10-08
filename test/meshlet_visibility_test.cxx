#include <doctest/doctest.h>

#include <cstdint>

#include "rendering/meshlet_visibility.hxx"

TEST_CASE("meshlet visibility word counts round up to whole 32-bit words") {
    CHECK(meshlet_visibility_word_count(0) == 0);
    CHECK(meshlet_visibility_word_count(1) == 1);
    CHECK(meshlet_visibility_word_count(31) == 1);
    CHECK(meshlet_visibility_word_count(32) == 1);
    CHECK(meshlet_visibility_word_count(33) == 2);
    CHECK(meshlet_visibility_word_count(maximum_meshlet_visibility_bits) == maximum_meshlet_visibility_bits / 32U);
}

TEST_CASE("consecutive batches get contiguous, disjoint bit ranges") {
    MeshletVisibilityLayout layout;

    auto const first = layout.reserve(3, 100);
    auto const second = layout.reserve(2, 7);
    auto const third = layout.reserve(1, 50);

    CHECK(first == 0);
    CHECK(second == 300);
    CHECK(third == 314);
    CHECK(layout.total_bits() == 364);
    CHECK(layout.word_count() == 12);
    CHECK(layout.fits());
}

TEST_CASE("an instance's offset steps by the meshlet count within its batch") {
    MeshletVisibilityLayout layout;
    [[maybe_unused]] auto const skipped = layout.reserve(4, 25);
    auto const batch = layout.reserve(5, 40);

    CHECK(batch == 100);

    for (std::uint32_t instance = 0; instance < 5; ++instance) {
        auto const offset = meshlet_visibility_offset(batch, instance, 40);
        CHECK(offset == 100 + (instance * 40));
    }

    CHECK(meshlet_visibility_offset(batch, 4, 40) + 39 == layout.total_bits() - 1);
}

TEST_CASE("batches without meshlet bits take none") {
    MeshletVisibilityLayout layout;

    CHECK(layout.total_bits() == 0);
    CHECK(layout.word_count() == 0);
    CHECK(layout.fits());

    CHECK(layout.reserve(0, 64) == 0);
    CHECK(layout.reserve(8, 0) == 0);
    CHECK(layout.reserve(1, 1) == 0);
    CHECK(layout.total_bits() == 1);
}

TEST_CASE("the layout stops fitting one bit past the cap") {
    MeshletVisibilityLayout at_cap;
    [[maybe_unused]] auto const full = at_cap.reserve(1, static_cast<std::uint32_t>(maximum_meshlet_visibility_bits));
    CHECK(at_cap.total_bits() == maximum_meshlet_visibility_bits);
    CHECK(at_cap.fits());

    MeshletVisibilityLayout over_cap;
    [[maybe_unused]] auto const first =
            over_cap.reserve(1, static_cast<std::uint32_t>(maximum_meshlet_visibility_bits));
    [[maybe_unused]] auto const last_bit = over_cap.reserve(1, 1);
    CHECK(over_cap.total_bits() == maximum_meshlet_visibility_bits + 1);
    CHECK_FALSE(over_cap.fits());

    MeshletVisibilityLayout huge;
    for (int batch = 0; batch < 4; ++batch) {
        [[maybe_unused]] auto const reserved = huge.reserve(1U << 20U, 1U << 12U);
    }

    CHECK(huge.total_bits() == std::uint64_t{4} << 32U);
    CHECK_FALSE(huge.fits());
}

TEST_CASE("offsets past the cap read as 0 so they never overflow 32 bits") {
    CHECK(meshlet_visibility_offset(maximum_meshlet_visibility_bits - 1, 0, 64) ==
          static_cast<std::uint32_t>(maximum_meshlet_visibility_bits - 1));
    CHECK(meshlet_visibility_offset(maximum_meshlet_visibility_bits, 0, 64) == 0);
    CHECK(meshlet_visibility_offset(0, 1U << 31U, 1U << 8U) == 0);
}
