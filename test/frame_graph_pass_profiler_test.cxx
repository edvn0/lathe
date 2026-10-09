#include <doctest/doctest.h>

#include "rendering/frame_graph/pass_profiler.hxx"

using frame_graph::timestamp_distance;

TEST_SUITE("unit") {
    TEST_CASE("timestamp distance is plain subtraction for 64-bit counters") {
        CHECK(timestamp_distance(100, 250, 64) == 150);
        CHECK(timestamp_distance(250, 100, 64) == -150);
    }

    TEST_CASE("timestamp distance follows a narrow counter across its wrap") {
        constexpr auto bits = 36U;
        constexpr auto top = (std::uint64_t{1} << bits) - 1;

        CHECK(timestamp_distance(top - 4, 5, bits) == 10);
        CHECK(timestamp_distance(5, top - 4, bits) == -10);
    }

    TEST_CASE("timestamp distance treats an unreported width as 64 bits") {
        CHECK(timestamp_distance(10, 30, 0) == 20);
    }
}
