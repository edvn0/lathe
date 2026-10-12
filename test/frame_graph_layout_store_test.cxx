#include <doctest/doctest.h>

#include "app/frame_graph_layout_store.hxx"

using namespace gui;

TEST_SUITE("unit") {
    TEST_CASE("positions survive a round trip") {
        auto positions = NodePositions{};
        positions[stable_key("stage", "bloom")] = {.x = 12.5F, .y = -340.0F};
        positions[stable_key("pass", "forward_pass")] = {.x = 0.0F, .y = 1.0F / 3.0F};

        auto const parsed = parse_positions(serialise_positions(positions));
        CHECK(parsed == positions);
    }

    TEST_CASE("serialised positions are ordered by key so the file is stable") {
        auto positions = NodePositions{};
        positions[3] = {.x = 1.0F, .y = 2.0F};
        positions[1] = {.x = 3.0F, .y = 4.0F};

        CHECK(serialise_positions(positions) == "1 3 4\n3 1 2\n");
    }

    TEST_CASE("parsing skips malformed lines and keeps the rest") {
        auto const parsed = parse_positions("zz 1 2\n5 1\n7 8 9\n\n9 oops 1\n");

        REQUIRE(parsed.size() == 1);
        CHECK(parsed.at(7) == NodePosition{.x = 8.0F, .y = 9.0F});
    }

    TEST_CASE("stable keys differ by scope and name and fit in 56 bits") {
        CHECK(stable_key("stage", "bloom") != stable_key("pass", "bloom"));
        CHECK(stable_key("pass", "a") != stable_key("pass", "b"));
        CHECK(stable_key("pass", "forward_pass") < (std::uintptr_t{1} << 56U));
        CHECK(stable_key("pass", "forward_pass") == stable_key("pass", "forward_pass"));
    }
}
