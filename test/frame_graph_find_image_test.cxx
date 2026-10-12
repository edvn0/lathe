#include <doctest/doctest.h>

#include "frame_graph_test_support.hxx"

using namespace frame_graph;

TEST_SUITE("unit") {
    TEST_CASE("find_image returns the latest version of a named image") {
        auto graph = FrameGraph{};
        auto target = graph.import_image({.debug_name = "target"});
        auto const before = graph.find_image("target");
        REQUIRE(before.has_value());
        CHECK(before->index == 0);

        graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &p) {
            target = p.color(target, LoadOp::clear, StoreOp::store);
            p.side_effect();
            return RecordFn{};
        });

        auto const after = graph.find_image("target");
        REQUIRE(after.has_value());
        CHECK(after->index == before->index);
        CHECK(after->generation == target.generation);
        CHECK(after->generation > before->generation);
    }

    TEST_CASE("find_image ignores buffers and unknown names") {
        auto graph = FrameGraph{};
        [[maybe_unused]] auto const buffer = graph.import_buffer({.debug_name = "data"});

        CHECK_FALSE(graph.find_image("data").has_value());
        CHECK_FALSE(graph.find_image("missing").has_value());
    }
}
