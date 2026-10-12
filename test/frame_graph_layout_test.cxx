#include <doctest/doctest.h>

#include "frame_graph_test_support.hxx"
#include "rendering/frame_graph/layout.hxx"

using namespace frame_graph;

namespace {

    constexpr auto compute_stage = static_cast<ShaderStages>(ShaderStage::compute);

    auto dedicated() -> QueueTopology {
        auto topology = QueueTopology{};
        topology.family = {0, 2};
        return topology;
    }

    // produce (async compute) -> consume, plus an unrelated pass.
    auto make_view() -> FrameGraphView {
        auto graph = FrameGraph{};
        auto data = graph.import_buffer({.debug_name = "data"});
        auto other = graph.import_buffer({.debug_name = "other"});
        graph.add_pass("produce", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            data = p.write(data, Use::shader_write, compute_stage);
            return RecordFn{};
        });
        graph.add_pass("consume", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const r = p.read(data, Use::shader_read, compute_stage);
            p.side_effect();
            return RecordFn{};
        });
        graph.add_pass("independent", PassType::compute, {}, [&](PassBuilder &p) {
            other = p.write(other, Use::shader_write, compute_stage);
            p.side_effect();
            return RecordFn{};
        });

        auto compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        return FrameGraphView{.graph = graph.description(), .compiled = *compiled, .revision = 1};
    }

}

TEST_SUITE("unit") {
    TEST_CASE("layout derives an edge from the pass that produced the accessed version") {
        auto const result = layout(make_view());

        REQUIRE(result.edges.size() == 1);
        CHECK(result.edges[0].from == 0);
        CHECK(result.edges[0].to == 1);
        CHECK(result.edges[0].resources.size() == 1);
        CHECK(result.edges[0].cross_queue);
    }

    TEST_CASE("layout places a consumer one column after its producer") {
        auto const result = layout(make_view());
        auto const params = LayoutParams{};

        CHECK(result.nodes[0].depth == 0);
        CHECK(result.nodes[1].depth == 1);
        CHECK(result.nodes[2].depth == 0);
        CHECK(result.nodes[1].x == doctest::Approx(params.column_width));
        CHECK(result.nodes[0].x == doctest::Approx(0.0F));
    }

    TEST_CASE("layout separates queues into lanes") {
        auto const result = layout(make_view());

        CHECK(result.nodes[0].queue != result.nodes[1].queue);
        CHECK(result.nodes[0].y != doctest::Approx(result.nodes[1].y));
        CHECK(result.nodes[2].queue == result.nodes[1].queue);
        CHECK(result.nodes[2].y == doctest::Approx(result.nodes[1].y));
    }

    TEST_CASE("layout collapses grouped passes into one node and drops the edge inside it") {
        auto const view = make_view();
        auto const groups = std::vector<std::uint32_t>{7, 7, no_group};
        auto const result = layout(view, {}, {.group_of_pass = groups});

        REQUIRE(result.nodes.size() == 2);
        CHECK(result.nodes[0].passes == std::vector<std::uint32_t>{0, 1});
        CHECK(result.nodes[0].group == 7);
        CHECK(result.nodes[1].group == no_group);
        CHECK(result.edges.empty());
    }

    TEST_CASE("layout ignores edges that run against declaration order when finding depth") {
        auto const view = make_view();
        // 0 and 2 grouped, 1 on its own: 0 -> 1 is a forward edge, and so is 1's relation to the merged node.
        auto const groups = std::vector<std::uint32_t>{3, no_group, 3};
        auto const result = layout(view, {}, {.group_of_pass = groups});

        REQUIRE(result.nodes.size() == 2);
        CHECK(result.nodes[0].depth == 0);
        CHECK(result.nodes[1].depth == 1);
    }

    TEST_CASE("layout stacks nodes sharing a column and lane by their heights") {
        auto graph = FrameGraph{};
        auto first = graph.import_buffer({.debug_name = "first"});
        auto second = graph.import_buffer({.debug_name = "second"});
        graph.add_pass("a", PassType::compute, {}, [&](PassBuilder &p) {
            first = p.write(first, Use::shader_write, compute_stage);
            p.side_effect();
            return RecordFn{};
        });
        graph.add_pass("b", PassType::compute, {}, [&](PassBuilder &p) {
            second = p.write(second, Use::shader_write, compute_stage);
            p.side_effect();
            return RecordFn{};
        });
        auto compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        auto const view = FrameGraphView{.graph = graph.description(), .compiled = *compiled, .revision = 1};

        auto const groups = std::vector<std::uint32_t>{0, 1};
        auto const heights = std::vector<float>{200.0F, 50.0F};
        auto const params = LayoutParams{};
        auto const result = layout(view, params, {.group_of_pass = groups, .group_heights = heights});

        REQUIRE(result.nodes.size() == 2);
        REQUIRE(result.nodes[0].queue == result.nodes[1].queue);
        CHECK(result.nodes[0].x == doctest::Approx(result.nodes[1].x));
        CHECK(result.nodes[1].y == doctest::Approx(result.nodes[0].y + 200.0F + params.row_gap));
    }

    TEST_CASE("layout carries the culled flag") {
        auto view = make_view();
        view.compiled.pass_culled.assign(view.graph.passes.size(), false);
        view.compiled.pass_culled[2] = true;

        auto const result = layout(view);
        CHECK_FALSE(result.nodes[0].culled);
        CHECK(result.nodes[2].culled);
    }

    TEST_CASE("layout of an empty view is empty") {
        auto const result = layout(FrameGraphView{});
        CHECK(result.nodes.empty());
        CHECK(result.edges.empty());
    }
}
