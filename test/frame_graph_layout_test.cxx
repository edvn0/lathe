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
