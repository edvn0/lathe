#include <doctest/doctest.h>

#include "rendering/frame_graph/compiler.hxx"

TEST_SUITE("unit") {
    TEST_CASE("frame graph compiler compiles with only its own headers") {
        auto graph = frame_graph::FrameGraph{};
        auto const image = graph.import_image({
                .entry = {.layout = VK_IMAGE_LAYOUT_UNDEFINED},
                .exit = {.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
                .swapchain = true,
                .debug_name = "swapchain",
        });
        graph.add_pass("clear", frame_graph::PassType::raster, {}, [&](frame_graph::PassBuilder &pass) {
            [[maybe_unused]] auto const next =
                    pass.color(image, frame_graph::LoadOp::clear, frame_graph::StoreOp::store);
            return frame_graph::RecordFn{};
        });

        auto const compiled = frame_graph::compile(graph, frame_graph::QueueTopology{});
        REQUIRE(compiled.has_value());
        CHECK(compiled->batches.size() == 1);
        CHECK(compiled->batches.front().passes.size() == 1);
    }
}
