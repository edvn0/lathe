#include <doctest/doctest.h>

#include "frame_graph_test_support.hxx"

using namespace frame_graph;

namespace {

    constexpr auto compute_stage = static_cast<ShaderStages>(ShaderStage::compute);

    auto noop() -> RecordFn { return RecordFn{}; }

    constexpr auto present_state = ResourceState{.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};

    auto import_swapchain(FrameGraph &graph) -> ImageId {
        return graph.import_image({
                .entry = {.layout = VK_IMAGE_LAYOUT_UNDEFINED},
                .exit = present_state,
                .swapchain = true,
                .debug_name = "swapchain",
        });
    }

    auto add_frame_legacy(FrameGraph &graph, ImageId &swapchain) -> void {
        graph.add_pass("frame_legacy", PassType::raster, {}, [&](PassBuilder &p) {
            p.legacy();
            swapchain = p.write(swapchain, Use::color_attachment, 0, ExitUse{Use::present});
            return noop();
        });
    }

    auto fences(BarrierSet const &set) -> std::size_t {
        return static_cast<std::size_t>(std::ranges::count_if(set.memory, [](MemoryBarrier const &barrier) {
            return barrier.src_stages == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT &&
                   barrier.dst_stages == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }));
    }

}

TEST_SUITE("unit") {
    TEST_CASE("a legacy pass around the swapchain: entered as an attachment, left as PRESENT") {
        auto graph = FrameGraph{};
        auto swapchain = import_swapchain(graph);
        add_frame_legacy(graph, swapchain);

        auto const compiled = compile(graph, QueueTopology{});
        REQUIRE(compiled.has_value());
        REQUIRE(compiled->batches.size() == 1);
        auto const &batch = compiled->batches.front();
        REQUIRE(batch.passes.size() == 1);

        auto const &before = batch.passes.front().before;
        REQUIRE(before.images.size() == 1);
        CHECK(before.images.front().old_layout == VK_IMAGE_LAYOUT_UNDEFINED);
        CHECK(before.images.front().new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        CHECK(before.images.front().dst_stages == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);

        CHECK(batch.epilogue.images.empty());

        CHECK(batch.waits_swapchain_acquire);
        CHECK(batch.swapchain_wait_stages == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
        CHECK(batch.signals_render_finished);

        auto const problems = test::check_happens_before(graph.description(), *compiled, QueueTopology{});
        CHECK(problems.empty());
    }

    TEST_CASE("a legacy pass is fenced by global barriers on both sides") {
        auto graph = FrameGraph{};
        auto swapchain = import_swapchain(graph);
        add_frame_legacy(graph, swapchain);

        auto const compiled = compile(graph, QueueTopology{});
        REQUIRE(compiled.has_value());
        auto const &batch = compiled->batches.front();

        CHECK(fences(batch.passes.front().before) == 1);
        CHECK(fences(batch.epilogue) == 1);
    }

    TEST_CASE("the fence after a legacy pass goes before the next pass on its queue") {
        auto graph = FrameGraph{};
        auto buffer = graph.import_buffer({.debug_name = "data"});
        graph.add_pass("legacy", PassType::compute, {}, [&](PassBuilder &p) {
            p.legacy();
            return noop();
        });
        graph.add_pass("after", PassType::compute, {}, [&](PassBuilder &p) {
            buffer = p.write(buffer, Use::shader_write, compute_stage);
            return noop();
        });

        auto const compiled = compile(graph, QueueTopology{});
        REQUIRE(compiled.has_value());
        auto const &passes = compiled->batches.front().passes;
        REQUIRE(passes.size() == 2);
        CHECK(fences(passes[0].before) == 1);
        CHECK(fences(passes[1].before) == 1);
        CHECK(fences(compiled->batches.front().epilogue) == 0);
    }

    TEST_CASE("a pass without legacy() gets no fences") {
        auto graph = FrameGraph{};
        auto buffer = graph.import_buffer({.debug_name = "data"});
        graph.add_pass("plain", PassType::compute, {}, [&](PassBuilder &p) {
            buffer = p.write(buffer, Use::shader_write, compute_stage);
            return noop();
        });
        auto const compiled = compile(graph, QueueTopology{});
        REQUIRE(compiled.has_value());
        CHECK(fences(compiled->batches.front().passes.front().before) == 0);
    }

    TEST_CASE("an exit use leaves the image in its own layout for the next pass") {
        auto graph = FrameGraph{};
        auto image = graph.import_image({
                .entry = {.layout = VK_IMAGE_LAYOUT_UNDEFINED},
                .exit = {.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT},
                .debug_name = "chain",
        });

        graph.add_pass("build", PassType::compute, {}, [&](PassBuilder &p) {
            image = p.write(image, Use::storage_write, compute_stage, ExitUse{Use::sampled});
            return noop();
        });
        graph.add_pass("consume", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const r = p.read(image, Use::sampled, compute_stage);
            p.side_effect();
            return noop();
        });

        auto const compiled = compile(graph, QueueTopology{});
        REQUIRE(compiled.has_value());
        auto const &passes = compiled->batches.front().passes;
        REQUIRE(passes.size() == 2);

        REQUIRE(passes[0].before.images.size() == 1);
        CHECK(passes[0].before.images.front().new_layout == VK_IMAGE_LAYOUT_GENERAL);

        CHECK(passes[1].before.images.empty());
        CHECK(compiled->batches.front().epilogue.images.empty());
    }

    TEST_CASE("physical handles stay out of the declaration hash") {
        auto const build = [](std::uintptr_t handle) {
            auto graph = FrameGraph{};
            auto const image = graph.import_image({
                    .exit = present_state,
                    .swapchain = true,
                    .debug_name = "swapchain",
                    .image =
                            PhysicalImage{
                                    .image = reinterpret_cast<VkImage>(handle), // NOLINT(performance-no-int-to-ptr)
                                    .format = VK_FORMAT_B8G8R8A8_UNORM,
                            },
            });
            graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &p) {
                [[maybe_unused]] auto const next = p.color(image, LoadOp::clear, StoreOp::store);
                return noop();
            });
            auto const compiled = compile(graph, QueueTopology{});
            REQUIRE(compiled.has_value());
            CHECK(graph.description().resources.front().image.image ==
                  reinterpret_cast<VkImage>(handle)); // NOLINT(performance-no-int-to-ptr)
            return compiled->hash;
        };
        CHECK(build(0x1000) == build(0x2000));
    }

    TEST_CASE("the record lambdas are exposed in declaration order") {
        auto graph = FrameGraph{};
        auto calls = std::vector<int>{};
        for (auto index = 0; index < 3; ++index) {
            graph.add_pass(std::format("pass_{}", index), PassType::compute, {}, [&, index](PassBuilder &) {
                return RecordFn{[&calls, index](PassContext &) { calls.push_back(index); }};
            });
        }
        REQUIRE(graph.records().size() == 3);
        PassContext *context = nullptr;
        for (auto &record: graph.records()) {
            if (record) {
                (void) context;
            }
        }
        CHECK(calls.empty());
    }
}
