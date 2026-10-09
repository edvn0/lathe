#include <doctest/doctest.h>

#include "frame_graph_test_support.hxx"
#include "gpu/image.hxx"
#include "rendering/frame_graph/names.hxx"
#include "rendering/frame_graph/resource_info.hxx"

using namespace frame_graph;

namespace {

    constexpr auto compute_stage = static_cast<ShaderStages>(ShaderStage::compute);

    // produce writes `data`; consume and extra read it; other is unrelated.
    auto make_view() -> FrameGraphView {
        auto graph = FrameGraph{};
        auto data = graph.import_buffer({.debug_name = "data"});
        graph.add_pass("produce", PassType::compute, {}, [&](PassBuilder &p) {
            data = p.write(data, Use::shader_write, compute_stage);
            return RecordFn{};
        });
        graph.add_pass("consume", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const r = p.read(data, Use::shader_read, compute_stage);
            p.side_effect();
            return RecordFn{};
        });
        graph.add_pass("extra", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const r = p.read(data, Use::shader_read, compute_stage);
            p.side_effect();
            return RecordFn{};
        });

        auto topology = QueueTopology{};
        topology.family = {0, 2};
        auto compiled = compile(graph, topology);
        REQUIRE(compiled.has_value());
        return FrameGraphView{.graph = graph.description(), .compiled = *compiled, .revision = 1};
    }

}

TEST_SUITE("unit") {
    TEST_CASE("resource info lists producers, consumers and the span of passes") {
        auto const info = describe_resource(make_view(), 0);

        CHECK(info.producers == std::vector<std::uint32_t>{0});
        CHECK(info.consumers == std::vector<std::uint32_t>{1, 2});
        CHECK(info.first_pass == 0);
        CHECK(info.last_pass == 2);
        CHECK_FALSE(info.placement.has_value());
        CHECK(info.shares_memory_with.empty());
    }

    TEST_CASE("resource info reports the placement and which transients overlap it") {
        auto view = make_view();
        view.transients.placements = {
                {.resource = 0, .block = 0, .offset = 0, .size = 256},
                {.resource = 1, .block = 0, .offset = 128, .size = 256},
                {.resource = 2, .block = 0, .offset = 256, .size = 64},
                {.resource = 3, .block = 1, .offset = 0, .size = 256},
        };

        auto const info = describe_resource(view, 0);
        REQUIRE(info.placement.has_value());
        CHECK(info.placement->size == 256);
        CHECK(info.shares_memory_with == std::vector<std::uint32_t>{1});
    }

    TEST_CASE("resource info for an untouched resource is empty") {
        auto const info = describe_resource(make_view(), 99);
        CHECK(info.producers.empty());
        CHECK(info.consumers.empty());
        CHECK_FALSE(info.first_pass.has_value());
    }

    TEST_CASE("common formats are named") {
        CHECK(format_name(VK_FORMAT_R16G16B16A16_SFLOAT) == "RGBA16 SFLOAT");
        CHECK(format_name(VK_FORMAT_D32_SFLOAT) == "D32 SFLOAT");
        CHECK(format_name(VK_FORMAT_ASTC_4x4_UNORM_BLOCK) == "other format");
    }

    TEST_CASE("only single-sample transient images with a sampled view are previewable") {
        auto const sampled = image_descriptor_view_bit(ImageDescriptorView::sampled_2d);
        auto const make = [&](bool imported, VkSampleCountFlagBits samples, std::uint32_t views) {
            return ResourceDesc{
                    .name = "image",
                    .kind = ResourceKind::image,
                    .imported = imported,
                    .transient_image = TransientImageDesc{.samples = samples, .descriptor_views = views},
            };
        };

        CHECK(previewable(make(false, VK_SAMPLE_COUNT_1_BIT, sampled)));
        CHECK_FALSE(previewable(make(true, VK_SAMPLE_COUNT_1_BIT, sampled)));
        CHECK_FALSE(previewable(make(false, VK_SAMPLE_COUNT_4_BIT, sampled)));
        CHECK_FALSE(previewable(make(false, VK_SAMPLE_COUNT_1_BIT, 0)));

        auto buffer = ResourceDesc{.name = "buffer", .kind = ResourceKind::buffer};
        CHECK_FALSE(previewable(buffer));
    }
}
