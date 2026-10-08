#include <doctest/doctest.h>

#include "frame_graph_test_support.hxx"

using namespace frame_graph;

namespace {

    constexpr auto compute_stage = static_cast<ShaderStages>(ShaderStage::compute);
    constexpr auto geometry_stages = ShaderStage::vertex | ShaderStage::task | ShaderStage::mesh;

    auto noop() -> RecordFn { return RecordFn{}; }

    auto make_skin_chain(bool consumers_declare_scratch = true) -> FrameGraph {
        auto graph = FrameGraph{};
        auto upload_source = graph.import_buffer({.read_only = true, .debug_name = "skin_upload"});
        auto input = graph.import_buffer({.debug_name = "skin_input"});
        auto scratch = graph.import_buffer({.debug_name = "skin_scratch"});
        auto target = graph.import_image({
                .exit = {.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
                .swapchain = true,
                .debug_name = "target",
        });
        auto shadow = graph.import_image({.debug_name = "shadow_atlas"});
        auto depth = ImageId{};

        graph.add_pass("skin_upload", PassType::transfer, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const s = p.read(upload_source, Use::transfer_read);
            input = p.write_discard(input, Use::transfer_write);
            return noop();
        });
        graph.add_pass("skin", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const i = p.read(input, Use::shader_read, compute_stage);
            scratch = p.write_discard(scratch, Use::shader_write, compute_stage);
            return noop();
        });
        graph.add_pass("shadow_pass", PassType::raster, {}, [&](PassBuilder &p) {
            if (consumers_declare_scratch) {
                [[maybe_unused]] auto const v = p.read(scratch, Use::shader_read, geometry_stages);
            }
            shadow = p.write_depth(shadow, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("early_prepass", PassType::raster, {}, [&](PassBuilder &p) {
            if (consumers_declare_scratch) {
                [[maybe_unused]] auto const v = p.read(scratch, Use::shader_read, geometry_stages);
            }
            depth = p.create({.format = VK_FORMAT_D32_SFLOAT, .extent = {4, 4, 1}, .debug_name = "depth"});
            depth = p.write_depth(depth, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("forward", PassType::raster, {}, [&](PassBuilder &p) {
            if (consumers_declare_scratch) {
                [[maybe_unused]] auto const v = p.read(scratch, Use::shader_read, geometry_stages | ShaderStage::fragment);
            }
            [[maybe_unused]] auto const d = p.read(depth, Use::sampled, static_cast<ShaderStages>(ShaderStage::task));
            [[maybe_unused]] auto const s = p.read(shadow, Use::sampled, static_cast<ShaderStages>(ShaderStage::fragment));
            target = p.color(target, LoadOp::clear, StoreOp::store);
            return noop();
        });
        return graph;
    }

    auto resource_named(GraphDesc const &graph, std::string_view name) -> std::uint32_t {
        for (auto index = std::size_t{0}; index < graph.resources.size(); ++index) {
            if (graph.resources[index].name == name) {
                return static_cast<std::uint32_t>(index);
            }
        }
        return 0;
    }

    auto barriers_before(GraphDesc const &desc, CompiledGraph const &compiled, std::string_view name)
            -> std::vector<BufferBarrier> {
        for (auto const &batch: compiled.batches) {
            for (auto const &pass: batch.passes) {
                if (desc.passes[pass.pass].name == name) {
                    return pass.before.buffers;
                }
            }
        }
        return {};
    }

    auto has_barrier(std::vector<BufferBarrier> const &barriers, std::uint32_t resource, VkPipelineStageFlags2 src,
                     VkPipelineStageFlags2 dst, VkAccessFlags2 dst_access) -> bool {
        for (auto const &barrier: barriers) {
            if (barrier.resource == resource && (barrier.src_stages & src) != 0 && (barrier.dst_stages & dst) != 0 &&
                (barrier.src_access & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) != 0 &&
                (barrier.dst_access & dst_access) != 0) {
                return true;
            }
        }
        return false;
    }

    constexpr auto raster_geometry_stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    constexpr auto read_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

}

TEST_SUITE("unit") {
    TEST_CASE("skinning: the first raster reader of the scratch waits for the compute write") {
        auto graph = make_skin_chain();
        auto const compiled = compile(graph, QueueTopology{});
        REQUIRE(compiled.has_value());
        auto const &desc = graph.description();
        auto const scratch = resource_named(desc, "skin_scratch");

        CHECK(has_barrier(barriers_before(desc, *compiled, "shadow_pass"), scratch,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, raster_geometry_stages, read_access));
    }

    TEST_CASE("skinning: the palette upload is ordered before the skin pass") {
        auto graph = make_skin_chain();
        auto const compiled = compile(graph, QueueTopology{});
        REQUIRE(compiled.has_value());
        auto const &desc = graph.description();
        auto const input = resource_named(desc, "skin_input");

        auto found = false;
        for (auto const &barrier: barriers_before(desc, *compiled, "skin")) {
            found = found || (barrier.resource == input && (barrier.src_stages & (VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT)) != 0 &&
                              (barrier.dst_stages & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) != 0);
        }
        CHECK(found);
    }

    TEST_CASE("skinning: an undeclared consumer gets no barrier, which is why every pass declares the read") {
        auto graph = make_skin_chain(false);
        auto const compiled = compile(graph, QueueTopology{});
        REQUIRE(compiled.has_value());
        auto const &desc = graph.description();
        auto const scratch = resource_named(desc, "skin_scratch");

        for (auto const *name: {"shadow_pass", "early_prepass", "forward"}) {
            CHECK_FALSE(has_barrier(barriers_before(desc, *compiled, name), scratch,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, raster_geometry_stages, read_access));
        }
    }

    TEST_CASE("skinning: the chain is sound on every topology and scheduler") {
        auto same_family = QueueTopology{};
        same_family.family = {0, 0};
        same_family.queue_index = {0, 1};
        auto dedicated = QueueTopology{};
        dedicated.family = {0, 2};
        for (auto const &topology: {QueueTopology{}, same_family, dedicated}) {
            for (auto const mode: {SchedulerMode::declaration_order, SchedulerMode::overlap}) {
                auto graph = make_skin_chain();
                auto const compiled = compile(graph, topology, {.scheduler = mode});
                REQUIRE(compiled.has_value());
                auto const problems = test::check_happens_before(graph.description(), *compiled, topology);
                for (auto const &problem: problems) {
                    MESSAGE(problem);
                }
                CHECK(problems.empty());
            }
        }
    }
}
