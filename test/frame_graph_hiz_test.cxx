#include <doctest/doctest.h>

#include "frame_graph_test_support.hxx"

using namespace frame_graph;

namespace {

    constexpr auto compute_stage = static_cast<ShaderStages>(ShaderStage::compute);
    constexpr auto task_stage = static_cast<ShaderStages>(ShaderStage::task);

    auto noop() -> RecordFn { return RecordFn{}; }

    auto dedicated() -> QueueTopology {
        auto topology = QueueTopology{};
        topology.family = {0, 2};
        return topology;
    }

    auto make_occlusion_chain() -> FrameGraph {
        auto graph = FrameGraph{};
        constexpr auto hiz_state = ResourceState{
                .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        };
        auto hiz = graph.import_image({.entry = hiz_state, .exit = hiz_state, .debug_name = "hiz"});
        auto visible = graph.import_buffer({.debug_name = "visible"});
        auto target = graph.import_image({
                .exit = {.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
                .swapchain = true,
                .debug_name = "target",
        });
        auto depth = ImageId{};
        auto resolved = ImageId{};

        graph.add_pass("main_cs", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const h = p.read(hiz, Use::sampled, compute_stage);
            visible = p.write(visible, Use::shader_write, compute_stage);
            return noop();
        });
        graph.add_pass("early_prepass", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const v = p.read(visible, Use::indirect_read);
            depth = p.create({.format = VK_FORMAT_D32_SFLOAT, .extent = {4, 4, 1}, .debug_name = "depth"});
            depth = p.write_depth(depth, LoadOp::clear, StoreOp::store);
            resolved = p.create({.format = VK_FORMAT_R32_SFLOAT, .extent = {4, 4, 1}, .debug_name = "resolved"});
            resolved = p.write(resolved, Use::depth_resolve);
            return noop();
        });
        graph.add_pass("hiz_build", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            [[maybe_unused]] auto const r = p.read(resolved, Use::sampled, compute_stage);
            hiz = p.write(hiz, Use::storage_write, compute_stage);
            return noop();
        });
        graph.add_pass("late_cs", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            [[maybe_unused]] auto const h = p.read(hiz, Use::sampled, compute_stage);
            visible = p.write(visible, Use::shader_read_write, compute_stage);
            return noop();
        });
        graph.add_pass("late_prepass", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const v = p.read(visible, Use::indirect_read);
            depth = p.write_depth(depth, LoadOp::load, StoreOp::store);
            resolved = p.write(resolved, Use::depth_resolve);
            return noop();
        });
        graph.add_pass("forward", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const v = p.read(visible, Use::indirect_read);
            [[maybe_unused]] auto const d = p.read(depth, Use::sampled, task_stage);
            target = p.color(target, LoadOp::clear, StoreOp::store);
            return noop();
        });
        return graph;
    }

    auto pass_named(GraphDesc const &graph, std::string_view name) -> std::uint32_t {
        for (auto index = std::size_t{0}; index < graph.passes.size(); ++index) {
            if (graph.passes[index].name == name) {
                return static_cast<std::uint32_t>(index);
            }
        }
        return 0;
    }

    auto resource_named(GraphDesc const &graph, std::string_view name) -> std::uint32_t {
        for (auto index = std::size_t{0}; index < graph.resources.size(); ++index) {
            if (graph.resources[index].name == name) {
                return static_cast<std::uint32_t>(index);
            }
        }
        return 0;
    }

}

TEST_SUITE("unit") {
    TEST_CASE("the occlusion chain keeps Hi-Z owned by graphics across the frame boundary") {
        auto graph = make_occlusion_chain();
        auto const compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        auto const &desc = graph.description();
        auto const hiz = resource_named(desc, "hiz");

        CHECK(compiled->pass_queue[pass_named(desc, "hiz_build")] == LogicalQueue::compute);
        CHECK(compiled->pass_queue[pass_named(desc, "late_cs")] == LogicalQueue::compute);

        auto to_compute = 0;
        auto to_graphics = 0;
        for (auto const &transfer: compiled->transfers) {
            if (transfer.resource != hiz) {
                continue;
            }
            to_compute += transfer.to == LogicalQueue::compute ? 1 : 0;
            to_graphics += transfer.to == LogicalQueue::graphics ? 1 : 0;
        }
        CHECK(to_compute == 0);
        CHECK(to_graphics == 1);

        auto const &last = compiled->batches.back();
        CHECK(last.queue == LogicalQueue::graphics);
        CHECK(last.signals_render_finished);
        CHECK_FALSE(last.waits.empty());
        auto has_hiz_acquire = false;
        for (auto const &barrier: last.acquires.images) {
            has_hiz_acquire = has_hiz_acquire || (barrier.resource == hiz && barrier.op == OwnershipOp::acquire &&
                                                  barrier.new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        CHECK(has_hiz_acquire);

        auto const &main_cs = compiled->batches.front().passes.empty() ? compiled->batches[1].passes.front()
                                                                       : compiled->batches.front().passes.front();
        REQUIRE(desc.passes[main_cs.pass].name == "main_cs");
        for (auto const &barrier: main_cs.before.images) {
            CHECK(barrier.resource != hiz);
        }

        auto const problems = test::check_happens_before(desc, *compiled, dedicated());
        for (auto const &problem: problems) {
            MESSAGE(problem);
        }
        CHECK(problems.empty());
    }

    TEST_CASE("the occlusion chain is sound on every topology and scheduler") {
        auto same_family = QueueTopology{};
        same_family.family = {0, 0};
        same_family.queue_index = {0, 1};
        for (auto const &topology: {QueueTopology{}, same_family, dedicated()}) {
            for (auto const mode: {SchedulerMode::declaration_order, SchedulerMode::overlap}) {
                auto graph = make_occlusion_chain();
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

    TEST_CASE("two consecutive frames compile to the same plan") {
        auto first = make_occlusion_chain();
        auto second = make_occlusion_chain();
        auto const a = compile(first, dedicated());
        auto const b = compile(second, dedicated());
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        CHECK(a->hash == b->hash);
        CHECK(a->schedule == b->schedule);
        REQUIRE(a->batches.size() == b->batches.size());
        for (auto index = std::size_t{0}; index < a->batches.size(); ++index) {
            CHECK(a->batches[index].waits == b->batches[index].waits);
        }
    }
}
