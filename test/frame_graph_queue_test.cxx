#include <doctest/doctest.h>

#include "frame_graph_test_support.hxx"

using namespace frame_graph;

namespace {

    constexpr auto compute_stage = static_cast<ShaderStages>(ShaderStage::compute);
    constexpr auto fragment_stage = static_cast<ShaderStages>(ShaderStage::fragment);

    auto noop() -> RecordFn { return RecordFn{}; }

    auto single_queue() -> QueueTopology { return QueueTopology{}; }

    auto same_family() -> QueueTopology {
        auto topology = QueueTopology{};
        topology.family = {0, 0};
        topology.queue_index = {0, 1};
        return topology;
    }

    auto dedicated() -> QueueTopology {
        auto topology = QueueTopology{};
        topology.family = {0, 2};
        topology.queue_index = {0, 0};
        return topology;
    }

    auto count_waits(CompiledGraph const &compiled) -> std::size_t {
        auto total = std::size_t{0};
        for (auto const &batch: compiled.batches) {
            total += batch.waits.size();
        }
        return total;
    }

    auto expect_sound(GraphDesc const &graph, CompiledGraph const &compiled, QueueTopology const &topology) -> void {
        auto const problems = test::check_happens_before(graph, compiled, topology);
        for (auto const &problem: problems) {
            MESSAGE(problem);
        }
        CHECK(problems.empty());
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("async shape: graphics, compute, graphics splits into three batches") {
        auto graph = FrameGraph{};
        auto depth = ImageId{};
        graph.add_pass("A", PassType::raster, {}, [&](PassBuilder &p) {
            depth = p.create({.format = VK_FORMAT_D32_SFLOAT, .extent = {4, 4, 1}, .debug_name = "depth"});
            depth = p.write_depth(depth, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("B", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            [[maybe_unused]] auto const s = p.read(depth, Use::sampled, compute_stage);
            p.side_effect();
            return noop();
        });
        graph.add_pass("C", PassType::raster, {}, [&](PassBuilder &p) {
            depth = p.write_depth(depth, LoadOp::clear, StoreOp::store);
            p.side_effect();
            return noop();
        });

        auto const compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        REQUIRE(compiled->batches.size() == 3);
        CHECK(compiled->batches[0].queue == LogicalQueue::graphics);
        CHECK(compiled->batches[1].queue == LogicalQueue::compute);
        CHECK(compiled->batches[2].queue == LogicalQueue::graphics);

        // A -> B is a real ownership transfer (the contents are needed); B -> C is a WAR where C discards, so only a
        // semaphore and a layout transition from UNDEFINED.
        REQUIRE(compiled->transfers.size() == 1);
        auto const &transfer = compiled->transfers.front();
        CHECK(transfer.from == LogicalQueue::graphics);
        CHECK(transfer.to == LogicalQueue::compute);
        CHECK(transfer.release_batch == 0);
        CHECK(transfer.acquire_batch == 1);
        CHECK(transfer.old_layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        CHECK(transfer.new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        REQUIRE(compiled->batches[1].waits.size() == 1);
        CHECK(compiled->batches[1].waits.front().queue == LogicalQueue::graphics);
        CHECK(compiled->batches[1].waits.front().signal_index == 0);
        REQUIRE(compiled->batches[2].waits.size() == 1);
        CHECK(compiled->batches[2].waits.front().queue == LogicalQueue::compute);

        REQUIRE(compiled->batches[2].passes.size() == 1);
        auto const &c_before = compiled->batches[2].passes.front().before;
        REQUIRE(c_before.images.size() == 1);
        CHECK(c_before.images.front().old_layout == VK_IMAGE_LAYOUT_UNDEFINED);
        CHECK(c_before.images.front().op == OwnershipOp::none);

        expect_sound(graph.description(), *compiled, dedicated());
    }

    TEST_CASE("same-family topology keeps the waits and drops the ownership transfers") {
        auto graph = FrameGraph{};
        auto depth = ImageId{};
        graph.add_pass("A", PassType::raster, {}, [&](PassBuilder &p) {
            depth = p.create({.format = VK_FORMAT_D32_SFLOAT, .extent = {4, 4, 1}, .debug_name = "depth"});
            depth = p.write_depth(depth, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("B", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            [[maybe_unused]] auto const s = p.read(depth, Use::sampled, compute_stage);
            p.side_effect();
            return noop();
        });

        auto const compiled = compile(graph, same_family());
        REQUIRE(compiled.has_value());
        CHECK(compiled->transfers.empty());
        CHECK(count_waits(*compiled) == 1);
        for (auto const &batch: compiled->batches) {
            CHECK(batch.acquires.empty());
            CHECK(batch.releases.empty());
        }
        // The layout change is still made, by a plain barrier on the reading queue.
        auto const &b_before = compiled->batches[1].passes.front().before;
        REQUIRE(b_before.images.size() == 1);
        CHECK(b_before.images.front().old_layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        CHECK(b_before.images.front().new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        CHECK(b_before.images.front().src_family == VK_QUEUE_FAMILY_IGNORED);
        expect_sound(graph.description(), *compiled, same_family());
    }

    TEST_CASE("an import whose first use is on compute is released in the prologue batch") {
        auto graph = FrameGraph{};
        auto const image = graph.import_image({
                .entry = {.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                          .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT},
                .exit = {.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT},
                .debug_name = "history",
        });
        graph.add_pass("use", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            [[maybe_unused]] auto const s = p.read(image, Use::sampled, compute_stage);
            p.side_effect();
            return noop();
        });

        auto const compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        auto const &prologue = compiled->batches.front();
        CHECK(prologue.is_prologue);
        CHECK(prologue.passes.empty());
        REQUIRE(prologue.releases.images.size() == 1);
        CHECK(prologue.releases.images.front().op == OwnershipOp::release);
        auto const &compute = compiled->batches[1];
        CHECK(compute.queue == LogicalQueue::compute);
        REQUIRE(compute.waits.size() == 1);
        CHECK(compute.waits.front().signal_index == 0);
        REQUIRE(compute.acquires.images.size() == 1);
        expect_sound(graph.description(), *compiled, dedicated());
    }

    TEST_CASE("an import whose last use is on compute is acquired back on graphics") {
        auto graph = FrameGraph{};
        auto const buffer = graph.import_buffer({
                .entry = {.stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
                .exit = {.stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT},
                .debug_name = "stats",
        });
        graph.add_pass("write", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            [[maybe_unused]] auto const next = p.write(buffer, Use::shader_write, compute_stage);
            return noop();
        });

        auto const compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        auto const &last = compiled->batches.back();
        CHECK(last.queue == LogicalQueue::graphics);
        CHECK(last.signals_render_finished);
        REQUIRE(last.acquires.buffers.size() == 1);
        CHECK(last.acquires.buffers.front().op == OwnershipOp::acquire);
        REQUIRE(last.waits.size() == 1);
        CHECK(last.waits.front().queue == LogicalQueue::compute);
        expect_sound(graph.description(), *compiled, dedicated());
    }

    TEST_CASE("concurrent sharing needs no transfers and orders no reader against another") {
        auto graph = FrameGraph{};
        auto const buffer = graph.import_buffer({
                .sharing = Sharing::concurrent,
                .debug_name = "shared",
        });
        graph.add_pass("compute_a", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            [[maybe_unused]] auto const r = p.read(buffer, Use::shader_read, compute_stage);
            p.side_effect();
            return noop();
        });
        graph.add_pass("compute_b", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            [[maybe_unused]] auto const r = p.read(buffer, Use::shader_read, compute_stage);
            p.side_effect();
            return noop();
        });
        graph.add_pass("graphics_c", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const r = p.read(buffer, Use::shader_read, compute_stage);
            p.side_effect();
            return noop();
        });

        auto const compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        CHECK(compiled->transfers.empty());
        CHECK(count_waits(*compiled) == 0);
        expect_sound(graph.description(), *compiled, dedicated());
    }

    TEST_CASE("an exclusive read-only buffer ping-ponging between queues pays a transfer per switch") {
        auto graph = FrameGraph{};
        auto const buffer = graph.import_buffer({.debug_name = "table"});
        for (auto step = 0; step < 4; ++step) {
            graph.add_pass(std::format("read_{}", step), PassType::compute, {}, [&](PassBuilder &p) {
                p.queue(step % 2 == 1 ? QueueAffinity::compute_required : QueueAffinity::graphics);
                [[maybe_unused]] auto const r = p.read(buffer, Use::shader_read, compute_stage);
                p.side_effect();
                return noop();
            });
        }
        auto const compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        // Three switches inside the frame (G to C, C to G, G to C), plus the epilogue returning the import to graphics.
        CHECK(compiled->transfers.size() == 4);
        expect_sound(graph.description(), *compiled, dedicated());
    }

    TEST_CASE("two transfers into one batch from one queue merge into a single wait") {
        auto graph = FrameGraph{};
        auto first = graph.import_buffer({.debug_name = "first"});
        auto second = graph.import_buffer({.debug_name = "second"});
        graph.add_pass("write_first", PassType::compute, {}, [&](PassBuilder &p) {
            first = p.write(first, Use::shader_write, compute_stage);
            return noop();
        });
        graph.add_pass("write_second", PassType::compute, {}, [&](PassBuilder &p) {
            second = p.write(second, Use::shader_write, compute_stage);
            return noop();
        });
        graph.add_pass("consume", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            [[maybe_unused]] auto const a = p.read(first, Use::shader_read, compute_stage);
            [[maybe_unused]] auto const b = p.read(second, Use::shader_read, compute_stage);
            p.side_effect();
            return noop();
        });
        auto const compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        // Two inside the frame, plus the epilogue returning both imports to graphics.
        CHECK(compiled->transfers.size() == 4);
        auto const consumer = std::ranges::find_if(compiled->batches,
                                                   [](Batch const &b) { return b.queue == LogicalQueue::compute; });
        REQUIRE(consumer != compiled->batches.end());
        REQUIRE(consumer->waits.size() == 1);
        CHECK(consumer->waits.front().signal_index == 1);
        expect_sound(graph.description(), *compiled, dedicated());
    }

    TEST_CASE("compute_preferred is demoted when nothing can overlap it") {
        auto const build = [](bool with_independent_graphics, QueueAffinity affinity) {
            auto graph = FrameGraph{};
            auto image = ImageId{};
            graph.add_pass("produce", PassType::raster, {}, [&](PassBuilder &p) {
                image = p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "image"});
                image = p.color(image, LoadOp::clear, StoreOp::store);
                return noop();
            });
            graph.add_pass("async_candidate", PassType::compute, {}, [&](PassBuilder &p) {
                p.queue(affinity);
                [[maybe_unused]] auto const s = p.read(image, Use::sampled, compute_stage);
                p.side_effect();
                return noop();
            });
            if (with_independent_graphics) {
                graph.add_pass("independent", PassType::raster, {}, [&](PassBuilder &p) {
                    auto other =
                            p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "other"});
                    [[maybe_unused]] auto const out = p.color(other, LoadOp::clear, StoreOp::store);
                    p.side_effect();
                    return noop();
                });
            }
            return graph;
        };

        auto alone = build(false, QueueAffinity::compute_preferred);
        auto const demoted = compile(alone, dedicated());
        REQUIRE(demoted.has_value());
        CHECK(demoted->pass_queue[1] == LogicalQueue::graphics);
        CHECK(demoted->transfers.empty());

        auto overlapped = build(true, QueueAffinity::compute_preferred);
        auto const kept = compile(overlapped, dedicated());
        REQUIRE(kept.has_value());
        CHECK(kept->pass_queue[1] == LogicalQueue::compute);

        auto required = build(false, QueueAffinity::compute_required);
        auto const forced = compile(required, dedicated());
        REQUIRE(forced.has_value());
        CHECK(forced->pass_queue[1] == LogicalQueue::compute);

        auto const off = compile(overlapped, dedicated(), {.async_compute = false});
        REQUIRE(off.has_value());
        CHECK(off->pass_queue[1] == LogicalQueue::graphics);
        CHECK(off->batches.size() == 1);
    }

    TEST_CASE("swapchain wiring on a multi-queue plan") {
        auto graph = FrameGraph{};
        auto swapchain = graph.import_image({
                .exit = {.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
                .swapchain = true,
                .debug_name = "swapchain",
        });
        auto data = graph.import_buffer({.debug_name = "data"});
        graph.add_pass("prepare", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            data = p.write(data, Use::shader_write, compute_stage);
            return noop();
        });
        graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const r = p.read(data, Use::shader_read, fragment_stage);
            swapchain = p.color(swapchain, LoadOp::clear, StoreOp::store);
            return noop();
        });

        auto const compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        auto acquires = 0;
        auto finished = 0;
        for (auto const &batch: compiled->batches) {
            acquires += batch.waits_swapchain_acquire ? 1 : 0;
            finished += batch.signals_render_finished ? 1 : 0;
            if (batch.waits_swapchain_acquire) {
                REQUIRE(batch.passes.size() == 1);
                CHECK(graph.description().passes[batch.passes.front().pass].name == "draw");
            }
        }
        CHECK(acquires == 1);
        CHECK(finished == 1);
        CHECK(compiled->batches.back().signals_render_finished);
        expect_sound(graph.description(), *compiled, dedicated());
    }

    TEST_CASE("timestamp slots are contiguous per queue and skip culled passes") {
        auto graph = FrameGraph{};
        auto buffer = graph.import_buffer({.debug_name = "b"});
        for (auto index = 0; index < 4; ++index) {
            graph.add_pass(std::format("p{}", index), PassType::compute, {}, [&](PassBuilder &p) {
                p.queue(index % 2 == 0 ? QueueAffinity::graphics : QueueAffinity::compute_required);
                buffer = p.write(buffer, Use::shader_read_write, compute_stage);
                return noop();
            });
        }
        auto const compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        CHECK(compiled->timestamp_passes[0] == std::vector<std::uint32_t>{0, 2});
        CHECK(compiled->timestamp_passes[1] == std::vector<std::uint32_t>{1, 3});
        for (auto const &batch: compiled->batches) {
            for (auto const &pass: batch.passes) {
                auto const &slots = compiled->timestamp_passes[static_cast<std::size_t>(batch.queue)];
                CHECK(slots[pass.timestamp_slot] == pass.pass);
            }
        }
        expect_sound(graph.description(), *compiled, dedicated());
    }

    TEST_CASE("one code path: dedicated topology with async off matches the single queue plan") {
        auto const build = [] {
            auto graph = FrameGraph{};
            auto image = ImageId{};
            graph.add_pass("A", PassType::raster, {}, [&](PassBuilder &p) {
                image = p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "image"});
                image = p.color(image, LoadOp::clear, StoreOp::store);
                return noop();
            });
            graph.add_pass("B", PassType::compute, {}, [&](PassBuilder &p) {
                p.queue(QueueAffinity::compute_required);
                [[maybe_unused]] auto const s = p.read(image, Use::sampled, compute_stage);
                p.side_effect();
                return noop();
            });
            return graph;
        };
        auto first = build();
        auto second = build();
        auto const single = compile(first, single_queue());
        auto const off = compile(second, dedicated(), {.async_compute = false});
        REQUIRE(single.has_value());
        REQUIRE(off.has_value());
        REQUIRE(single->batches.size() == 1);
        REQUIRE(off->batches.size() == 1);
        auto const &a = single->batches.front().passes;
        auto const &b = off->batches.front().passes;
        REQUIRE(a.size() == b.size());
        for (auto index = std::size_t{0}; index < a.size(); ++index) {
            CHECK(a[index].before.images == b[index].before.images);
            CHECK(a[index].before.buffers == b[index].before.buffers);
        }
        CHECK(count_waits(*off) == 0);
        CHECK(off->transfers.empty());
    }

    TEST_CASE("random graphs: the plan is sound on every topology") {
        struct Topology {
            char const *name;
            QueueTopology topology;
        };
        auto const topologies = std::array{
                Topology{"single", single_queue()},
                Topology{"same_family", same_family()},
                Topology{"dedicated", dedicated()},
        };

        for (auto seed = std::uint32_t{1}; seed <= 500; ++seed) {
            for (auto const &candidate: topologies) {
                auto graph = FrameGraph{};
                test::build_random_graph(graph, seed);

                auto const compiled = compile(graph, candidate.topology);
                if (!compiled.has_value()) {
                    MESSAGE("seed " << seed << " topology " << candidate.name << ": "
                                    << std::format("{}", compiled.error()));
                }
                REQUIRE(compiled.has_value());

                auto const problems = test::check_happens_before(graph.description(), *compiled, candidate.topology);
                for (auto const &problem: problems) {
                    MESSAGE(std::format("seed {} topology {}: {}", seed, candidate.name, problem));
                }
                if (!problems.empty()) {
                    MESSAGE(test::describe(graph.description(), *compiled));
                }
                REQUIRE(problems.empty());

                if (std::string_view{candidate.name} == "single") {
                    CHECK(count_waits(*compiled) == 0);
                    CHECK(compiled->transfers.empty());
                }
                if (std::string_view{candidate.name} == "same_family") {
                    CHECK(compiled->transfers.empty());
                }
            }
        }
    }
}
