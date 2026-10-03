#include <doctest/doctest.h>

#include "frame_graph_test_support.hxx"

using namespace frame_graph;

namespace {

    constexpr auto compute_stage = static_cast<ShaderStages>(ShaderStage::compute);

    auto noop() -> RecordFn { return RecordFn{}; }

    auto dedicated() -> QueueTopology {
        auto topology = QueueTopology{};
        topology.family = {0, 2};
        return topology;
    }

    auto same_family() -> QueueTopology {
        auto topology = QueueTopology{};
        topology.family = {0, 0};
        topology.queue_index = {0, 1};
        return topology;
    }

    auto position_of(CompiledGraph const &compiled, std::uint32_t pass) -> std::size_t {
        return static_cast<std::size_t>(std::ranges::find(compiled.schedule, pass) - compiled.schedule.begin());
    }

    auto count_waits(CompiledGraph const &compiled) -> std::size_t {
        auto total = std::size_t{0};
        for (auto const &batch: compiled.batches) {
            total += batch.waits.size();
        }
        return total;
    }

    // produce (compute) -> consume (graphics), with an independent graphics pass declared after the consumer.
    auto make_overlap_graph() -> FrameGraph {
        auto graph = FrameGraph{};
        auto data = graph.import_buffer({.debug_name = "data"});
        auto other = graph.import_buffer({.debug_name = "other"});
        graph.add_pass("produce", PassType::compute, {}, [&](PassBuilder &p) {
            p.queue(QueueAffinity::compute_required);
            data = p.write(data, Use::shader_write, compute_stage);
            return noop();
        });
        graph.add_pass("consume", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const r = p.read(data, Use::shader_read, compute_stage);
            p.side_effect();
            return noop();
        });
        graph.add_pass("independent", PassType::compute, {}, [&](PassBuilder &p) {
            other = p.write(other, Use::shader_write, compute_stage);
            return noop();
        });
        return graph;
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("declaration order is the schedule") {
        auto graph = make_overlap_graph();
        auto const compiled = compile(graph, dedicated());
        REQUIRE(compiled.has_value());
        CHECK(compiled->schedule == std::vector<std::uint32_t>{0, 1, 2});
    }

    TEST_CASE("the scheduler hoists an independent pass into the gap behind an async producer") {
        auto graph = make_overlap_graph();
        auto const declared = compile(graph, dedicated());
        auto const overlapped = compile(graph, dedicated(), {.scheduler = SchedulerMode::overlap});
        REQUIRE(declared.has_value());
        REQUIRE(overlapped.has_value());

        // The consumer waits for the compute producer; the independent pass fills the gap instead of trailing it.
        CHECK(overlapped->schedule == std::vector<std::uint32_t>{0, 2, 1});
        CHECK(position_of(*overlapped, 2) < position_of(*overlapped, 1));
        CHECK(count_waits(*overlapped) <= count_waits(*declared));
        auto const problems = test::check_happens_before(graph.description(), *overlapped, dedicated());
        CHECK(problems.empty());
    }

    TEST_CASE("pinned and legacy passes never move and are never crossed") {
        // fence_kind -1 is the control without a fence: "independent" hoists ahead of "consume". With a fence declared
        // between them, hoisting would cross it, so the declared order stays.
        for (auto const fence_kind: {-1, 0, 1}) {
            auto graph = FrameGraph{};
            auto data = graph.import_buffer({.debug_name = "data"});
            auto other = graph.import_buffer({.debug_name = "other"});
            graph.add_pass("produce", PassType::compute, {}, [&](PassBuilder &p) {
                p.queue(QueueAffinity::compute_required);
                data = p.write(data, Use::shader_write, compute_stage);
                return noop();
            });
            graph.add_pass("consume", PassType::compute, {}, [&](PassBuilder &p) {
                [[maybe_unused]] auto const r = p.read(data, Use::shader_read, compute_stage);
                p.side_effect();
                return noop();
            });
            if (fence_kind >= 0) {
                graph.add_pass("fence", PassType::compute, {}, [&](PassBuilder &p) {
                    if (fence_kind == 0) {
                        p.pinned();
                    } else {
                        p.legacy();
                    }
                    return noop();
                });
            }
            graph.add_pass("independent", PassType::compute, {}, [&](PassBuilder &p) {
                other = p.write(other, Use::shader_write, compute_stage);
                return noop();
            });

            auto const compiled = compile(graph, dedicated(), {.scheduler = SchedulerMode::overlap});
            REQUIRE(compiled.has_value());
            if (fence_kind < 0) {
                CHECK(compiled->schedule == std::vector<std::uint32_t>{0, 2, 1});
            } else {
                CHECK(compiled->schedule == std::vector<std::uint32_t>{0, 1, 2, 3});
            }
        }
    }

    TEST_CASE("scheduling is deterministic") {
        auto first = make_overlap_graph();
        auto second = make_overlap_graph();
        auto const a = compile(first, dedicated(), {.scheduler = SchedulerMode::overlap});
        auto const b = compile(second, dedicated(), {.scheduler = SchedulerMode::overlap});
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        CHECK(a->schedule == b->schedule);
        CHECK(a->hash == b->hash);
    }

    TEST_CASE("random graphs: the overlap schedule is a topological order and never adds waits") {
        for (auto seed = std::uint32_t{1}; seed <= 500; ++seed) {
            for (auto const &topology: {dedicated(), same_family()}) {
                auto graph = FrameGraph{};
                test::build_random_graph(graph, seed, 10);

                auto const declared = compile(graph, topology);
                auto const overlapped = compile(graph, topology, {.scheduler = SchedulerMode::overlap});
                REQUIRE(declared.has_value());
                REQUIRE(overlapped.has_value());

                // Same passes, different order.
                auto sorted = overlapped->schedule;
                std::ranges::sort(sorted);
                auto expected = declared->schedule;
                std::ranges::sort(expected);
                REQUIRE(sorted == expected);

                // Conflicting passes keep their relative order, as do fences against everything.
                auto const &desc = graph.description();
                auto const live = std::vector<bool>(declared->pass_culled.begin(), declared->pass_culled.end());
                auto live_mask = std::vector<bool>(live.size());
                for (auto index = std::size_t{0}; index < live.size(); ++index) {
                    live_mask[index] = !declared->pass_culled[index];
                }
                auto const successors = dependency_successors(desc, live_mask);
                for (auto source = std::size_t{0}; source < successors.size(); ++source) {
                    for (auto const target: successors[source]) {
                        REQUIRE(position_of(*overlapped, static_cast<std::uint32_t>(source)) <
                                position_of(*overlapped, static_cast<std::uint32_t>(target)));
                    }
                }
                for (auto const fence: declared->schedule) {
                    if (!desc.passes[fence].pinned && !desc.passes[fence].legacy) {
                        continue;
                    }
                    for (auto const other: declared->schedule) {
                        if (other < fence) {
                            REQUIRE(position_of(*overlapped, other) < position_of(*overlapped, fence));
                        }
                        if (other > fence) {
                            REQUIRE(position_of(*overlapped, fence) < position_of(*overlapped, other));
                        }
                    }
                }

                CHECK(count_waits(*overlapped) <= count_waits(*declared));

                auto const problems = test::check_happens_before(desc, *overlapped, topology);
                for (auto const &problem: problems) {
                    MESSAGE(std::format("seed {}: {}", seed, problem));
                }
                REQUIRE(problems.empty());
            }
        }
    }
}
