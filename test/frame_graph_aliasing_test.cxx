#include <doctest/doctest.h>

#include <algorithm>

#include "frame_graph_test_support.hxx"
#include "rendering/frame_graph/aliasing.hxx"
#include "rendering/frame_graph/compiler.hxx"
#include "rendering/frame_graph/describe.hxx"

using namespace frame_graph;

namespace {

    constexpr auto compute_stage = static_cast<ShaderStages>(ShaderStage::compute);

    auto noop() -> RecordFn { return RecordFn{}; }

    auto single_queue() -> QueueTopology { return QueueTopology{}; }

    auto dedicated() -> QueueTopology {
        auto topology = QueueTopology{};
        topology.family = {0, 2};
        topology.queue_index = {0, 0};
        return topology;
    }

    auto same_family() -> QueueTopology {
        auto topology = QueueTopology{};
        topology.family = {0, 0};
        topology.queue_index = {0, 1};
        return topology;
    }

    auto transient(PassBuilder &pass, char const *name) -> ImageId {
        return pass.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = name});
    }

    auto uniform(GraphDesc const &graph, std::uint64_t size, std::uint64_t alignment = 1, std::uint32_t type_bits = ~0U)
            -> std::vector<MemoryRequirement> {
        auto requirements = std::vector<MemoryRequirement>(graph.resources.size());
        for (auto index = std::size_t{0}; index < graph.resources.size(); ++index) {
            if (graph.resources[index].transient_image.has_value()) {
                requirements[index] =
                        MemoryRequirement{.size = size, .alignment = alignment, .memory_type_bits = type_bits};
            }
        }
        return requirements;
    }

    auto resource_named(GraphDesc const &graph, std::string_view name) -> std::uint32_t {
        for (auto index = std::uint32_t{0}; index < graph.resources.size(); ++index) {
            if (graph.resources[index].name == name) {
                return index;
            }
        }
        FAIL("no resource named " << name);
        return 0;
    }

    auto disjoint_chains(FrameGraph &graph) -> void {
        auto first = ImageId{};
        auto second = ImageId{};
        graph.add_pass("write_first", PassType::compute, {}, [&](PassBuilder &pass) {
            pass.side_effect();
            first = pass.write(transient(pass, "first"), Use::storage_write, compute_stage);
            return noop();
        });
        graph.add_pass("read_first", PassType::compute, {}, [&](PassBuilder &pass) {
            pass.side_effect();
            [[maybe_unused]] auto const read = pass.read(first, Use::sampled, compute_stage);
            return noop();
        });
        graph.add_pass("write_second", PassType::compute, {}, [&](PassBuilder &pass) {
            pass.side_effect();
            second = pass.write(transient(pass, "second"), Use::storage_write, compute_stage);
            return noop();
        });
        graph.add_pass("read_second", PassType::compute, {}, [&](PassBuilder &pass) {
            pass.side_effect();
            [[maybe_unused]] auto const read = pass.read(second, Use::sampled, compute_stage);
            return noop();
        });
    }

    auto overlapping_pair(FrameGraph &graph) -> void {
        auto first = ImageId{};
        auto second = ImageId{};
        graph.add_pass("write_first", PassType::compute, {}, [&](PassBuilder &pass) {
            pass.side_effect();
            first = pass.write(transient(pass, "first"), Use::storage_write, compute_stage);
            return noop();
        });
        graph.add_pass("write_second", PassType::compute, {}, [&](PassBuilder &pass) {
            pass.side_effect();
            second = pass.write(transient(pass, "second"), Use::storage_write, compute_stage);
            return noop();
        });
        graph.add_pass("read_both", PassType::compute, {}, [&](PassBuilder &pass) {
            pass.side_effect();
            [[maybe_unused]] auto const a = pass.read(first, Use::sampled, compute_stage);
            [[maybe_unused]] auto const b = pass.read(second, Use::sampled, compute_stage);
            return noop();
        });
    }

    auto two_queues(FrameGraph &graph, bool ordered) -> void {
        auto const handoff = graph.import_buffer({.entry = {}, .exit = {}, .debug_name = "handoff"});
        auto buffer = handoff;
        graph.add_pass("graphics", PassType::compute, {}, [&](PassBuilder &pass) {
            pass.side_effect();
            pass.queue(QueueAffinity::graphics);
            [[maybe_unused]] auto const first = pass.write(transient(pass, "first"), Use::storage_write, compute_stage);
            if (ordered) {
                buffer = pass.write(buffer, Use::shader_write, compute_stage);
            }
            return noop();
        });
        graph.add_pass("compute", PassType::compute, {}, [&](PassBuilder &pass) {
            pass.side_effect();
            pass.queue(QueueAffinity::compute_required);
            if (ordered) {
                [[maybe_unused]] auto const in = pass.read(buffer, Use::shader_read, compute_stage);
            }
            [[maybe_unused]] auto const second =
                    pass.write(transient(pass, "second"), Use::storage_write, compute_stage);
            return noop();
        });
    }

    struct Reference {
        std::vector<std::vector<bool>> batch_before;
        std::vector<std::pair<std::int64_t, std::size_t>> location;

        Reference(GraphDesc const &graph, CompiledGraph const &compiled) {
            auto const count = compiled.batches.size();
            location.assign(graph.passes.size(), {-1, 0});
            for (auto batch = std::size_t{0}; batch < count; ++batch) {
                for (auto position = std::size_t{0}; position < compiled.batches[batch].passes.size(); ++position) {
                    location[compiled.batches[batch].passes[position].pass] = {static_cast<std::int64_t>(batch),
                                                                               position};
                }
            }
            batch_before.assign(count, std::vector<bool>(count, false));
            for (auto changed = true; changed;) {
                changed = false;
                auto const set = [&](std::size_t a, std::size_t b) {
                    if (!batch_before[a][b]) {
                        batch_before[a][b] = true;
                        changed = true;
                    }
                };
                for (auto b = std::size_t{0}; b < count; ++b) {
                    for (auto a = std::size_t{0}; a < b; ++a) {
                        if (compiled.batches[a].queue == compiled.batches[b].queue) {
                            set(a, b);
                        }
                    }
                    for (auto const &wait: compiled.batches[b].waits) {
                        for (auto a = std::size_t{0}; a < count; ++a) {
                            if (compiled.batches[a].queue == wait.queue &&
                                compiled.batches[a].signal_index <= wait.signal_index) {
                                set(a, b);
                            }
                        }
                    }
                    for (auto a = std::size_t{0}; a < count; ++a) {
                        for (auto c = std::size_t{0}; c < count; ++c) {
                            if (batch_before[a][b] && batch_before[b][c]) {
                                set(a, c);
                            }
                        }
                    }
                }
            }
        }

        [[nodiscard]] auto before(std::uint32_t a, std::uint32_t b) const -> bool {
            auto const [batch_a, position_a] = location[a];
            auto const [batch_b, position_b] = location[b];
            if (batch_a < 0 || batch_b < 0) {
                return false;
            }
            return batch_a == batch_b
                           ? position_a < position_b
                           : batch_before[static_cast<std::size_t>(batch_a)][static_cast<std::size_t>(batch_b)];
        }
    };

    auto passes_using(GraphDesc const &graph, CompiledGraph const &compiled, std::uint32_t resource)
            -> std::vector<std::uint32_t> {
        auto passes = std::vector<std::uint32_t>{};
        for (auto pass = std::uint32_t{0}; pass < graph.passes.size(); ++pass) {
            if (compiled.pass_culled[pass]) {
                continue;
            }
            if (std::ranges::any_of(graph.passes[pass].accesses,
                                    [&](AccessDesc const &a) { return a.resource == resource; })) {
                passes.push_back(pass);
            }
        }
        return passes;
    }

    auto offsets_overlap(TransientPlacement const &a, TransientPlacement const &b) -> bool {
        return a.block == b.block && a.offset < b.offset + b.size && b.offset < a.offset + a.size;
    }

}

TEST_SUITE("unit") {
    TEST_CASE("transients with disjoint lifetimes on one queue share memory") {
        auto graph = FrameGraph{};
        disjoint_chains(graph);
        auto const compiled = compile(graph, single_queue());
        REQUIRE(compiled.has_value());

        auto const &desc = graph.description();
        auto const plan = plan_transients(desc, *compiled, uniform(desc, 256), true);

        auto const *first = plan.placement_of(resource_named(desc, "first"));
        auto const *second = plan.placement_of(resource_named(desc, "second"));
        REQUIRE(first != nullptr);
        REQUIRE(second != nullptr);
        CHECK(first->block == second->block);
        CHECK(first->offset == second->offset);
        CHECK(plan.blocks.size() == 1);
        CHECK(plan.total_bytes == 256);
        CHECK(plan.unaliased_bytes == 512);
    }

    TEST_CASE("transients alive at the same pass do not share memory") {
        auto graph = FrameGraph{};
        overlapping_pair(graph);
        auto const compiled = compile(graph, single_queue());
        REQUIRE(compiled.has_value());

        auto const &desc = graph.description();
        auto const plan = plan_transients(desc, *compiled, uniform(desc, 256), true);

        auto const *first = plan.placement_of(resource_named(desc, "first"));
        auto const *second = plan.placement_of(resource_named(desc, "second"));
        REQUIRE(first != nullptr);
        REQUIRE(second != nullptr);
        CHECK_FALSE(offsets_overlap(*first, *second));
        CHECK(plan.total_bytes == 512);
        CHECK(plan.barriers.empty());
    }

    TEST_CASE("with aliasing off every transient has a block of its own") {
        auto graph = FrameGraph{};
        disjoint_chains(graph);
        auto const compiled = compile(graph, single_queue());
        REQUIRE(compiled.has_value());

        auto const &desc = graph.description();
        auto const plan = plan_transients(desc, *compiled, uniform(desc, 256), false);

        CHECK(plan.blocks.size() == 2);
        CHECK(plan.total_bytes == plan.unaliased_bytes);
        CHECK(plan.barriers.empty());
        for (auto const &placement: plan.placements) {
            CHECK(placement.offset == 0);
        }
    }

    TEST_CASE("handing memory on gets a barrier before the newcomer's first pass") {
        auto graph = FrameGraph{};
        disjoint_chains(graph);
        auto const compiled = compile(graph, single_queue());
        REQUIRE(compiled.has_value());

        auto const &desc = graph.description();
        auto const plan = plan_transients(desc, *compiled, uniform(desc, 256), true);

        REQUIRE(plan.barriers.size() == 1);
        CHECK(plan.barriers[0].pass == 2);
        CHECK((plan.barriers[0].barrier.src_stages & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) != 0);
        CHECK((plan.barriers[0].barrier.src_access & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) != 0);
        CHECK((plan.barriers[0].barrier.dst_access & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) != 0);
    }

    TEST_CASE("independent work on two queues never shares memory, ordered work does") {
        {
            auto graph = FrameGraph{};
            two_queues(graph, false);
            auto const compiled = compile(graph, dedicated());
            REQUIRE(compiled.has_value());

            auto const &desc = graph.description();
            auto const plan = plan_transients(desc, *compiled, uniform(desc, 256), true);
            auto const *first = plan.placement_of(resource_named(desc, "first"));
            auto const *second = plan.placement_of(resource_named(desc, "second"));
            REQUIRE(first != nullptr);
            REQUIRE(second != nullptr);
            CHECK_FALSE(offsets_overlap(*first, *second));
        }
        {
            auto graph = FrameGraph{};
            two_queues(graph, true);
            auto const compiled = compile(graph, dedicated());
            REQUIRE(compiled.has_value());

            auto const &desc = graph.description();
            auto const plan = plan_transients(desc, *compiled, uniform(desc, 256), true);
            auto const *first = plan.placement_of(resource_named(desc, "first"));
            auto const *second = plan.placement_of(resource_named(desc, "second"));
            REQUIRE(first != nullptr);
            REQUIRE(second != nullptr);
            CHECK(offsets_overlap(*first, *second));
            CHECK(plan.barriers.size() == 1);
        }
        {
            auto graph = FrameGraph{};
            two_queues(graph, false);
            auto const compiled = compile(graph, single_queue());
            REQUIRE(compiled.has_value());

            auto const &desc = graph.description();
            auto const plan = plan_transients(desc, *compiled, uniform(desc, 256), true);
            CHECK(plan.total_bytes == 256);
        }
    }

    TEST_CASE("transients that accept different memory types go in separate blocks") {
        auto graph = FrameGraph{};
        disjoint_chains(graph);
        auto const compiled = compile(graph, single_queue());
        REQUIRE(compiled.has_value());

        auto const &desc = graph.description();
        auto requirements = uniform(desc, 256);
        requirements[resource_named(desc, "first")].memory_type_bits = 0b01U;
        requirements[resource_named(desc, "second")].memory_type_bits = 0b10U;

        auto const plan = plan_transients(desc, *compiled, requirements, true);
        CHECK(plan.blocks.size() == 2);
        CHECK(plan.total_bytes == 512);
    }

    TEST_CASE("blocks keep the strictest alignment and offsets respect it") {
        auto graph = FrameGraph{};
        overlapping_pair(graph);
        auto const compiled = compile(graph, single_queue());
        REQUIRE(compiled.has_value());

        auto const &desc = graph.description();
        auto requirements = uniform(desc, 100, 64);
        auto const plan = plan_transients(desc, *compiled, requirements, true);

        REQUIRE(plan.blocks.size() == 1);
        CHECK(plan.blocks[0].alignment == 64);
        for (auto const &placement: plan.placements) {
            CHECK(placement.offset % 64 == 0);
        }
        CHECK(plan.blocks[0].size == 228);
    }

    TEST_CASE("culled transients are not allocated") {
        auto graph = FrameGraph{};
        graph.add_pass("dead", PassType::compute, {}, [&](PassBuilder &pass) {
            [[maybe_unused]] auto const unused =
                    pass.write(transient(pass, "unused"), Use::storage_write, compute_stage);
            return noop();
        });
        graph.add_pass("alive", PassType::compute, {}, [&](PassBuilder &pass) {
            pass.side_effect();
            [[maybe_unused]] auto const used = pass.write(transient(pass, "used"), Use::storage_write, compute_stage);
            return noop();
        });
        auto const compiled = compile(graph, single_queue());
        REQUIRE(compiled.has_value());

        auto const &desc = graph.description();
        auto const plan = plan_transients(desc, *compiled, uniform(desc, 256), true);
        CHECK(plan.placements.size() == 1);
        CHECK(plan.placement_of(resource_named(desc, "used")) != nullptr);
    }

    TEST_CASE("transients_disjoint agrees with the placement") {
        auto graph = FrameGraph{};
        disjoint_chains(graph);
        auto const compiled = compile(graph, single_queue());
        REQUIRE(compiled.has_value());

        auto const &desc = graph.description();
        CHECK(transients_disjoint(desc, *compiled, resource_named(desc, "first"), resource_named(desc, "second")));

        auto overlapping = FrameGraph{};
        overlapping_pair(overlapping);
        auto const overlapping_plan = compile(overlapping, single_queue());
        REQUIRE(overlapping_plan.has_value());
        auto const &overlapping_desc = overlapping.description();
        CHECK_FALSE(transients_disjoint(overlapping_desc, *overlapping_plan, resource_named(overlapping_desc, "first"),
                                        resource_named(overlapping_desc, "second")));
    }

    TEST_CASE("property: no two transients with overlapping lifetimes share memory, on every topology") {
        auto aliased_pairs = std::size_t{0};
        auto graphs_with_transients = std::size_t{0};

        for (auto seed = std::uint32_t{0}; seed < 3000; ++seed) {
            auto graph = FrameGraph{};
            test::build_random_graph(graph, seed);
            if (!graph.declaration_errors().empty()) {
                continue;
            }

            for (auto const &topology: {single_queue(), same_family(), dedicated()}) {
                auto const compiled = compile(graph, topology);
                if (!compiled) {
                    continue;
                }

                auto const &desc = graph.description();
                auto requirements = std::vector<MemoryRequirement>(desc.resources.size());
                for (auto index = std::size_t{0}; index < desc.resources.size(); ++index) {
                    if (desc.resources[index].transient_image.has_value()) {
                        requirements[index] = MemoryRequirement{
                                .size = std::uint64_t{64} * (1U + ((seed + static_cast<std::uint32_t>(index)) % 7U)),
                                .alignment = 16U << ((seed + index) % 3U),
                                .memory_type_bits = ((seed + index) % 5U == 0) ? 0b01U : 0b11U,
                        };
                    }
                }

                for (auto const alias: {true, false}) {
                    auto const plan = plan_transients(desc, *compiled, requirements, alias);
                    if (!plan.placements.empty()) {
                        ++graphs_with_transients;
                    }

                    auto const reference = Reference{desc, *compiled};
                    for (auto a = std::size_t{0}; a < plan.placements.size(); ++a) {
                        auto const &pa = plan.placements[a];
                        REQUIRE(pa.block < plan.blocks.size());
                        auto const &block = plan.blocks[pa.block];
                        CHECK(pa.offset + pa.size <= block.size);
                        CHECK(pa.offset % requirements[pa.resource].alignment == 0);
                        CHECK(pa.offset % block.alignment == 0);
                        CHECK((block.memory_type_bits & requirements[pa.resource].memory_type_bits) != 0);

                        for (auto b = a + 1; b < plan.placements.size(); ++b) {
                            auto const &pb = plan.placements[b];
                            if (!offsets_overlap(pa, pb)) {
                                continue;
                            }
                            ++aliased_pairs;
                            REQUIRE(alias);

                            auto const first = passes_using(desc, *compiled, pa.resource);
                            auto const second = passes_using(desc, *compiled, pb.resource);
                            auto const all_before = [&](auto const &x, auto const &y) {
                                return std::ranges::all_of(x, [&](std::uint32_t i) {
                                    return std::ranges::all_of(y,
                                                               [&](std::uint32_t j) { return reference.before(i, j); });
                                });
                            };
                            CHECK((all_before(first, second) || all_before(second, first)));
                        }
                    }

                    CHECK(plan.total_bytes <= plan.unaliased_bytes);
                    if (!alias) {
                        CHECK(plan.total_bytes == plan.unaliased_bytes);
                    }
                }
            }
        }

        CHECK(graphs_with_transients > 100);
        CHECK(aliased_pairs > 0);
    }
}

TEST_SUITE("unit") {
    TEST_CASE("describe lists passes, uses, barriers and the transient placement") {
        auto graph = FrameGraph{};
        disjoint_chains(graph);
        auto const compiled = compile(graph, single_queue());
        REQUIRE(compiled.has_value());

        auto const &desc = graph.description();
        auto const plan = plan_transients(desc, *compiled, uniform(desc, 256), true);
        auto const text = describe(desc, *compiled, &plan);

        CHECK(text.find("batch 0 queue graphics") != std::string::npos);
        CHECK(text.find("pass write_first") != std::string::npos);
        CHECK(text.find("use 'first'") != std::string::npos);
        CHECK(text.find("alias memory") != std::string::npos);
        CHECK(text.find("transients: 256 bytes in 1 blocks (512 without aliasing)") != std::string::npos);
        CHECK(text.find("'second' block 0 offset 0 size 256") != std::string::npos);
    }
}
