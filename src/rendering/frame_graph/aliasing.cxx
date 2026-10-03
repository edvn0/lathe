#include "rendering/frame_graph/aliasing.hxx"

#include <algorithm>
#include <utility>

#include "rendering/frame_graph/use_table.hxx"

namespace frame_graph {
    namespace {

        struct PassLocation {
            std::int64_t batch = -1; // -1: not part of the compiled plan (culled)
            std::uint32_t position = 0;
        };

        // The compiled plan's order, as everything the aliasing rules need: where each live pass sits, and which
        // batches are ordered before which.
        struct Order {
            std::vector<PassLocation> pass_location;
            std::vector<std::vector<bool>>
                    batch_before; // batch_before[a][b]: every command of a precedes every one of b

            [[nodiscard]] auto before(std::uint32_t first, std::uint32_t second) const -> bool {
                auto const a = pass_location[first];
                auto const b = pass_location[second];
                if (a.batch < 0 || b.batch < 0) {
                    return false;
                }
                if (a.batch == b.batch) {
                    return a.position < b.position;
                }
                return batch_before[static_cast<std::size_t>(a.batch)][static_cast<std::size_t>(b.batch)];
            }
        };

        auto build_order(GraphDesc const &graph, CompiledGraph const &compiled) -> Order {
            auto order = Order{};
            order.pass_location.assign(graph.passes.size(), PassLocation{});

            auto const batch_count = compiled.batches.size();
            for (auto batch = std::size_t{0}; batch < batch_count; ++batch) {
                auto const &passes = compiled.batches[batch].passes;
                for (auto position = std::size_t{0}; position < passes.size(); ++position) {
                    order.pass_location[passes[position].pass] = PassLocation{
                            .batch = static_cast<std::int64_t>(batch),
                            .position = static_cast<std::uint32_t>(position),
                    };
                }
            }

            order.batch_before.assign(batch_count, std::vector<bool>(batch_count, false));
            for (auto later = std::size_t{0}; later < batch_count; ++later) {
                for (auto earlier = std::size_t{0}; earlier < later; ++earlier) {
                    // Batches on one queue run in submission order.
                    if (compiled.batches[earlier].queue == compiled.batches[later].queue) {
                        order.batch_before[earlier][later] = true;
                    }
                }
                for (auto const &wait: compiled.batches[later].waits) {
                    for (auto earlier = std::size_t{0}; earlier < later; ++earlier) {
                        auto const &other = compiled.batches[earlier];
                        if (other.queue == wait.queue && other.signal_index <= wait.signal_index) {
                            order.batch_before[earlier][later] = true;
                        }
                    }
                }
            }

            // Transitive closure; every edge points forward in submission order, so one sweep per intermediate batch.
            for (auto through = std::size_t{0}; through < batch_count; ++through) {
                for (auto from = std::size_t{0}; from < through; ++from) {
                    if (!order.batch_before[from][through]) {
                        continue;
                    }
                    for (auto to = through + 1; to < batch_count; ++to) {
                        if (order.batch_before[through][to]) {
                            order.batch_before[from][to] = true;
                        }
                    }
                }
            }
            return order;
        }

        // The live passes that touch `resource`, as declaration indices.
        auto touching_passes(GraphDesc const &graph, CompiledGraph const &compiled, std::uint32_t resource)
                -> std::vector<std::uint32_t> {
            auto passes = std::vector<std::uint32_t>{};
            for (auto pass = std::uint32_t{0}; pass < graph.passes.size(); ++pass) {
                if (pass < compiled.pass_culled.size() && compiled.pass_culled[pass]) {
                    continue;
                }
                if (std::ranges::any_of(graph.passes[pass].accesses,
                                        [&](AccessDesc const &access) { return access.resource == resource; })) {
                    passes.push_back(pass);
                }
            }
            return passes;
        }

        auto all_before(Order const &order, std::vector<std::uint32_t> const &first,
                        std::vector<std::uint32_t> const &second) -> bool {
            return std::ranges::all_of(first, [&](std::uint32_t a) {
                return std::ranges::all_of(second, [&](std::uint32_t b) { return order.before(a, b); });
            });
        }

        auto align_up(std::uint64_t value, std::uint64_t alignment) -> std::uint64_t {
            return alignment <= 1 ? value : (value + alignment - 1) / alignment * alignment;
        }

        struct Candidate {
            std::uint32_t resource = 0;
            std::vector<std::uint32_t> passes;
        };

        struct Placed {
            std::size_t candidate = 0;
            std::uint32_t block = 0;
            std::uint64_t offset = 0;
            std::uint64_t size = 0;
        };

        // The scope a pass's accesses to `resource` cover, for the barrier that hands memory on.
        auto access_scope(GraphDesc const &graph, std::vector<std::uint32_t> const &passes, std::uint32_t resource,
                          VkPipelineStageFlags2 &stages, VkAccessFlags2 &access) -> void {
            for (auto const pass: passes) {
                for (auto const &used: graph.passes[pass].accesses) {
                    if (used.resource != resource) {
                        continue;
                    }
                    auto const info = use_info(used.exit_use ? *used.exit_use : used.use, used.stages);
                    stages |= info.stages;
                    access |= info.access;
                }
            }
        }

    } // namespace

    auto transient_usage(GraphDesc const &graph, CompiledGraph const &compiled, std::uint32_t resource)
            -> VkImageUsageFlags {
        auto usage = VkImageUsageFlags{0};
        for (auto const pass: touching_passes(graph, compiled, resource)) {
            for (auto const &access: graph.passes[pass].accesses) {
                if (access.resource != resource) {
                    continue;
                }
                for (auto const use: {access.use, access.exit_use.value_or(access.use)}) {
                    switch (use) {
                        case Use::color_attachment:
                        case Use::color_resolve:
                            usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                            break;
                        case Use::depth_attachment:
                        case Use::depth_resolve:
                            usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
                            break;
                        case Use::sampled:
                            usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                            break;
                        case Use::storage_read:
                        case Use::storage_write:
                        case Use::storage_read_write:
                            usage |= VK_IMAGE_USAGE_STORAGE_BIT;
                            break;
                        case Use::transfer_src:
                            usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                            break;
                        case Use::transfer_dst:
                            usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                            break;
                        default:
                            break;
                    }
                }
            }
        }
        return usage;
    }

    auto transients_disjoint(GraphDesc const &graph, CompiledGraph const &compiled, std::uint32_t first_resource,
                             std::uint32_t second_resource) -> bool {
        auto const order = build_order(graph, compiled);
        auto const first = touching_passes(graph, compiled, first_resource);
        auto const second = touching_passes(graph, compiled, second_resource);
        return all_before(order, first, second) || all_before(order, second, first);
    }

    auto plan_transients(GraphDesc const &graph, CompiledGraph const &compiled,
                         std::span<MemoryRequirement const> requirements, bool alias) -> TransientPlan {
        auto plan = TransientPlan{};
        auto const order = build_order(graph, compiled);

        auto candidates = std::vector<Candidate>{};
        for (auto resource = std::uint32_t{0}; resource < graph.resources.size(); ++resource) {
            if (!graph.resources[resource].transient_image.has_value() || resource >= requirements.size() ||
                requirements[resource].size == 0) {
                continue;
            }
            auto passes = touching_passes(graph, compiled, resource);
            if (passes.empty()) {
                continue; // culled away: nothing to allocate
            }
            candidates.push_back(Candidate{.resource = resource, .passes = std::move(passes)});
            plan.unaliased_bytes += requirements[resource].size;
        }

        // Largest first, so the big ones set the layout and the small ones fill the gaps. Ties keep slot order.
        auto by_size = std::vector<std::size_t>(candidates.size());
        for (auto index = std::size_t{0}; index < by_size.size(); ++index) {
            by_size[index] = index;
        }
        std::ranges::stable_sort(by_size, [&](std::size_t a, std::size_t b) {
            return requirements[candidates[a].resource].size > requirements[candidates[b].resource].size;
        });

        auto const conflicts = [&](std::size_t a, std::size_t b) {
            return !(all_before(order, candidates[a].passes, candidates[b].passes) ||
                     all_before(order, candidates[b].passes, candidates[a].passes));
        };

        auto placed = std::vector<Placed>{};
        for (auto const index: by_size) {
            auto const &requirement = requirements[candidates[index].resource];

            auto chosen_block = std::int64_t{-1};
            auto chosen_offset = std::uint64_t{0};

            if (alias) {
                for (auto block = std::uint32_t{0}; block < plan.blocks.size() && chosen_block < 0; ++block) {
                    auto &candidate_block = plan.blocks[block];
                    if ((candidate_block.memory_type_bits & requirement.memory_type_bits) == 0) {
                        continue;
                    }

                    // Intervals of what this one conflicts with, by offset.
                    auto taken = std::vector<std::pair<std::uint64_t, std::uint64_t>>{};
                    for (auto const &other: placed) {
                        if (other.block == block && conflicts(index, other.candidate)) {
                            taken.emplace_back(other.offset, other.offset + other.size);
                        }
                    }
                    std::ranges::sort(taken);

                    auto const alignment = std::max(requirement.alignment, candidate_block.alignment);
                    auto offset = std::uint64_t{0};
                    for (auto const &[begin, end]: taken) {
                        if (offset + requirement.size <= begin) {
                            break;
                        }
                        offset = std::max(offset, align_up(end, alignment));
                    }
                    chosen_block = block;
                    chosen_offset = align_up(offset, alignment);
                }
            }

            if (chosen_block < 0) {
                chosen_block = static_cast<std::int64_t>(plan.blocks.size());
                plan.blocks.push_back(TransientBlock{.alignment = requirement.alignment,
                                                     .memory_type_bits = requirement.memory_type_bits});
                chosen_offset = 0;
            }

            auto &block = plan.blocks[static_cast<std::size_t>(chosen_block)];
            block.size = std::max(block.size, chosen_offset + requirement.size);
            block.alignment = std::max(block.alignment, requirement.alignment);
            block.memory_type_bits &= requirement.memory_type_bits;
            placed.push_back(Placed{
                    .candidate = index,
                    .block = static_cast<std::uint32_t>(chosen_block),
                    .offset = chosen_offset,
                    .size = requirement.size,
            });
        }

        // Back to slot order for the caller.
        std::ranges::sort(placed, [&](Placed const &a, Placed const &b) {
            return candidates[a.candidate].resource < candidates[b.candidate].resource;
        });
        for (auto const &p: placed) {
            plan.placements.push_back(TransientPlacement{
                    .resource = candidates[p.candidate].resource,
                    .block = p.block,
                    .offset = p.offset,
                    .size = p.size,
            });
        }
        for (auto const &block: plan.blocks) {
            plan.total_bytes += block.size;
        }

        // Memory handed from one transient to the next needs a dependency before the newcomer's first use.
        if (alias) {
            for (auto const &newcomer: placed) {
                auto stages = VkPipelineStageFlags2{VK_PIPELINE_STAGE_2_NONE};
                auto access = VkAccessFlags2{VK_ACCESS_2_NONE};
                for (auto const &previous: placed) {
                    if (&previous == &newcomer || previous.block != newcomer.block ||
                        previous.offset + previous.size <= newcomer.offset ||
                        newcomer.offset + newcomer.size <= previous.offset ||
                        !all_before(order, candidates[previous.candidate].passes,
                                    candidates[newcomer.candidate].passes)) {
                        continue;
                    }
                    access_scope(graph, candidates[previous.candidate].passes, candidates[previous.candidate].resource,
                                 stages, access);
                }
                if (stages == VK_PIPELINE_STAGE_2_NONE) {
                    continue;
                }

                // The newcomer's first pass in execution order.
                auto first_pass = std::uint32_t{0};
                auto first_position = std::size_t{compiled.schedule.size()};
                for (auto const pass: candidates[newcomer.candidate].passes) {
                    auto const position = static_cast<std::size_t>(std::ranges::find(compiled.schedule, pass) -
                                                                   compiled.schedule.begin());
                    if (position < first_position) {
                        first_position = position;
                        first_pass = pass;
                    }
                }

                auto dst_stages = VkPipelineStageFlags2{VK_PIPELINE_STAGE_2_NONE};
                auto dst_access = VkAccessFlags2{VK_ACCESS_2_NONE};
                access_scope(graph, {first_pass}, candidates[newcomer.candidate].resource, dst_stages, dst_access);

                plan.barriers.push_back(AliasingBarrier{
                        .pass = first_pass,
                        .barrier = MemoryBarrier{.src_stages = stages,
                                                 .src_access = access,
                                                 .dst_stages = dst_stages,
                                                 .dst_access = dst_access},
                });
            }

            // One barrier per pass: several newcomers can start at the same pass.
            std::ranges::sort(plan.barriers,
                              [](AliasingBarrier const &a, AliasingBarrier const &b) { return a.pass < b.pass; });
            auto merged = std::vector<AliasingBarrier>{};
            for (auto const &barrier: plan.barriers) {
                if (!merged.empty() && merged.back().pass == barrier.pass) {
                    merged.back().barrier.src_stages |= barrier.barrier.src_stages;
                    merged.back().barrier.src_access |= barrier.barrier.src_access;
                    merged.back().barrier.dst_stages |= barrier.barrier.dst_stages;
                    merged.back().barrier.dst_access |= barrier.barrier.dst_access;
                } else {
                    merged.push_back(barrier);
                }
            }
            plan.barriers = std::move(merged);
        }

        return plan;
    }

} // namespace frame_graph
