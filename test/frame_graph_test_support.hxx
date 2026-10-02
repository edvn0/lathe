#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include "rendering/frame_graph/compiler.hxx"

// Test support for the frame graph compiler: an independent happens-before checker. Given the declarations and a
// compiled plan it asks, for every hazard pair (RAW, WAR, WAW on one resource), whether the plan orders the two
// accesses, using only what the plan actually records: barrier scopes, semaphore waits and ownership transfers.
namespace frame_graph::test {

    struct Event {
        std::uint32_t pass = 0;
        std::uint32_t resource = 0;
        UseInfo info;
        bool discard = false;
        std::size_t batch = 0;
        LogicalQueue queue = LogicalQueue::graphics;
        std::size_t slot = 0;
    };

    struct PlacedBarrier {
        std::uint32_t resource = 0;
        bool global = false; // a MemoryBarrier applies to every resource
        VkPipelineStageFlags2 src_stages = 0;
        VkAccessFlags2 src_access = 0;
        VkPipelineStageFlags2 dst_stages = 0;
        VkAccessFlags2 dst_access = 0;
        LogicalQueue queue = LogicalQueue::graphics;
        std::size_t slot = 0;
        OwnershipOp op = OwnershipOp::none;
    };

    inline auto covers_stages(VkPipelineStageFlags2 scope, VkPipelineStageFlags2 needed) -> bool {
        return (scope & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) != 0 || (needed & ~scope) == 0;
    }

    // Whether `barrier` makes `first`'s access ordered (and, for a write, available) before `second`'s access.
    inline auto barrier_orders(PlacedBarrier const &barrier, Event const &first, Event const &second) -> bool {
        if (barrier.op != OwnershipOp::none) {
            return false;
        }
        if (!covers_stages(barrier.src_stages, first.info.stages)) {
            return false;
        }
        if (first.info.writes) {
            auto const needed = first.info.access & write_access_mask;
            if ((barrier.src_access & VK_ACCESS_2_MEMORY_WRITE_BIT) == 0 && (needed & ~barrier.src_access) != 0) {
                return false;
            }
        }
        if (!covers_stages(barrier.dst_stages, second.info.stages)) {
            return false;
        }
        if (first.info.writes && second.info.reads) {
            auto const needed = second.info.access & ~write_access_mask;
            if ((barrier.dst_access & VK_ACCESS_2_MEMORY_READ_BIT) == 0 && (needed & ~barrier.dst_access) != 0) {
                return false;
            }
        }
        return true;
    }


    // A readable dump of a plan, for failure messages.
    inline auto describe(GraphDesc const &graph, CompiledGraph const &compiled) -> std::string {
        auto text = std::string{};
        auto const set_text = [&](char const *label, BarrierSet const &set) {
            for (auto const &b: set.images) {
                text += std::format("      {} image '{}' src({:#x},{:#x}) dst({:#x},{:#x}) layout {}->{} op {}\n",
                                    label, graph.resources[b.resource].name, b.src_stages, b.src_access, b.dst_stages,
                                    b.dst_access, static_cast<int>(b.old_layout), static_cast<int>(b.new_layout),
                                    static_cast<int>(b.op));
            }
            for (auto const &b: set.buffers) {
                text += std::format("      {} buffer '{}' src({:#x},{:#x}) dst({:#x},{:#x}) op {}\n", label,
                                    graph.resources[b.resource].name, b.src_stages, b.src_access, b.dst_stages,
                                    b.dst_access, static_cast<int>(b.op));
            }
        };
        for (auto index = std::size_t{0}; index < compiled.batches.size(); ++index) {
            auto const &batch = compiled.batches[index];
            text += std::format("batch {} queue {} signal {} waits {}\n", index, static_cast<int>(batch.queue),
                                batch.signal_index, batch.waits.size());
            set_text("acquire", batch.acquires);
            for (auto const &pass: batch.passes) {
                auto const &desc = graph.passes[pass.pass];
                text += std::format("    pass {}\n", desc.name);
                for (auto const &access: desc.accesses) {
                    auto const info = use_info(access.use, access.stages);
                    text += std::format("      use '{}' {} stages {:#x} access {:#x}{}\n",
                                        graph.resources[access.resource].name, static_cast<int>(access.use),
                                        info.stages, info.access, access.discard ? " discard" : "");
                }
                set_text("before", pass.before);
            }
            set_text("release", batch.releases);
            set_text("epilogue", batch.epilogue);
        }
        return text;
    }

    // Returns one message per violation; empty means the plan is sound.
    inline auto check_happens_before(GraphDesc const &graph, CompiledGraph const &compiled,
                                     QueueTopology const &topology) -> std::vector<std::string> {
        auto problems = std::vector<std::string>{};
        auto const batch_count = compiled.batches.size();

        // Structural checks on batches, signals and waits.
        auto last_signal = std::array<std::int64_t, logical_queue_count>{-1, -1};
        auto per_queue_count = std::array<std::uint32_t, logical_queue_count>{};
        for (auto index = std::size_t{0}; index < batch_count; ++index) {
            auto const &batch = compiled.batches[index];
            auto const q = static_cast<std::size_t>(batch.queue);
            if (static_cast<std::int64_t>(batch.signal_index) != last_signal[q] + 1) {
                problems.push_back(
                        std::format("batch {} signal index {} is not consecutive", index, batch.signal_index));
            }
            last_signal[q] = batch.signal_index;
            per_queue_count[q] += 1;
            for (auto const &wait: batch.waits) {
                auto found = false;
                for (auto earlier = std::size_t{0}; earlier < index; ++earlier) {
                    auto const &other = compiled.batches[earlier];
                    found = found || (other.queue == wait.queue && other.signal_index == wait.signal_index);
                }
                if (!found) {
                    problems.push_back(std::format("batch {} waits for a signal that is not yet submitted", index));
                }
                if (wait.queue == batch.queue) {
                    problems.push_back(std::format("batch {} waits on its own queue", index));
                }
            }
        }
        if (per_queue_count != compiled.signal_count) {
            problems.push_back("signal_count does not match the batches per queue");
        }

        // done[y] = batches that are complete before batch y starts.
        auto done = std::vector<std::vector<bool>>(batch_count, std::vector<bool>(batch_count, false));
        for (auto y = std::size_t{0}; y < batch_count; ++y) {
            for (auto const &wait: compiled.batches[y].waits) {
                for (auto z = std::size_t{0}; z < y; ++z) {
                    auto const &zb = compiled.batches[z];
                    if (zb.queue == wait.queue && zb.signal_index <= wait.signal_index) {
                        done[y][z] = true;
                        for (auto x = std::size_t{0}; x < z; ++x) {
                            if (done[z][x]) {
                                done[y][x] = true;
                            }
                        }
                    }
                }
            }
        }

        // Flatten the plan into events and placed barriers with a per-queue slot order.
        auto events = std::vector<Event>{};
        auto barriers = std::vector<PlacedBarrier>{};
        auto slot = std::array<std::size_t, logical_queue_count>{};
        auto const place = [&](BarrierSet const &set, LogicalQueue queue) {
            auto const q = static_cast<std::size_t>(queue);
            slot[q] += 1;
            for (auto const &b: set.images) {
                barriers.push_back({b.resource, false, b.src_stages, b.src_access, b.dst_stages, b.dst_access, queue,
                                    slot[q], b.op});
            }
            for (auto const &b: set.buffers) {
                barriers.push_back({b.resource, false, b.src_stages, b.src_access, b.dst_stages, b.dst_access, queue,
                                    slot[q], b.op});
            }
            for (auto const &b: set.memory) {
                barriers.push_back({0, true, b.src_stages, b.src_access, b.dst_stages, b.dst_access, queue, slot[q],
                                    OwnershipOp::none});
            }
        };
        for (auto index = std::size_t{0}; index < batch_count; ++index) {
            auto const &batch = compiled.batches[index];
            place(batch.acquires, batch.queue);
            for (auto const &compiled_pass: batch.passes) {
                place(compiled_pass.before, batch.queue);
                auto const q = static_cast<std::size_t>(batch.queue);
                slot[q] += 1;
                for (auto const &access: graph.passes[compiled_pass.pass].accesses) {
                    events.push_back(Event{
                            .pass = compiled_pass.pass,
                            .resource = access.resource,
                            .info = use_info(access.use, access.stages),
                            .discard = access.discard,
                            .batch = index,
                            .queue = batch.queue,
                            .slot = slot[q],
                    });
                }
            }
            place(batch.releases, batch.queue);
            place(batch.epilogue, batch.queue);
        }

        for (auto resource = std::size_t{0}; resource < graph.resources.size(); ++resource) {
            auto mine = std::vector<Event>{};
            for (auto const &event: events) {
                if (event.resource == resource) {
                    mine.push_back(event);
                }
            }
            std::ranges::sort(mine, [](Event const &a, Event const &b) { return a.pass < b.pass; });
            auto const n = mine.size();
            if (n < 2) {
                continue;
            }

            // order[i][j]: access i is ordered before access j (transitively).
            auto order = std::vector<std::vector<bool>>(n, std::vector<bool>(n, false));
            for (auto i = std::size_t{0}; i < n; ++i) {
                for (auto j = i + 1; j < n; ++j) {
                    auto const &a = mine[i];
                    auto const &b = mine[j];
                    if (a.batch != b.batch && done[b.batch][a.batch]) {
                        order[i][j] = true;
                        continue;
                    }
                    if (a.queue != b.queue) {
                        continue;
                    }
                    for (auto const &barrier: barriers) {
                        if (barrier.queue == a.queue && (barrier.global || barrier.resource == resource) &&
                            barrier.slot > a.slot && barrier.slot < b.slot && barrier_orders(barrier, a, b)) {
                            order[i][j] = true;
                            break;
                        }
                    }
                }
            }
            for (auto k = std::size_t{0}; k < n; ++k) {
                for (auto i = std::size_t{0}; i < k; ++i) {
                    for (auto j = k + 1; j < n; ++j) {
                        if (order[i][k] && order[k][j]) {
                            order[i][j] = true;
                        }
                    }
                }
            }

            for (auto i = std::size_t{0}; i < n; ++i) {
                for (auto j = i + 1; j < n; ++j) {
                    auto const &a = mine[i];
                    auto const &b = mine[j];
                    if (!a.info.writes && !b.info.writes) {
                        continue;
                    }
                    // Concurrent resources: reads on different queues need no order between them (handled by the
                    // read/read skip above); writes still do.
                    if (!order[i][j]) {
                        problems.push_back(std::format("hazard on '{}' between pass '{}' and pass '{}' is not ordered",
                                                       graph.resources[resource].name, graph.passes[a.pass].name,
                                                       graph.passes[b.pass].name));
                    }
                }
            }

            // Ownership of an exclusive resource moves at every queue switch between consecutive accesses, unless the
            // later access discards. Each such switch needs a transfer from the earlier queue to the later one.
            for (auto i = std::size_t{0}; i + 1 < n; ++i) {
                auto const &a = mine[i];
                auto const &b = mine[i + 1];
                auto const needs_transfer = a.queue != b.queue &&
                                            graph.resources[resource].sharing == Sharing::exclusive &&
                                            graph.resources[resource].kind != ResourceKind::token &&
                                            topology.family[0] != topology.family[1] && !b.discard;
                if (!needs_transfer) {
                    continue;
                }
                auto found = false;
                for (auto const &t: compiled.transfers) {
                    if (t.resource != resource || t.from != a.queue || t.to != b.queue) {
                        continue;
                    }
                    auto const &release = compiled.batches[t.release_batch];
                    auto const &acquire = compiled.batches[t.acquire_batch];
                    found = found || (release.signal_index >= compiled.batches[a.batch].signal_index &&
                                      acquire.signal_index <= compiled.batches[b.batch].signal_index);
                }
                if (!found) {
                    problems.push_back(std::format("no ownership transfer for '{}' between passes '{}' and '{}'",
                                                   graph.resources[resource].name, graph.passes[a.pass].name,
                                                   graph.passes[b.pass].name));
                }
            }
        }
        // Every transfer has matching release and acquire halves, and the acquire is ordered after the release.
        for (auto const &t: compiled.transfers) {
            auto const &release = compiled.batches[t.release_batch];
            auto const &acquire = compiled.batches[t.acquire_batch];
            if (release.queue != t.from || acquire.queue != t.to) {
                problems.push_back("transfer halves are on the wrong queues");
            }
            if (!done[t.acquire_batch][t.release_batch]) {
                problems.push_back("transfer acquire is not ordered after its release");
            }
            auto const has_half = [&](BarrierSet const &set, OwnershipOp op) {
                auto match = false;
                for (auto const &b: set.images) {
                    match = match || (t.is_image && b.resource == t.resource && b.op == op &&
                                      b.old_layout == t.old_layout && b.new_layout == t.new_layout &&
                                      b.src_family == topology.family[static_cast<std::size_t>(t.from)] &&
                                      b.dst_family == topology.family[static_cast<std::size_t>(t.to)]);
                }
                for (auto const &b: set.buffers) {
                    match = match || (!t.is_image && b.resource == t.resource && b.op == op &&
                                      b.src_family == topology.family[static_cast<std::size_t>(t.from)] &&
                                      b.dst_family == topology.family[static_cast<std::size_t>(t.to)]);
                }
                return match;
            };
            if (!has_half(release.releases, OwnershipOp::release) ||
                !has_half(acquire.acquires, OwnershipOp::acquire)) {
                problems.push_back(
                        std::format("transfer of '{}' lacks matching barriers", graph.resources[t.resource].name));
            }
        }

        // The swapchain is acquired once and render_finished is signalled by the last graphics batch.
        auto last_graphics = std::int64_t{-1};
        auto finished = 0;
        auto acquires = 0;
        for (auto index = std::size_t{0}; index < batch_count; ++index) {
            auto const &batch = compiled.batches[index];
            if (batch.queue == LogicalQueue::graphics) {
                last_graphics = static_cast<std::int64_t>(index);
            }
            finished += batch.signals_render_finished ? 1 : 0;
            acquires += batch.waits_swapchain_acquire ? 1 : 0;
        }
        if (finished != 1 || last_graphics < 0 ||
            !compiled.batches[static_cast<std::size_t>(last_graphics)].signals_render_finished) {
            problems.push_back("render_finished is not signalled exactly once by the last graphics batch");
        }
        if (acquires > 1) {
            problems.push_back("more than one batch waits on the swapchain acquire");
        }
        return problems;
    }

} // namespace frame_graph::test
