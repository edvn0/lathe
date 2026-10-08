#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include "rendering/frame_graph/compiler.hxx"
#include "rendering/frame_graph/describe.hxx"

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
        bool global = false;
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

    inline auto describe(GraphDesc const &graph, CompiledGraph const &compiled) -> std::string {
        return frame_graph::describe(graph, compiled);
    }

    inline auto check_happens_before(GraphDesc const &graph, CompiledGraph const &compiled,
                                     QueueTopology const &topology) -> std::vector<std::string> {
        auto problems = std::vector<std::string>{};
        auto const batch_count = compiled.batches.size();

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
            problems.emplace_back("signal_count does not match the batches per queue");
        }

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

        auto position = std::vector<std::size_t>(graph.passes.size(), 0);
        for (auto index = std::size_t{0}; index < compiled.schedule.size(); ++index) {
            position[compiled.schedule[index]] = index;
        }

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
            std::ranges::sort(mine,
                              [&](Event const &a, Event const &b) { return position[a.pass] < position[b.pass]; });
            auto const n = mine.size();
            if (n < 2) {
                continue;
            }

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
                    if (!order[i][j]) {
                        problems.push_back(std::format("hazard on '{}' between pass '{}' and pass '{}' is not ordered",
                                                       graph.resources[resource].name, graph.passes[a.pass].name,
                                                       graph.passes[b.pass].name));
                    }
                }
            }

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
        for (auto const &t: compiled.transfers) {
            auto const &release = compiled.batches[t.release_batch];
            auto const &acquire = compiled.batches[t.acquire_batch];
            if (release.queue != t.from || acquire.queue != t.to) {
                problems.emplace_back("transfer halves are on the wrong queues");
            }
            if (!done[t.acquire_batch][t.release_batch]) {
                problems.emplace_back("transfer acquire is not ordered after its release");
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
            problems.emplace_back("render_finished is not signalled exactly once by the last graphics batch");
        }
        if (acquires > 1) {
            problems.emplace_back("more than one batch waits on the swapchain acquire");
        }
        return problems;
    }

    inline auto build_random_graph(FrameGraph &graph, std::uint32_t seed, std::uint32_t fence_percent = 0) -> void {
        constexpr auto compute_stage = static_cast<ShaderStages>(ShaderStage::compute);
        constexpr auto fragment_stage = static_cast<ShaderStages>(ShaderStage::fragment);
        auto rng = std::uint64_t{seed} * 6364136223846793005ULL + 1442695040888963407ULL;
        auto const next = [&](std::uint32_t bound) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<std::uint32_t>((rng >> 33U) % bound);
        };
        auto const noop = []() { return RecordFn{}; };
        auto const pass_count = 2 + next(23);
        auto const resource_count = 1 + next(12);

        struct Slot {
            bool image = false;
            bool imported = false;
            ImageId image_id{};
            BufferId buffer_id{};
            bool written = false;
        };
        auto slots = std::vector<Slot>(resource_count);
        auto names = std::vector<std::string>{};
        for (auto index = std::uint32_t{0}; index < resource_count; ++index) {
            names.push_back(std::format("r{}", index));
        }
        for (auto index = std::uint32_t{0}; index < resource_count; ++index) {
            auto &slot = slots[index];
            slot.image = next(2) == 0;
            slot.imported = !slot.image || next(2) == 0;
            if (slot.imported && slot.image) {
                slot.image_id = graph.import_image({
                        .entry = {.layout = VK_IMAGE_LAYOUT_GENERAL,
                                  .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
                        .exit = {.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT},
                        .sharing = next(4) == 0 ? Sharing::concurrent : Sharing::exclusive,
                        .debug_name = names[index],
                });
                slot.written = true;
            } else if (slot.imported) {
                slot.buffer_id = graph.import_buffer({
                        .entry = {.stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
                        .exit = {.stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT},
                        .sharing = next(4) == 0 ? Sharing::concurrent : Sharing::exclusive,
                        .debug_name = names[index],
                });
                slot.written = true;
            }
        }

        for (auto pass_index = std::uint32_t{0}; pass_index < pass_count; ++pass_index) {
            auto const kind = next(4);
            auto const type = kind == 0 ? PassType::raster : kind == 3 ? PassType::transfer : PassType::compute;
            auto const affinity_roll = next(4);
            auto const affinity = type == PassType::raster ? QueueAffinity::graphics
                                  : affinity_roll == 0     ? QueueAffinity::graphics
                                  : affinity_roll == 1     ? QueueAffinity::compute_required
                                                           : QueueAffinity::compute_preferred;
            auto const last = pass_index + 1 == pass_count;
            graph.add_pass(std::format("pass_{}", pass_index), type, {}, [&](PassBuilder &p) {
                p.queue(affinity);
                if (fence_percent != 0 && next(100) < fence_percent) {
                    p.pinned();
                }
                if (last || next(5) == 0) {
                    p.side_effect();
                }
                auto const touches = 1 + next(3);
                auto used = std::vector<std::uint32_t>{};
                for (auto touch = std::uint32_t{0}; touch < touches; ++touch) {
                    auto const index = next(resource_count);
                    if (std::ranges::find(used, index) != used.end()) {
                        continue;
                    }
                    used.push_back(index);
                    auto &slot = slots[index];
                    auto const stage = type == PassType::raster ? fragment_stage : compute_stage;
                    auto const write = !slot.written || next(3) == 0;
                    if (slot.image) {
                        if (!slot.imported && !slot.written) {
                            slot.image_id = p.create(
                                    {.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "t"});
                        }
                        if (type == PassType::raster) {
                            slot.image_id = write ? p.color(slot.image_id, next(2) == 0 ? LoadOp::clear : LoadOp::load,
                                                            StoreOp::store)
                                                  : slot.image_id;
                            if (!write) {
                                [[maybe_unused]] auto const r = p.read(slot.image_id, Use::sampled, stage);
                            }
                        } else if (type == PassType::transfer) {
                            if (write) {
                                slot.image_id = p.write(slot.image_id, Use::transfer_dst);
                            } else {
                                [[maybe_unused]] auto const r = p.read(slot.image_id, Use::transfer_src);
                            }
                        } else if (write) {
                            slot.image_id = p.write(slot.image_id,
                                                    next(2) == 0 ? Use::storage_write : Use::storage_read_write, stage);
                        } else {
                            [[maybe_unused]] auto const r = p.read(slot.image_id, Use::sampled, stage);
                        }
                        slot.written = true;
                    } else {
                        if (type == PassType::transfer) {
                            if (write) {
                                slot.buffer_id = p.write(slot.buffer_id, Use::transfer_write);
                            } else {
                                [[maybe_unused]] auto const r = p.read(slot.buffer_id, Use::transfer_read);
                            }
                        } else if (write) {
                            slot.buffer_id = p.write(slot.buffer_id,
                                                     next(2) == 0 ? Use::shader_write : Use::shader_read_write, stage);
                        } else {
                            [[maybe_unused]] auto const r = p.read(slot.buffer_id, Use::shader_read, stage);
                        }
                        slot.written = true;
                    }
                }
                return noop();
            });
        }
    }

}
