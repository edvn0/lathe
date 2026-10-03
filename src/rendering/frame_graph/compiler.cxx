#include "rendering/frame_graph/compiler.hxx"

#include "rendering/frame_graph/scheduler.hxx"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <optional>

namespace frame_graph {
    namespace {

        // Nodes order the work of one frame: the virtual prologue (-1), the passes by declaration index, and the
        // virtual epilogue (pass_count). Every cross-queue edge runs from a lower node to a higher one.
        constexpr std::int64_t prologue_node = -1;
        constexpr std::int64_t no_node = std::numeric_limits<std::int64_t>::min();
        constexpr auto no_resource = std::numeric_limits<std::uint32_t>::max();

        constexpr auto queue_index(LogicalQueue queue) noexcept -> std::size_t {
            return static_cast<std::size_t>(queue);
        }

        constexpr auto other_queue(LogicalQueue queue) noexcept -> LogicalQueue {
            return queue == LogicalQueue::graphics ? LogicalQueue::compute : LogicalQueue::graphics;
        }

        struct Tracked {
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            bool accessed = false;
            LogicalQueue owner = LogicalQueue::graphics;
            bool has_write = false;
            LogicalQueue write_queue = LogicalQueue::graphics;
            VkPipelineStageFlags2 write_stages = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 write_access = VK_ACCESS_2_NONE;
            std::array<VkPipelineStageFlags2, logical_queue_count> read_stages{}; // since the last write
            std::array<VkPipelineStageFlags2, logical_queue_count> visible_stages{};
            std::array<VkAccessFlags2, logical_queue_count> visible_access{};
            std::array<std::int64_t, logical_queue_count> last_access{no_node, no_node};
            std::int64_t last_write_node = no_node;
            // Per queue: accesses there are ordered after the last cross-queue modification only by a semaphore wait,
            // which covers just the waiting batch. Later accesses chain from `chain_src` with an execution barrier.
            std::array<bool, logical_queue_count> cross_ordered{};
            std::array<VkPipelineStageFlags2, logical_queue_count> chain_src{};
            std::array<VkPipelineStageFlags2, logical_queue_count> chained{};
            bool entry_has_work = false; // the import's entry state holds work a later queue must wait for
        };

        struct AccessSpec {
            VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 access = VK_ACCESS_2_NONE;
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            bool is_image = false;
            bool writes = false;
            bool discard = false;

            // What the pass leaves the resource as, when that differs from what it entered as.
            struct Exit {
                VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
                VkAccessFlags2 access = VK_ACCESS_2_NONE;
                VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            };
            std::optional<Exit> exit;
        };

        struct Edge {
            std::int64_t src = 0;
            std::int64_t dst = 0;
            VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
        };

        struct PendingTransfer {
            std::uint32_t resource = 0;
            bool is_image = false;
            LogicalQueue from = LogicalQueue::graphics;
            LogicalQueue to = LogicalQueue::graphics;
            std::int64_t release_node = 0;
            std::int64_t acquire_node = 0;
            VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        };

        struct Hasher {
            std::uint64_t state = 0xcbf29ce484222325ULL;

            auto mix(std::uint64_t value) -> void {
                for (auto shift = 0U; shift < 64U; shift += 8U) {
                    state ^= (value >> shift) & 0xFFU;
                    state *= 0x100000001b3ULL;
                }
            }

            auto mix(std::string_view text) -> void {
                for (auto const c: text) {
                    state ^= static_cast<std::uint8_t>(c);
                    state *= 0x100000001b3ULL;
                }
                mix(text.size());
            }
        };

        auto fail(FrameGraphErrorType type, PassDesc const &pass, GraphDesc const &graph,
                  std::uint32_t resource) -> std::unexpected<FrameGraphError> {
            return std::unexpected(FrameGraphError{
                    .type = type,
                    .pass = pass.name,
                    .resource = resource < graph.resources.size() ? graph.resources[resource].name : std::string{},
            });
        }

        auto validate(GraphDesc const &graph) -> std::expected<void, FrameGraphError> {
            for (auto const &pass: graph.passes) {
                auto const compute_affinity = pass.affinity != QueueAffinity::graphics;
                if (pass.type == PassType::raster && compute_affinity) {
                    return fail(FrameGraphErrorType::raster_pass_with_compute_affinity, pass, graph, no_resource);
                }
                for (auto const &access: pass.accesses) {
                    if (is_attachment_use(access.use) && pass.type != PassType::raster) {
                        return fail(FrameGraphErrorType::attachment_in_non_raster_pass, pass, graph, access.resource);
                    }
                    if (is_token_use(access.use) && compute_affinity) {
                        return fail(FrameGraphErrorType::token_on_compute_pass, pass, graph, access.resource);
                    }
                }
            }
            return {};
        }

        auto cull(GraphDesc const &graph) -> std::vector<bool> {
            auto const pass_count = graph.passes.size();
            auto live = std::vector<bool>(pass_count, false);
            auto stack = std::vector<std::size_t>{};

            auto const mark = [&](std::int64_t pass) {
                if (pass >= 0 && !live[static_cast<std::size_t>(pass)]) {
                    live[static_cast<std::size_t>(pass)] = true;
                    stack.push_back(static_cast<std::size_t>(pass));
                }
            };

            for (auto index = std::size_t{0}; index < pass_count; ++index) {
                auto const &pass = graph.passes[index];
                auto root = pass.side_effect || pass.legacy || pass.pinned;
                for (auto const &access: pass.accesses) {
                    auto const &resource = graph.resources[access.resource];
                    // A pass that writes an imported resource's final version leaves the frame with it.
                    root = root || resource.swapchain || (resource.imported && access.produces);
                }
                if (root) {
                    mark(static_cast<std::int64_t>(index));
                }
            }

            while (!stack.empty()) {
                auto const index = stack.back();
                stack.pop_back();
                for (auto const &access: graph.passes[index].accesses) {
                    if (access.discard) {
                        continue;
                    }
                    mark(graph.producers[access.resource][access.version]);
                }
            }
            return live;
        }

        // reach[i] has bit j set when j is reachable from i (descendant).
        auto reachability(std::vector<std::vector<std::size_t>> const &successors)
                -> std::vector<std::vector<std::uint64_t>> {
            auto const count = successors.size();
            auto const words = (count + 63) / 64;
            auto reach = std::vector<std::vector<std::uint64_t>>(count, std::vector<std::uint64_t>(words, 0));
            // Successors always have higher indices, so a reverse sweep sees completed rows.
            for (auto index = count; index-- > 0;) {
                for (auto const next: successors[index]) {
                    reach[index][next / 64] |= std::uint64_t{1} << (next % 64);
                    for (auto word = std::size_t{0}; word < words; ++word) {
                        reach[index][word] |= reach[next][word];
                    }
                }
            }
            return reach;
        }

        auto test_bit(std::vector<std::uint64_t> const &bits, std::size_t index) -> bool {
            return ((bits[index / 64] >> (index % 64)) & 1U) != 0;
        }

        // compute_required goes to compute. compute_preferred goes to compute unless every live graphics pass is an
        // ancestor or descendant of it: that would cost two semaphores and overlap nothing.
        auto resolve_queues(GraphDesc const &graph, std::vector<bool> const &live,
                            bool async) -> std::vector<LogicalQueue> {
            auto queues = std::vector<LogicalQueue>(graph.passes.size(), LogicalQueue::graphics);
            if (!async) {
                return queues;
            }

            auto const successors = dependency_successors(graph, live);
            auto const reach = reachability(successors);

            for (auto index = std::size_t{0}; index < graph.passes.size(); ++index) {
                if (!live[index]) {
                    continue;
                }
                auto const affinity = graph.passes[index].affinity;
                if (affinity == QueueAffinity::compute_required) {
                    queues[index] = LogicalQueue::compute;
                    continue;
                }
                if (affinity != QueueAffinity::compute_preferred) {
                    continue;
                }
                auto overlappable = false;
                for (auto other = std::size_t{0}; other < graph.passes.size() && !overlappable; ++other) {
                    if (other == index || !live[other] || graph.passes[other].affinity != QueueAffinity::graphics) {
                        continue;
                    }
                    auto const related = test_bit(reach[index], other) || test_bit(reach[other], index);
                    overlappable = !related;
                }
                queues[index] = overlappable ? LogicalQueue::compute : LogicalQueue::graphics;
            }
            return queues;
        }

        auto exit_of(AccessDesc const &access) -> std::optional<AccessSpec::Exit> {
            if (!access.exit_use) {
                return std::nullopt;
            }
            auto const exit = use_info(*access.exit_use, access.stages);
            return AccessSpec::Exit{.stages = exit.stages, .access = exit.access, .layout = exit.layout};
        }

        // Applies accesses to the tracked state and derives barriers, ownership transfers and cross-queue edges.
        class Tracker {
        public:
            Tracker(GraphDesc const &graph, QueueTopology const &topology, CompileOptions const &options,
                    bool multi_queue, std::size_t node_count) :
                graph_(graph), topology_(topology), options_(options), multi_queue_(multi_queue),
                tracked_(graph.resources.size()), acquires_(node_count), releases_(node_count) {
                for (auto index = std::size_t{0}; index < graph.resources.size(); ++index) {
                    auto const &resource = graph.resources[index];
                    if (!resource.imported) {
                        continue;
                    }
                    auto &state = tracked_[index];
                    state.layout = resource.entry.layout;
                    state.accessed = true;
                    state.owner = LogicalQueue::graphics;
                    state.has_write = true;
                    state.write_queue = LogicalQueue::graphics;
                    state.write_stages = resource.entry.stages;
                    state.write_access = resource.entry.access;
                    state.last_access[queue_index(LogicalQueue::graphics)] = prologue_node;
                    state.last_write_node = prologue_node;
                    state.entry_has_work = resource.entry.stages != 0 || resource.entry.access != 0;
                    // A read-only entry state was made visible by the previous frame's exit barrier, so a first read
                    // within it needs no barrier.
                    if ((resource.entry.access & write_access_mask) == 0) {
                        state.visible_stages[queue_index(LogicalQueue::graphics)] = resource.entry.stages;
                        state.visible_access[queue_index(LogicalQueue::graphics)] = resource.entry.access;
                    }
                }
            }

            auto apply(std::uint32_t resource, LogicalQueue queue, std::int64_t node, AccessSpec spec,
                       BarrierSet &before) -> void {
                auto &state = tracked_[resource];
                auto const desc_kind = graph_.resources[resource].kind;
                auto const qi = queue_index(queue);
                auto const other = other_queue(queue);
                auto const oi = queue_index(other);

                if (!state.accessed) {
                    state.owner = queue;
                }

                auto const layout_change = spec.is_image && state.layout != spec.layout;
                auto const old_layout = spec.discard ? VK_IMAGE_LAYOUT_UNDEFINED : state.layout;
                auto const own_write = state.has_write && state.write_queue == queue;

                auto const needs_transfer =
                        multi_queue_ && state.accessed && state.owner != queue && desc_kind != ResourceKind::token &&
                        graph_.resources[resource].sharing == Sharing::exclusive &&
                        !topology_.same_family(LogicalQueue::graphics, LogicalQueue::compute) && !spec.discard;

                // A layout transition is a write: it must follow the other queue's accesses in the old layout.
                auto const modifies = spec.writes || layout_change;
                auto ordered_by_edge = false;
                if (multi_queue_ && state.accessed) {
                    ordered_by_edge = add_edges(state, spec, modifies, other, node, needs_transfer);
                }

                // A reader that is not itself waiting must still be ordered after the cross-queue write the previous
                // reader waited on: chain from that reader's stages with an execution-only barrier.
                auto chain_stages = VkPipelineStageFlags2{VK_PIPELINE_STAGE_2_NONE};
                if (!needs_transfer && !modifies && !ordered_by_edge && state.cross_ordered[qi] &&
                    (spec.stages & ~state.chained[qi]) != 0) {
                    chain_stages = state.chain_src[qi];
                    state.chained[qi] |= spec.stages;
                }

                if (needs_transfer) {
                    emit_transfer(resource, state, spec, queue, node);
                } else {
                    emit_barrier(resource, state, spec, queue, own_write, layout_change, old_layout, chain_stages,
                                 before);
                }

                if (ordered_by_edge) {
                    state.cross_ordered[qi] = !spec.writes;
                    state.chain_src[qi] = spec.stages;
                    state.chained[qi] = VK_PIPELINE_STAGE_2_NONE;
                } else if (modifies) {
                    state.cross_ordered[qi] = false;
                }

                // State update.
                if (modifies || needs_transfer) {
                    state.has_write = true;
                    state.write_queue = queue;
                    state.write_stages = spec.stages;
                    state.write_access = spec.access;
                    state.read_stages = {};
                    state.visible_stages = {};
                    state.visible_access = {};
                    state.last_write_node = node;
                    state.last_access[oi] = no_node;
                }
                if (!spec.writes) {
                    state.read_stages[qi] |= spec.stages;
                    state.visible_stages[qi] |= spec.stages;
                    state.visible_access[qi] |= spec.access;
                }
                state.last_access[qi] = node;
                state.owner = queue;
                if (spec.is_image) {
                    state.layout = spec.layout;
                }
                state.accessed = true;

                // A pass that manages the resource itself leaves it as its exit use says, as if that were its write.
                if (spec.exit) {
                    state.has_write = true;
                    state.write_queue = queue;
                    state.write_stages = spec.exit->stages;
                    state.write_access = spec.exit->access;
                    state.read_stages = {};
                    state.visible_stages = {};
                    state.visible_access = {};
                    // The pass made its writes visible to a read-only exit state itself, so readers in that scope need
                    // no further barrier.
                    if ((spec.exit->access & write_access_mask) == 0) {
                        state.visible_stages[qi] = spec.exit->stages;
                        state.visible_access[qi] = spec.exit->access;
                    }
                    if (spec.is_image) {
                        state.layout = spec.exit->layout;
                    }
                }
            }

            [[nodiscard]] auto edges() const -> std::vector<Edge> const & { return edges_; }
            [[nodiscard]] auto transfers() const -> std::vector<PendingTransfer> const & { return transfers_; }
            [[nodiscard]] auto acquires_at(std::int64_t node) -> BarrierSet & {
                return acquires_[static_cast<std::size_t>(node + 1)];
            }
            [[nodiscard]] auto releases_at(std::int64_t node) -> BarrierSet & {
                return releases_[static_cast<std::size_t>(node + 1)];
            }
            [[nodiscard]] auto layout_of(std::uint32_t resource) const -> VkImageLayout {
                return tracked_[resource].layout;
            }

        private:
            auto add_edge(std::int64_t src, std::int64_t dst, VkPipelineStageFlags2 stages) -> bool {
                if (src == no_node || src == dst) {
                    return false;
                }
                edges_.push_back(Edge{.src = src, .dst = dst, .stages = stages});
                return true;
            }

            auto add_edges(Tracked const &state, AccessSpec const &spec, bool modifies, LogicalQueue other,
                           std::int64_t node, bool needs_transfer) -> bool {
                auto const oi = queue_index(other);
                if (needs_transfer) {
                    // The release must follow the owner's last access, even the prologue's.
                    return add_edge(state.last_access[oi], node, spec.stages);
                }
                // Without a transfer, the prologue only matters if its entry state left work behind.
                auto const src = modifies                                          ? state.last_access[oi]
                                 : (state.has_write && state.write_queue == other) ? state.last_write_node
                                                                                   : no_node;
                if (src == prologue_node && !state.entry_has_work) {
                    return false;
                }
                return add_edge(src, node, spec.stages);
            }

            auto emit_transfer(std::uint32_t resource, Tracked const &state, AccessSpec const &spec, LogicalQueue queue,
                               std::int64_t node) -> void {
                auto const owner = state.owner;
                auto const oi = queue_index(owner);
                auto const release_node = state.last_access[oi];
                auto const write_on_owner = state.has_write && state.write_queue == owner;
                auto const src_stages =
                        (write_on_owner ? state.write_stages : VK_PIPELINE_STAGE_2_NONE) | state.read_stages[oi];
                auto const src_access = write_on_owner ? state.write_access : VK_ACCESS_2_NONE;
                auto const src_family = topology_.family[oi];
                auto const dst_family = topology_.family[queue_index(queue)];

                if (spec.is_image) {
                    auto barrier = ImageBarrier{
                            .resource = resource,
                            .src_stages = src_stages,
                            .src_access = src_access,
                            .dst_stages = VK_PIPELINE_STAGE_2_NONE,
                            .dst_access = VK_ACCESS_2_NONE,
                            .old_layout = state.layout,
                            .new_layout = spec.layout,
                            .src_family = src_family,
                            .dst_family = dst_family,
                            .op = OwnershipOp::release,
                    };
                    releases_at(release_node).images.push_back(barrier);
                    barrier.src_stages = VK_PIPELINE_STAGE_2_NONE;
                    barrier.src_access = VK_ACCESS_2_NONE;
                    barrier.dst_stages = spec.stages;
                    barrier.dst_access = spec.access;
                    barrier.op = OwnershipOp::acquire;
                    acquires_at(node).images.push_back(barrier);
                } else {
                    auto barrier = BufferBarrier{
                            .resource = resource,
                            .src_stages = src_stages,
                            .src_access = src_access,
                            .src_family = src_family,
                            .dst_family = dst_family,
                            .op = OwnershipOp::release,
                    };
                    releases_at(release_node).buffers.push_back(barrier);
                    barrier.src_stages = VK_PIPELINE_STAGE_2_NONE;
                    barrier.src_access = VK_ACCESS_2_NONE;
                    barrier.dst_stages = spec.stages;
                    barrier.dst_access = spec.access;
                    barrier.op = OwnershipOp::acquire;
                    acquires_at(node).buffers.push_back(barrier);
                }
                transfers_.push_back(PendingTransfer{
                        .resource = resource,
                        .is_image = spec.is_image,
                        .from = owner,
                        .to = queue,
                        .release_node = release_node,
                        .acquire_node = node,
                        .old_layout = state.layout,
                        .new_layout = spec.layout,
                });
            }

            auto emit_barrier(std::uint32_t resource, Tracked const &state, AccessSpec const &spec, LogicalQueue queue,
                              bool own_write, bool layout_change, VkImageLayout old_layout,
                              VkPipelineStageFlags2 chain_stages, BarrierSet &before) -> void {
                auto const qi = queue_index(queue);
                auto src_stages = VkPipelineStageFlags2{VK_PIPELINE_STAGE_2_NONE};
                auto src_access = VkAccessFlags2{VK_ACCESS_2_NONE};
                auto dst_stages = spec.stages;
                auto dst_access = spec.access;
                auto need_barrier = false;

                if (spec.writes) {
                    // WAW, WAR, RAW-then-write: wait for everything before and make the last write available.
                    src_stages = (own_write ? state.write_stages : VK_PIPELINE_STAGE_2_NONE) | state.read_stages[qi];
                    // A discarding write still follows the earlier write in memory order, so that write must be made
                    // available. Only real write bits count: a layout transition recorded as a write carries read bits.
                    src_access = own_write ? (state.write_access & write_access_mask) : VK_ACCESS_2_NONE;
                    need_barrier = src_stages != 0 || layout_change;
                } else {
                    auto const covered = (spec.stages & ~state.visible_stages[qi]) == 0 &&
                                         (spec.access & ~state.visible_access[qi]) == 0;
                    need_barrier = layout_change || (own_write && !covered);
                    src_stages = own_write ? state.write_stages : VK_PIPELINE_STAGE_2_NONE;
                    if (layout_change) {
                        src_stages |= state.read_stages[qi];
                    }
                    src_access = own_write ? state.write_access : VK_ACCESS_2_NONE;
                }

                if (chain_stages != 0) {
                    need_barrier = true;
                    src_stages |= chain_stages;
                }

                auto const has_prior = own_write || state.read_stages[qi] != 0;
                if (options_.serialize && has_prior) {
                    need_barrier = true;
                    src_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    src_access = VK_ACCESS_2_MEMORY_WRITE_BIT;
                    dst_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    dst_access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
                }
                if (!need_barrier) {
                    return;
                }

                switch (graph_.resources[resource].kind) {
                    case ResourceKind::image:
                        before.images.push_back(ImageBarrier{
                                .resource = resource,
                                .src_stages = src_stages,
                                .src_access = src_access,
                                .dst_stages = dst_stages,
                                .dst_access = dst_access,
                                .old_layout = old_layout,
                                .new_layout = spec.layout,
                        });
                        break;
                    case ResourceKind::buffer:
                        before.buffers.push_back(BufferBarrier{
                                .resource = resource,
                                .src_stages = src_stages,
                                .src_access = src_access,
                                .dst_stages = dst_stages,
                                .dst_access = dst_access,
                        });
                        break;
                    case ResourceKind::token:
                        before.memory.push_back(MemoryBarrier{
                                .src_stages = src_stages,
                                .src_access = src_access,
                                .dst_stages = dst_stages,
                                .dst_access = dst_access,
                        });
                        break;
                }
            }

            GraphDesc const &graph_;
            QueueTopology const &topology_;
            CompileOptions const &options_;
            bool multi_queue_;
            std::vector<Tracked> tracked_;
            std::vector<BarrierSet> acquires_;
            std::vector<BarrierSet> releases_;
            std::vector<Edge> edges_;
            std::vector<PendingTransfer> transfers_;
        };

        struct BatchPlan {
            LogicalQueue queue = LogicalQueue::graphics;
            std::vector<std::int64_t> nodes;
        };

    } // namespace

    static auto declaration_hash(GraphDesc const &graph, QueueTopology const &topology,
                                 CompileOptions const &options) -> std::uint64_t;

    // Builds the plan for one schedule. A node is a position in `order`; the prologue is -1 and the epilogue is
    // order.size(). Every cross-queue edge runs from a lower node to a higher one.
    static auto build_plan(GraphDesc const &graph, QueueTopology const &topology, CompileOptions const &options,
                           bool multi_queue, std::vector<bool> const &live, std::vector<LogicalQueue> const &queues,
                           std::vector<std::uint32_t> const &order) -> CompiledGraph {
        auto const pass_count = graph.passes.size();
        auto const node_count = order.size();
        auto const epilogue_node = static_cast<std::int64_t>(node_count);

        auto result = CompiledGraph{};
        result.pass_culled.assign(pass_count, true);
        for (auto index = std::size_t{0}; index < pass_count; ++index) {
            result.pass_culled[index] = !live[index];
        }
        result.pass_queue = queues;
        result.schedule = order;

        auto tracker = Tracker{graph, topology, options, multi_queue, node_count + 2};
        auto before = std::vector<BarrierSet>(node_count + 2);
        auto swapchain_stages = std::vector<VkPipelineStageFlags2>(node_count + 2, VK_PIPELINE_STAGE_2_NONE);
        auto touches_swapchain = std::vector<bool>(node_count + 2, false);

        for (auto position = std::size_t{0}; position < node_count; ++position) {
            auto const index = order[position];
            auto const node = static_cast<std::int64_t>(position);
            auto const queue = queues[index];
            for (auto const &access: graph.passes[index].accesses) {
                auto const info = use_info(access.use, access.stages);
                if (graph.resources[access.resource].swapchain && !touches_swapchain[position + 1]) {
                    touches_swapchain[position + 1] = true;
                    swapchain_stages[position + 1] = info.stages;
                }
                tracker.apply(access.resource, queue, node,
                              AccessSpec{
                                      .stages = info.stages,
                                      .access = info.access,
                                      .layout = info.layout,
                                      .is_image = info.is_image,
                                      .writes = info.writes,
                                      .discard = access.discard,
                                      .exit = exit_of(access),
                              },
                              before[position + 1]);
            }
        }

        // Epilogue: leave every import in its declared exit state, on the graphics queue.
        auto epilogue_barriers = BarrierSet{};
        for (auto index = std::size_t{0}; index < graph.resources.size(); ++index) {
            auto const &resource = graph.resources[index];
            if (!resource.imported || resource.kind == ResourceKind::token) {
                continue;
            }
            auto const is_image = resource.kind == ResourceKind::image;
            // An UNDEFINED exit layout means "leave it as it is".
            auto const layout = (is_image && resource.exit.layout != VK_IMAGE_LAYOUT_UNDEFINED)
                                        ? resource.exit.layout
                                        : tracker.layout_of(static_cast<std::uint32_t>(index));
            tracker.apply(static_cast<std::uint32_t>(index), LogicalQueue::graphics, epilogue_node,
                          AccessSpec{
                                  .stages = resource.exit.stages,
                                  .access = resource.exit.access,
                                  .layout = is_image ? layout : VK_IMAGE_LAYOUT_UNDEFINED,
                                  .is_image = is_image,
                          },
                          epilogue_barriers);
        }

        // A legacy pass records its own barriers against everything outside the graph, so it is fenced by global
        // barriers: one before it, and one at the start of the next pass on its queue (or the epilogue).
        constexpr auto legacy_fence = MemoryBarrier{
                .src_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .src_access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dst_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .dst_access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        };
        for (auto position = std::size_t{0}; position < node_count; ++position) {
            if (!graph.passes[order[position]].legacy) {
                continue;
            }
            before[position + 1].memory.push_back(legacy_fence);

            auto after = &epilogue_barriers;
            for (auto next = position + 1; next < node_count; ++next) {
                if (queues[order[next]] == queues[order[position]]) {
                    after = &before[next + 1];
                    break;
                }
            }
            after->memory.push_back(legacy_fence);
        }

        // Split each queue's nodes into batches. A node with an incoming cross-queue edge starts a batch and a node
        // with an outgoing one ends it.
        auto has_incoming = std::vector<bool>(node_count + 2, false);
        auto has_outgoing = std::vector<bool>(node_count + 2, false);
        for (auto const &edge: tracker.edges()) {
            has_outgoing[static_cast<std::size_t>(edge.src + 1)] = true;
            has_incoming[static_cast<std::size_t>(edge.dst + 1)] = true;
        }

        auto plans = std::vector<BatchPlan>{};
        for (auto const queue: {LogicalQueue::graphics, LogicalQueue::compute}) {
            auto nodes = std::vector<std::int64_t>{};
            if (queue == LogicalQueue::graphics) {
                nodes.push_back(prologue_node);
            }
            for (auto position = std::size_t{0}; position < node_count; ++position) {
                if (queues[order[position]] == queue) {
                    nodes.push_back(static_cast<std::int64_t>(position));
                }
            }
            if (queue == LogicalQueue::graphics) {
                nodes.push_back(epilogue_node);
            }

            auto current = BatchPlan{.queue = queue, .nodes = {}};
            for (auto const node: nodes) {
                auto const slot = static_cast<std::size_t>(node + 1);
                if (has_incoming[slot] && !current.nodes.empty()) {
                    plans.push_back(std::move(current));
                    current = BatchPlan{.queue = queue, .nodes = {}};
                }
                current.nodes.push_back(node);
                if (has_outgoing[slot]) {
                    plans.push_back(std::move(current));
                    current = BatchPlan{.queue = queue, .nodes = {}};
                }
            }
            if (!current.nodes.empty()) {
                plans.push_back(std::move(current));
            }
        }

        // Submission order: by first node, so every wait refers to an already-submitted batch.
        std::ranges::stable_sort(plans, [](BatchPlan const &lhs, BatchPlan const &rhs) {
            return lhs.nodes.front() < rhs.nodes.front();
        });

        auto node_batch = std::vector<std::uint32_t>(node_count + 2, 0);
        auto next_signal = std::array<std::uint32_t, logical_queue_count>{};
        for (auto batch_index = std::size_t{0}; batch_index < plans.size(); ++batch_index) {
            auto const &plan = plans[batch_index];
            auto &batch = result.batches.emplace_back();
            batch.queue = plan.queue;
            batch.signal_index = next_signal[queue_index(plan.queue)]++;
            batch.is_prologue = plan.nodes.front() == prologue_node;
            batch.acquires = std::move(tracker.acquires_at(plan.nodes.front()));
            batch.releases = std::move(tracker.releases_at(plan.nodes.back()));
            for (auto const node: plan.nodes) {
                node_batch[static_cast<std::size_t>(node + 1)] = static_cast<std::uint32_t>(batch_index);
                if (node == prologue_node || node == epilogue_node) {
                    continue;
                }
                auto const slot = static_cast<std::size_t>(node + 1);
                auto &queue_passes = result.timestamp_passes[queue_index(plan.queue)];
                auto const pass_index = order[static_cast<std::size_t>(node)];
                batch.passes.push_back(CompiledPass{
                        .pass = pass_index,
                        .before = std::move(before[slot]),
                        .timestamp_slot = static_cast<std::uint32_t>(queue_passes.size()),
                });
                queue_passes.push_back(pass_index);
            }
        }
        result.signal_count = next_signal;

        // The epilogue transitions go in the last graphics batch.
        for (auto batch_index = result.batches.size(); batch_index-- > 0;) {
            auto &batch = result.batches[batch_index];
            if (batch.queue == LogicalQueue::graphics) {
                batch.epilogue = std::move(epilogue_barriers);
                batch.signals_render_finished = true;
                break;
            }
        }

        // Waits: one per other queue, at the max signal index, with the union of the covered first-use stages.
        for (auto const &edge: tracker.edges()) {
            auto const src_batch = node_batch[static_cast<std::size_t>(edge.src + 1)];
            auto const dst_batch = node_batch[static_cast<std::size_t>(edge.dst + 1)];
            if (src_batch == dst_batch) {
                continue;
            }
            auto const &source = result.batches[src_batch];
            auto &waiting = result.batches[dst_batch];
            if (source.queue == waiting.queue) {
                continue;
            }
            auto existing = std::ranges::find_if(waiting.waits,
                                                 [&](SemaphoreWait const &wait) { return wait.queue == source.queue; });
            if (existing == waiting.waits.end()) {
                waiting.waits.push_back(SemaphoreWait{
                        .queue = source.queue,
                        .signal_index = source.signal_index,
                        .stages = edge.stages,
                });
            } else {
                existing->signal_index = std::max(existing->signal_index, source.signal_index);
                existing->stages |= edge.stages;
            }
        }

        for (auto const &pending: tracker.transfers()) {
            result.transfers.push_back(OwnershipTransfer{
                    .resource = pending.resource,
                    .is_image = pending.is_image,
                    .from = pending.from,
                    .to = pending.to,
                    .release_batch = node_batch[static_cast<std::size_t>(pending.release_node + 1)],
                    .acquire_batch = node_batch[static_cast<std::size_t>(pending.acquire_node + 1)],
                    .old_layout = pending.old_layout,
                    .new_layout = pending.new_layout,
            });
        }

        // The first batch touching the swapchain waits on image acquisition.
        auto node_of_pass = std::vector<std::size_t>(pass_count, 0);
        for (auto position = std::size_t{0}; position < node_count; ++position) {
            node_of_pass[order[position]] = position;
        }
        for (auto &batch: result.batches) {
            auto done = false;
            for (auto const &pass: batch.passes) {
                auto const slot = node_of_pass[pass.pass] + 1;
                if (touches_swapchain[slot]) {
                    batch.waits_swapchain_acquire = true;
                    batch.swapchain_wait_stages = swapchain_stages[slot];
                    done = true;
                    break;
                }
            }
            if (done) {
                break;
            }
        }

        result.hash = declaration_hash(graph, topology, options);
        return result;
    }

    static auto count_waits(CompiledGraph const &plan) -> std::size_t {
        auto total = std::size_t{0};
        for (auto const &batch: plan.batches) {
            total += batch.waits.size();
        }
        return total;
    }

    static auto declaration_hash(GraphDesc const &graph, QueueTopology const &topology,
                                 CompileOptions const &options) -> std::uint64_t {
        auto hasher = Hasher{};
        hasher.mix(static_cast<std::uint64_t>(options.async_compute));
        hasher.mix(static_cast<std::uint64_t>(options.scheduler));
        hasher.mix(static_cast<std::uint64_t>(options.serialize));
        for (auto queue = std::size_t{0}; queue < logical_queue_count; ++queue) {
            hasher.mix(topology.family[queue]);
            hasher.mix(topology.queue_index[queue]);
        }
        for (auto const &pass: graph.passes) {
            hasher.mix(pass.name);
            hasher.mix(static_cast<std::uint64_t>(pass.type));
            hasher.mix(static_cast<std::uint64_t>(pass.affinity));
            hasher.mix(static_cast<std::uint64_t>(pass.side_effect) | (static_cast<std::uint64_t>(pass.legacy) << 1U));
            for (auto const &access: pass.accesses) {
                hasher.mix(access.resource);
                hasher.mix(access.version);
                hasher.mix(static_cast<std::uint64_t>(access.use));
                hasher.mix(access.stages);
                hasher.mix(static_cast<std::uint64_t>(access.discard));
                hasher.mix(access.exit_use ? 1 + static_cast<std::uint64_t>(*access.exit_use) : 0);
            }
        }
        for (auto const &resource: graph.resources) {
            hasher.mix(resource.name);
            hasher.mix(static_cast<std::uint64_t>(resource.kind));
            hasher.mix(static_cast<std::uint64_t>(resource.sharing));
            hasher.mix(static_cast<std::uint64_t>(resource.entry.layout));
            hasher.mix(static_cast<std::uint64_t>(resource.exit.layout));
            hasher.mix(resource.entry.stages);
            hasher.mix(resource.exit.stages);
        }
        return hasher.state;
    }

    auto compile(GraphDesc const &graph, QueueTopology const &topology,
                 CompileOptions const &options) -> std::expected<CompiledGraph, FrameGraphError> {
        if (auto const valid = validate(graph); !valid) {
            return std::unexpected(valid.error());
        }
        auto const multi_queue =
                options.async_compute && !topology.same_queue(LogicalQueue::graphics, LogicalQueue::compute);
        auto const live = cull(graph);
        auto const queues = resolve_queues(graph, live, multi_queue);

        auto const declared = schedule(graph, live, queues, SchedulerMode::declaration_order);
        auto plan = build_plan(graph, topology, options, multi_queue, live, queues, declared);
        if (options.scheduler == SchedulerMode::overlap) {
            auto const overlapped = schedule(graph, live, queues, SchedulerMode::overlap);
            if (overlapped != declared) {
                // Reordering must not cost more cross-queue waits than the declared order.
                auto candidate = build_plan(graph, topology, options, multi_queue, live, queues, overlapped);
                if (count_waits(candidate) <= count_waits(plan)) {
                    plan = std::move(candidate);
                }
            }
        }
        return plan;
    }

    auto compile(FrameGraph const &graph, QueueTopology const &topology,
                 CompileOptions const &options) -> std::expected<CompiledGraph, FrameGraphError> {
        if (!graph.declaration_errors().empty()) {
            return std::unexpected(graph.declaration_errors().front());
        }
        return compile(graph.description(), topology, options);
    }

} // namespace frame_graph
