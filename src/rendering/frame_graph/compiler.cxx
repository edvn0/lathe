#include "rendering/frame_graph/compiler.hxx"

#include <algorithm>
#include <bit>

namespace frame_graph {
    namespace {

        struct Tracked {
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            bool has_write = false;
            VkPipelineStageFlags2 write_stages = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 write_access = VK_ACCESS_2_NONE;
            VkPipelineStageFlags2 read_stages = VK_PIPELINE_STAGE_2_NONE; // reads since the last write
            VkPipelineStageFlags2 visible_stages = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 visible_access = VK_ACCESS_2_NONE;
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
                    return fail(FrameGraphErrorType::raster_pass_with_compute_affinity, pass, graph,
                                graph.resources.size());
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

            // Every pass that wrote a version a live pass consumed is already marked above. What remains culled is
            // dead.
            return live;
        }

    } // namespace

    auto compile(GraphDesc const &graph, QueueTopology const &topology,
                 CompileOptions const &options) -> std::expected<CompiledGraph, FrameGraphError> {
        // Single-queue only: with one queue (or async compute off) every pass resolves to graphics.
        if (options.async_compute && !topology.same_queue(LogicalQueue::graphics, LogicalQueue::compute)) {
            return std::unexpected(FrameGraphError{FrameGraphErrorType::unsupported_topology, {}, {}});
        }
        if (auto const valid = validate(graph); !valid) {
            return std::unexpected(valid.error());
        }

        auto const pass_count = graph.passes.size();
        auto result = CompiledGraph{};
        result.pass_culled.assign(pass_count, true);
        result.pass_queue.assign(pass_count, LogicalQueue::graphics);

        auto const live = cull(graph);
        for (auto index = std::size_t{0}; index < pass_count; ++index) {
            result.pass_culled[index] = !live[index];
        }

        auto tracked = std::vector<Tracked>(graph.resources.size());
        for (auto index = std::size_t{0}; index < graph.resources.size(); ++index) {
            auto const &resource = graph.resources[index];
            if (resource.imported) {
                tracked[index] = Tracked{
                        .layout = resource.entry.layout,
                        .has_write = true,
                        .write_stages = resource.entry.stages,
                        .write_access = resource.entry.access,
                };
            }
        }

        auto &batch = result.batches.emplace_back();
        batch.queue = LogicalQueue::graphics;
        batch.signals_render_finished = true;

        auto swapchain_seen = false;
        for (auto index = std::size_t{0}; index < pass_count; ++index) {
            if (!live[index]) {
                continue;
            }
            auto const &pass = graph.passes[index];
            auto compiled = CompiledPass{.pass = static_cast<std::uint32_t>(index)};

            auto const emit = [&](std::uint32_t resource_index, ImageBarrier image, BufferBarrier buffer,
                                  MemoryBarrier memory) {
                switch (graph.resources[resource_index].kind) {
                    case ResourceKind::image:
                        compiled.before.images.push_back(image);
                        break;
                    case ResourceKind::buffer:
                        compiled.before.buffers.push_back(buffer);
                        break;
                    case ResourceKind::token:
                        compiled.before.memory.push_back(memory);
                        break;
                }
            };

            for (auto const &access: pass.accesses) {
                auto const &resource = graph.resources[access.resource];
                auto &state = tracked[access.resource];
                auto const info = use_info(access.use, access.stages);

                if (resource.swapchain && !swapchain_seen) {
                    swapchain_seen = true;
                    batch.waits_swapchain_acquire = true;
                    batch.swapchain_wait_stages = info.stages;
                }

                auto const layout_change = info.is_image && state.layout != info.layout;
                auto const prior_stages = state.write_stages | state.read_stages;
                auto const has_prior = state.has_write || state.read_stages != 0;
                auto const old_layout = access.discard ? VK_IMAGE_LAYOUT_UNDEFINED : state.layout;

                auto need_barrier = false;
                auto src_stages = VkPipelineStageFlags2{VK_PIPELINE_STAGE_2_NONE};
                auto src_access = VkAccessFlags2{VK_ACCESS_2_NONE};
                auto dst_stages = info.stages;
                auto dst_access = info.access;

                if (info.writes) {
                    // WAW, WAR and RAW-with-write: wait for everything before, make the last write available.
                    need_barrier = has_prior || layout_change;
                    src_stages = prior_stages;
                    src_access = access.discard ? VK_ACCESS_2_NONE : state.write_access;
                } else {
                    auto const covered =
                            (info.stages & ~state.visible_stages) == 0 && (info.access & ~state.visible_access) == 0;
                    need_barrier = layout_change || (state.has_write && !covered);
                    src_stages = layout_change ? prior_stages : state.write_stages;
                    src_access = state.write_access;
                }

                if (options.serialize && has_prior) {
                    need_barrier = true;
                    src_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    src_access = VK_ACCESS_2_MEMORY_WRITE_BIT;
                    dst_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    dst_access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
                }

                if (need_barrier) {
                    emit(access.resource,
                         ImageBarrier{
                                 .resource = access.resource,
                                 .src_stages = src_stages,
                                 .src_access = src_access,
                                 .dst_stages = dst_stages,
                                 .dst_access = dst_access,
                                 .old_layout = old_layout,
                                 .new_layout = info.layout,
                         },
                         BufferBarrier{
                                 .resource = access.resource,
                                 .src_stages = src_stages,
                                 .src_access = src_access,
                                 .dst_stages = dst_stages,
                                 .dst_access = dst_access,
                         },
                         MemoryBarrier{
                                 .src_stages = src_stages,
                                 .src_access = src_access,
                                 .dst_stages = dst_stages,
                                 .dst_access = dst_access,
                         });
                }

                if (info.writes) {
                    state.has_write = true;
                    state.write_stages = info.stages;
                    state.write_access = info.access;
                    state.read_stages = 0;
                    state.visible_stages = 0;
                    state.visible_access = 0;
                } else if (need_barrier) {
                    // A layout transition is itself a write the new layout's readers must see.
                    if (layout_change) {
                        state.has_write = true;
                        state.write_stages = info.stages;
                        state.write_access = info.access;
                        state.read_stages = 0;
                        state.visible_stages = 0;
                        state.visible_access = 0;
                    }
                    state.visible_stages |= info.stages;
                    state.visible_access |= info.access;
                    state.read_stages |= info.stages;
                } else {
                    state.read_stages |= info.stages;
                    state.visible_stages |= info.stages;
                    state.visible_access |= info.access;
                }
                if (info.is_image) {
                    state.layout = info.layout;
                }
            }

            auto const slot = static_cast<std::uint32_t>(result.timestamp_passes[0].size());
            compiled.timestamp_slot = slot;
            result.timestamp_passes[0].push_back(static_cast<std::uint32_t>(index));
            batch.passes.push_back(std::move(compiled));
        }

        // Epilogue: leave every import in its declared exit state.
        for (auto index = std::size_t{0}; index < graph.resources.size(); ++index) {
            auto const &resource = graph.resources[index];
            if (!resource.imported) {
                continue;
            }
            auto const &state = tracked[index];
            auto const prior_stages = state.write_stages | state.read_stages;
            auto const exit_layout = resource.exit.layout;
            switch (resource.kind) {
                case ResourceKind::image: {
                    if (state.layout == exit_layout &&
                        (resource.exit.stages == 0 || (resource.exit.stages & ~state.visible_stages) == 0)) {
                        continue;
                    }
                    batch.epilogue.images.push_back(ImageBarrier{
                            .resource = static_cast<std::uint32_t>(index),
                            .src_stages = prior_stages,
                            .src_access = state.write_access,
                            .dst_stages = resource.exit.stages,
                            .dst_access = resource.exit.access,
                            .old_layout = state.layout,
                            .new_layout = exit_layout,
                    });
                    break;
                }
                case ResourceKind::buffer: {
                    if ((resource.exit.stages & ~state.visible_stages) == 0 &&
                        (resource.exit.access & ~state.visible_access) == 0) {
                        continue;
                    }
                    batch.epilogue.buffers.push_back(BufferBarrier{
                            .resource = static_cast<std::uint32_t>(index),
                            .src_stages = prior_stages,
                            .src_access = state.write_access,
                            .dst_stages = resource.exit.stages,
                            .dst_access = resource.exit.access,
                    });
                    break;
                }
                case ResourceKind::token:
                    break;
            }
        }

        auto hasher = Hasher{};
        hasher.mix(static_cast<std::uint64_t>(options.async_compute));
        hasher.mix(static_cast<std::uint64_t>(options.serialize));
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
            }
        }
        for (auto const &resource: graph.resources) {
            hasher.mix(resource.name);
            hasher.mix(static_cast<std::uint64_t>(resource.kind));
            hasher.mix(static_cast<std::uint64_t>(resource.entry.layout));
            hasher.mix(static_cast<std::uint64_t>(resource.exit.layout));
            hasher.mix(resource.entry.stages);
            hasher.mix(resource.exit.stages);
        }
        result.hash = hasher.state;
        return result;
    }

    auto compile(FrameGraph const &graph, QueueTopology const &topology,
                 CompileOptions const &options) -> std::expected<CompiledGraph, FrameGraphError> {
        if (!graph.declaration_errors().empty()) {
            return std::unexpected(graph.declaration_errors().front());
        }
        return compile(graph.description(), topology, options);
    }

} // namespace frame_graph
