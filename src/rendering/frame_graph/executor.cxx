#include "rendering/frame_graph/executor.hxx"

#include <format>
#include <optional>

#include "rendering/frame_graph/pass_context.hxx"

namespace frame_graph {
    namespace {

        [[maybe_unused]] constexpr auto queue_index(LogicalQueue queue) noexcept -> std::size_t {
            return static_cast<std::size_t>(queue);
        }

        auto describe(TranslateFailure const &failure, GraphDesc const &graph) -> std::string {
            auto const name = failure.resource < graph.resources.size() ? graph.resources[failure.resource].name
                                                                        : std::string{"<out of range>"};
            return std::format("no {} handle for resource '{}'",
                               failure.kind == TranslateFailureKind::missing_image ? "image" : "buffer", name);
        }

        // The stages and accesses a command buffer of a compute-only queue family may name in a barrier.
        constexpr auto compute_family_stages =
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
                VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_HOST_BIT |
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
                VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        constexpr auto compute_family_access =
                VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
                VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

        // A barrier recorded on a compute-only family cannot name graphics stages: a resource a graphics pass reads
        // is acquired on the compute queue by a barrier that carries the compute part of its scope only, and the
        // graphics reader's own barrier covers the rest.
        auto restrict_to_compute_family(BarrierSet barriers) -> BarrierSet {
            for (auto &image: barriers.images) {
                image.src_stages &= compute_family_stages;
                image.dst_stages &= compute_family_stages;
                image.src_access &= compute_family_access;
                image.dst_access &= compute_family_access;
            }
            for (auto &buffer: barriers.buffers) {
                buffer.src_stages &= compute_family_stages;
                buffer.dst_stages &= compute_family_stages;
                buffer.src_access &= compute_family_access;
                buffer.dst_access &= compute_family_access;
            }
            for (auto &memory: barriers.memory) {
                memory.src_stages &= compute_family_stages;
                memory.dst_stages &= compute_family_stages;
                memory.src_access &= compute_family_access;
                memory.dst_access &= compute_family_access;
            }
            return barriers;
        }

        // Whether batches of `queue` are recorded on a family that cannot run graphics work: the compute queue of a
        // topology where it has a family of its own.
        auto is_compute_only(LogicalQueue queue, ExecuteInfo const &info) -> bool {
            auto const topology = info.queue_set.topology();
            return queue == LogicalQueue::compute &&
                   !topology.same_family(LogicalQueue::graphics, LogicalQueue::compute);
        }

        // Records `barriers` as one vkCmdPipelineBarrier2, if there is anything to record.
        auto record_barriers(VkCommandBuffer command_buffer, BarrierSet const &barriers, ExecuteInfo const &info,
                             bool compute_only) -> std::expected<void, ExecuteError> {
            if (barriers.empty()) {
                return {};
            }

            auto const storage =
                    translate(compute_only ? restrict_to_compute_family(barriers) : barriers, info.resources);
            if (!storage) {
                return std::unexpected(ExecuteError{describe(storage.error(), info.graph)});
            }

            auto const dependency = storage->info();
            vkCmdPipelineBarrier2(command_buffer, &dependency);
            return {};
        }

        auto has_content(Batch const &batch) noexcept -> bool {
            return !batch.passes.empty() || !batch.acquires.empty() || !batch.releases.empty() ||
                   !batch.epilogue.empty();
        }

        auto record_pass(VkCommandBuffer command_buffer, Batch const &batch, CompiledPass const &compiled_pass,
                         ExecuteInfo const &info) -> std::expected<void, ExecuteError> {
            auto const &pass = info.graph.passes[compiled_pass.pass];
            auto const compute_only = is_compute_only(batch.queue, info);

            // Memory another transient used until now: wait for its accesses before this pass's own barriers.
            if (info.transients != nullptr) {
                for (auto const &aliasing: info.transients->barriers) {
                    if (aliasing.pass != compiled_pass.pass) {
                        continue;
                    }
                    auto handoff = BarrierSet{};
                    handoff.memory.push_back(aliasing.barrier);
                    if (auto recorded = record_barriers(command_buffer, handoff, info, compute_only); !recorded) {
                        return recorded;
                    }
                }
            }

            if (auto recorded = record_barriers(command_buffer, compiled_pass.before, info, compute_only); !recorded) {
                return recorded;
            }

            auto const label = pass.profile.label.empty() ? std::string_view{pass.name} : pass.profile.label;
            auto const name_id = pass.profile.name_id.empty() ? std::string_view{pass.name} : pass.profile.name_id;

#ifdef TRACY_ENABLE
            auto const *location = info.profiler.source_location(label, pass.profile.color);
            tracy::ScopedZone const cpu_zone{location};
            auto *const tracy_context = info.tracy_contexts[queue_index(batch.queue)];
            // Scoped so the zone closes with the pass; without a context there is nothing to time on the GPU.
            std::optional<tracy::VkCtxScope> gpu_zone;
            if (tracy_context != nullptr) {
                gpu_zone.emplace(tracy_context, location, command_buffer, true);
            }
#endif

            info.profiler.write_begin(command_buffer, batch.queue, info.frame_index, compiled_pass.timestamp_slot,
                                      name_id, label);

            // A raster pass that declared attachments is wrapped in dynamic rendering; the barrier above has put them
            // in their attachment layouts.
            auto rendering = std::optional<RenderingStorage>{};
            if (pass.rendering) {
                auto desc = *pass.rendering;
                if (desc.render_area.extent.width == 0 || desc.render_area.extent.height == 0) {
                    auto const first = !desc.colors.empty() ? desc.colors.front().resource
                                       : desc.depth         ? desc.depth->resource
                                                            : 0U;
                    if (auto const *image = info.resources.image(first); image != nullptr) {
                        desc.render_area =
                                VkRect2D{.offset = {0, 0}, .extent = {image->extent.width, image->extent.height}};
                    }
                }
                auto storage = translate(desc, info.resources);
                if (!storage) {
                    return std::unexpected(ExecuteError{describe(storage.error(), info.graph)});
                }
                rendering = std::move(*storage);
                auto const rendering_info = rendering->info();
                vkCmdBeginRendering(command_buffer, &rendering_info);
            }

            if (compiled_pass.pass < info.records.size() && info.records[compiled_pass.pass]) {
                auto context = PassContext{
                        .command_buffer = command_buffer,
                        .frame_index = info.frame_index,
                        .queue = batch.queue,
                        .resources = &info.resources,
                };
                info.records[compiled_pass.pass](context);
            }

            if (rendering) {
                vkCmdEndRendering(command_buffer);
            }

            info.profiler.write_end(command_buffer, batch.queue, info.frame_index, compiled_pass.timestamp_slot);
            return {};
        }

    } // namespace

    auto record(ExecuteInfo const &info) -> std::expected<std::vector<SubmitBatch>, ExecuteError> {
        auto submits = std::vector<SubmitBatch>{};
        submits.reserve(info.compiled.batches.size());

        for (auto index = std::size_t{0}; index < info.compiled.batches.size(); ++index) {
            auto const &batch = info.compiled.batches[index];

            auto command_buffer = VkCommandBuffer{VK_NULL_HANDLE};
            if (index == 0) {
                // The prologue batch: the frame's own graphics command buffer.
                command_buffer = info.prologue;
            } else if (has_content(batch)) {
                auto fresh = info.queue_set.command_buffer(batch.queue);
                if (!fresh) {
                    return std::unexpected(ExecuteError{std::format("no command buffer for batch {}", index)});
                }
                command_buffer = *fresh;
            }

            if (command_buffer != VK_NULL_HANDLE) {
                auto const compute_only = is_compute_only(batch.queue, info);
                if (auto recorded = record_barriers(command_buffer, batch.acquires, info, compute_only); !recorded) {
                    return std::unexpected(recorded.error());
                }

                for (auto const &compiled_pass: batch.passes) {
                    if (auto recorded = record_pass(command_buffer, batch, compiled_pass, info); !recorded) {
                        return std::unexpected(recorded.error());
                    }
                }

                if (auto recorded = record_barriers(command_buffer, batch.releases, info, compute_only); !recorded) {
                    return std::unexpected(recorded.error());
                }
                if (auto recorded = record_barriers(command_buffer, batch.epilogue, info, compute_only); !recorded) {
                    return std::unexpected(recorded.error());
                }

                // The prologue buffer stays open for the frame's owner to end.
                if (index != 0) {
                    auto const ended = vkEndCommandBuffer(command_buffer);
                    if (ended != VK_SUCCESS) {
                        return std::unexpected(
                                ExecuteError{std::format("vkEndCommandBuffer failed for batch {} with VkResult {}",
                                                         index, static_cast<int>(ended))});
                    }
                }
            }

            submits.push_back(SubmitBatch{
                    .queue = batch.queue,
                    .command_buffer = command_buffer,
                    .waits = batch.waits,
                    .signal_index = batch.signal_index,
                    .waits_swapchain_acquire = batch.waits_swapchain_acquire,
                    .swapchain_wait_stages = batch.swapchain_wait_stages,
                    .signals_render_finished = batch.signals_render_finished,
            });
        }

        return submits;
    }

} // namespace frame_graph
