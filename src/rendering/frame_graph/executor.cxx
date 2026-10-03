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

        // Records `barriers` as one vkCmdPipelineBarrier2, if there is anything to record.
        auto record_barriers(VkCommandBuffer command_buffer, BarrierSet const &barriers, ExecuteInfo const &info)
                -> std::expected<void, ExecuteError> {
            if (barriers.empty()) {
                return {};
            }

            auto const storage = translate(barriers, info.resources);
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

            // Memory another transient used until now: wait for its accesses before this pass's own barriers.
            if (info.transients != nullptr) {
                for (auto const &aliasing: info.transients->barriers) {
                    if (aliasing.pass != compiled_pass.pass) {
                        continue;
                    }
                    auto handoff = BarrierSet{};
                    handoff.memory.push_back(aliasing.barrier);
                    if (auto recorded = record_barriers(command_buffer, handoff, info); !recorded) {
                        return recorded;
                    }
                }
            }

            if (auto recorded = record_barriers(command_buffer, compiled_pass.before, info); !recorded) {
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
                if (auto recorded = record_barriers(command_buffer, batch.acquires, info); !recorded) {
                    return std::unexpected(recorded.error());
                }

                for (auto const &compiled_pass: batch.passes) {
                    if (auto recorded = record_pass(command_buffer, batch, compiled_pass, info); !recorded) {
                        return std::unexpected(recorded.error());
                    }
                }

                if (auto recorded = record_barriers(command_buffer, batch.releases, info); !recorded) {
                    return std::unexpected(recorded.error());
                }
                if (auto recorded = record_barriers(command_buffer, batch.epilogue, info); !recorded) {
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
