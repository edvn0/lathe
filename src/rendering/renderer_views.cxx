#include <algorithm>
#include <array>
#include <expected>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "gpu/context.hxx"
#include "rendering/renderer.hxx"

namespace {
    auto make_error(RendererErrorType type) -> RendererError {
        return RendererError{
                .type = type,
        };
    }

    constexpr auto queue_slot(LogicalQueue queue) noexcept -> std::size_t { return static_cast<std::size_t>(queue); }
}

auto Renderer::swap_view_state(ViewState &other) noexcept -> void {
    using std::swap;

    swap(frames_, other.frames);
    swap(ubos_, other.ubos);
    swap(extent_, other.extent);

    swap(hiz_, other.hiz);
    swap(hiz_history_view_projection_, other.hiz_history_view_projection);
    swap(hiz_history_valid_, other.hiz_history_valid);

    swap(view_mode_, other.view_mode);
    swap(lights_dirty_mask_, other.lights_dirty_mask);

    swap(shadow_atlas_, other.shadow_atlas);
    swap(shadow_cascade_cache_, other.shadow_cascade_cache);
    swap(shadow_frame_, other.shadow_frame);
    swap(cached_shadow_caster_revision_, other.cached_shadow_caster_revision);
    swap(shadow_scene_signature_, other.shadow_scene_signature);
    swap(cached_shadow_light_direction_, other.cached_shadow_light_direction);
    swap(cached_shadow_depth_bias_constant_, other.cached_shadow_depth_bias_constant);
    swap(cached_shadow_depth_bias_slope_, other.cached_shadow_depth_bias_slope);
    swap(shadow_scene_signature_valid_, other.shadow_scene_signature_valid);
    swap(shadow_global_state_valid_, other.shadow_global_state_valid);
    swap(shadow_atlas_initialized_, other.shadow_atlas_initialized);

    swap(plan_cache_, other.plan_cache);
    swap(frame_plan_, other.frame_plan);
    swap(transient_allocator_, other.transient_allocator);
    swap(pass_profiler_, other.pass_profiler);
    swap(logged_plan_misses_, other.logged_plan_misses);
    swap(logged_transient_bytes_, other.logged_transient_bytes);

    swap(last_frame_timings_, other.last_frame_timings);
    swap(last_frame_stats_, other.last_frame_stats);
    swap(last_cluster_stats_, other.last_cluster_stats);

    swap(history_samples_, other.samples);
}

auto Renderer::view_state(ViewId view) noexcept -> ViewState * {
    if (view == primary_view || view > extra_views_.size()) {
        return nullptr;
    }

    return extra_views_[view - 1].get();
}

auto Renderer::view_state(ViewId view) const noexcept -> ViewState const * {
    if (view == primary_view || view > extra_views_.size()) {
        return nullptr;
    }

    return extra_views_[view - 1].get();
}

auto Renderer::has_pending_extra_views() const noexcept -> bool {
    return std::ranges::any_of(extra_views_, [](auto const &view) { return view && view->camera_set; });
}

auto Renderer::create_view(VkExtent2D extent) -> std::expected<ViewId, RendererError> {
    if (!initialized_ || extent.width == 0 || extent.height == 0) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto view = std::make_unique<ViewState>();
    view->extent = extent;
    view->samples = samples_;
    view->lights_dirty_mask = (1U << frames_in_flight) - 1U;

    view->frames.resize(frames_in_flight);
    for (std::uint32_t frame_index = 0; frame_index < frames_in_flight; ++frame_index) {
        if (auto created = create_frame_resources(view->frames[frame_index], extent, frame_index); !created) {
            return std::unexpected(created.error());
        }
    }

    auto hiz = create_hiz_pyramid(extent);
    if (!hiz) {
        return std::unexpected(hiz.error());
    }
    view->hiz = std::move(*hiz);

    auto atlas = create_shadow_atlas();
    if (!atlas) {
        return std::unexpected(atlas.error());
    }
    view->shadow_atlas = std::move(*atlas);

    auto ubos = create_ubos();
    if (!ubos) {
        return std::unexpected(ubos.error());
    }
    view->ubos = std::move(*ubos);

    view->transient_allocator->initialize(context_, image_storage_, frames_in_flight);

    if (!view->pass_profiler->initialize(frame_graph::PassProfilerCreateInfo{
                .physical_device = context_.physical_device,
                .device = context_.device,
                .queue_family = {context_.queue_families.graphics, context_.queue_families.compute},
                .timestamp_period = timestamp_period_,
                .slots = frames_in_flight,
                .max_passes = 64,
        })) {
        error("Failed to create the frame graph pass profiler for a view");
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    auto const free_slot = std::ranges::find_if(extra_views_, [](auto const &entry) { return entry == nullptr; });
    if (free_slot != extra_views_.end()) {
        *free_slot = std::move(view);
        return static_cast<ViewId>(std::distance(extra_views_.begin(), free_slot) + 1);
    }

    extra_views_.push_back(std::move(view));
    return static_cast<ViewId>(extra_views_.size());
}

auto Renderer::destroy_view(ViewId view) -> void {
    if (view == primary_view || view > extra_views_.size() || extra_views_[view - 1] == nullptr) {
        return;
    }

    // The view's buffers and images may still be in flight.
    static_cast<void>(wait_idle());

    extra_views_[view - 1].reset();
}

auto Renderer::resize_view(ViewId view, VkExtent2D extent) -> std::expected<void, RendererError> {
    if (view == primary_view) {
        return resize(extent);
    }

    auto *const state = view_state(view);
    if (state == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    ViewScope const scope{*this, *state};
    return resize(extent);
}

auto Renderer::set_view_camera(ViewId view, CameraMatrices const &camera) -> void {
    if (auto *const state = view_state(view); state != nullptr) {
        state->camera = camera;
        state->camera_set = true;
    }
}

auto Renderer::set_view_mode(ViewId view, ViewMode mode) -> void {
    if (view == primary_view) {
        view_mode_ = mode;
    } else if (auto *const state = view_state(view); state != nullptr) {
        state->view_mode = mode;
    }
}

auto Renderer::view_mode(ViewId view) const noexcept -> ViewMode {
    if (auto const *const state = view_state(view); state != nullptr) {
        return state->view_mode;
    }

    return view_mode_;
}

auto Renderer::view_extent(ViewId view) const noexcept -> VkExtent2D {
    if (auto const *const state = view_state(view); state != nullptr) {
        return state->extent;
    }

    return extent_;
}

auto Renderer::view_viewport_target(ViewId view, std::uint32_t frame_index) const noexcept -> ImageHandle {
    if (auto const *const state = view_state(view); state != nullptr) {
        return frame_index < state->frames.size() ? state->frames[frame_index].viewport_target.handle()
                                                  : ImageHandle{};
    }

    return frame_index < frames_.size() ? frames_[frame_index].viewport_target.handle() : ImageHandle{};
}

auto Renderer::end_prepare() noexcept -> void {
    hold_submissions_ = false;
    clear_submissions();
}

auto Renderer::prepare_frame(VkCommandBuffer command_buffer, CameraMatrices const &matrices, std::uint32_t frame_index)
        -> std::expected<void, RendererError> {
    // The extra views build their own batches from the same submissions, so those stay until the last view is done.
    hold_submissions_ = has_pending_extra_views();

    auto primary = prepare_frame_view(command_buffer, matrices, frame_index, false);

    if (!hold_submissions_) {
        return primary;
    }

    for (auto const &entry: extra_views_) {
        if (entry == nullptr || !entry->camera_set) {
            continue;
        }

        entry->camera_set = false;
        entry->prepared = false;

        if (!primary) {
            continue;
        }

        ViewScope const scope{*this, *entry};
        auto const prepared = prepare_frame_view(command_buffer, entry->camera, frame_index, true);

        if (!prepared) {
            warn("Could not prepare an extra view: {}", describe(prepared.error()));
            continue;
        }

        entry->prepared = true;
    }

    end_prepare();

    return primary;
}

auto Renderer::record_frame(FrameRecordInfo const &info) -> std::expected<void, RendererError> {
    std::vector<ViewState *> views;
    for (auto const &entry: extra_views_) {
        if (entry != nullptr && entry->prepared) {
            views.push_back(entry.get());
            entry->prepared = false;
        }
    }

    if (views.empty()) {
        return record_frame_view(info, false);
    }

    // Each view is its own frame graph with its own submit batches. They go to the queues in one list: the frame's
    // command buffer (the uploads prepare recorded) first, then each extra view, then the primary view, which samples
    // the extra views' targets in its UI pass.
    auto const fresh_command_buffer = [&]() -> std::expected<VkCommandBuffer, RendererError> {
        auto buffer = context_.queue_set.command_buffer(LogicalQueue::graphics);
        if (!buffer) {
            return std::unexpected(make_error(RendererErrorType::device_error));
        }

        return *buffer;
    };

    std::vector<std::vector<SubmitBatch>> recorded;
    std::expected<void, RendererError> primary_result{};

    // Records every view into fresh command buffers. Run twice when a view reallocated its transients: that updates the
    // resource table, which a command buffer recorded earlier had bound.
    // Recording marks the shadow atlases as initialised, which the discarded first pass must not leave behind: the
    // atlases' layout is still what it was.
    auto const atlas_flags = [&] {
        std::vector<bool> flags{shadow_atlas_initialized_, local_shadow_atlas_initialized_};
        for (auto *const view: views) {
            flags.push_back(view->shadow_atlas_initialized);
        }
        return flags;
    }();
    auto const restore_atlas_flags = [&] {
        shadow_atlas_initialized_ = atlas_flags[0];
        local_shadow_atlas_initialized_ = atlas_flags[1];
        for (auto index = std::size_t{0}; index < views.size(); ++index) {
            views[index]->shadow_atlas_initialized = atlas_flags[2 + index];
        }
    };

    auto const record_views = [&]() -> bool {
        recorded.clear();
        transients_reallocated_ = false;

        for (auto *const view: views) {
            auto prologue = fresh_command_buffer();
            if (!prologue) {
                warn("Could not allocate a command buffer for an extra view");
                continue;
            }

            auto view_info = info;
            view_info.command_buffer = *prologue;

            {
                ViewScope const scope{*this, *view};
                if (auto const result = record_frame_view(view_info, true); !result) {
                    warn("Could not record an extra view: {}", describe(result.error()));
                    continue;
                }
            }

            // The first batch records into the prologue, which the executor leaves open for the caller.
            vkEndCommandBuffer(*prologue);
            recorded.push_back(std::move(submit_batches_));
        }

        if (recorded.empty()) {
            return false;
        }

        auto primary_prologue = fresh_command_buffer();
        if (!primary_prologue) {
            primary_result = std::unexpected(primary_prologue.error());
            return false;
        }

        auto primary_info = info;
        primary_info.command_buffer = *primary_prologue;

        primary_result = record_frame_view(primary_info, false);
        if (!primary_result) {
            return false;
        }

        vkEndCommandBuffer(*primary_prologue);
        recorded.push_back(std::move(submit_batches_));
        return true;
    };

    if (!record_views()) {
        if (!primary_result) {
            return primary_result;
        }
        return record_frame_view(info, false);
    }

    if (transients_reallocated_) {
        restore_atlas_flags();
    }

    if (transients_reallocated_ && !record_views()) {
        return primary_result ? record_frame_view(info, false) : primary_result;
    }

    std::vector<SubmitBatch> merged;
    merged_waits_.clear();

    auto counts = std::array<std::uint32_t, gpu_queue_count>{};

    merged.push_back(SubmitBatch{
            .queue = LogicalQueue::graphics,
            .command_buffer = info.command_buffer,
            .signal_index = counts[queue_slot(LogicalQueue::graphics)]++,
    });

    for (auto index = std::size_t{0}; index < recorded.size(); ++index) {
        auto const &batches = recorded[index];
        auto const is_primary = index + 1 == recorded.size();

        // Every graph numbers its signals from zero per queue; a wait names a signal by that number.
        auto const base = counts;

        for (auto batch: batches) {
            // Only the primary view's graph presents: the extra views never touch the swapchain, and the planner marks
            // the last batch of each graph as the one that finishes the frame.
            if (!is_primary) {
                batch.waits_swapchain_acquire = false;
                batch.signals_render_finished = false;
            }

            auto &waits = merged_waits_.emplace_back(batch.waits.begin(), batch.waits.end());
            for (auto &wait: waits) {
                wait.signal_index += base[queue_slot(wait.queue)];
            }

            batch.waits = waits;
            batch.signal_index += base[queue_slot(batch.queue)];
            merged.push_back(batch);
            ++counts[queue_slot(batch.queue)];
        }
    }

    submit_batches_ = std::move(merged);

    return {};
}
