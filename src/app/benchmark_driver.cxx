#include "app/benchmark_driver.hxx"

#include <volk.h>

#include <algorithm>
#include <format>
#include <utility>

#include "app/application.hxx"
#include "app/benchmark_environment.hxx"
#include "core/config.hxx"
#include "core/logger.hxx"
#include "core/memory_tracker.hxx"
#include "core/random.hxx"
#include "gpu/context.hxx"
#include "rendering/renderer.hxx"
#include "rendering/screenshot.hxx"

namespace {

    constexpr std::uint32_t minimum_case_warmup_frames = 4;

    [[nodiscard]] auto present_mode_name(VkPresentModeKHR mode) -> std::string {
        switch (mode) {
            case VK_PRESENT_MODE_IMMEDIATE_KHR:
                return "immediate";
            case VK_PRESENT_MODE_MAILBOX_KHR:
                return "mailbox";
            case VK_PRESENT_MODE_FIFO_KHR:
                return "fifo";
            case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
                return "fifo_relaxed";
            default:
                return std::format("other ({})", static_cast<int>(mode));
        }
    }

    [[nodiscard]] auto device_type_name(VkPhysicalDeviceType type) -> std::string {
        switch (type) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                return "integrated";
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                return "discrete";
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                return "virtual";
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                return "cpu";
            default:
                return "other";
        }
    }

    auto log_case_summary(BenchmarkCaseResult const &result) -> void {
        auto const &analysis = result.analysis;
        info("Benchmark {} (repeat {}): displayed p50 {:.2f} ms / p99 {:.2f} ms / max {:.2f} ms, GPU p50 {:.2f} ms, "
             "CPU busy p50 {:.2f} ms, {} hitch(es), {:.1f}% over the {:.2f} ms budget; frames limited by GPU {:.0f}% / "
             "CPU {:.0f}% / presentation {:.0f}%",
             result.id.key(), result.id.repeat, analysis.displayed.median_ms, analysis.displayed.p99_ms,
             analysis.displayed.max_ms, analysis.gpu_frame.median_ms, analysis.cpu_busy.median_ms,
             analysis.hitches.size(),
             analysis.budget.frames == 0 ? 0.0
                                         : 100.0 * static_cast<double>(analysis.budget.displayed_over_budget) /
                                                   static_cast<double>(analysis.budget.frames),
             analysis.budget.budget_ms, analysis.bound.gpu_bound_fraction() * 100.0F,
             analysis.bound.cpu_bound_fraction() * 100.0F, analysis.bound.presentation_bound_fraction() * 100.0F);
    }

}

auto BenchmarkDriver::create(BenchmarkOptions options,
                             Application const &application) -> std::expected<BenchmarkDriver, std::string> {
    BenchmarkDriver driver;

    auto const has_game_path = application.game && !application.game->benchmark_camera_path().empty();

    if (options.mode == BenchmarkMode::single) {
        if (!has_game_path) {
            return std::unexpected(std::string{"--benchmark: this game defines no benchmark_camera_path()"});
        }

        driver.scenarios_.push_back(BenchmarkScenario{.info = {.name = "game"}, .source = BenchmarkSceneSource::game});
        driver.cases_.push_back(BenchmarkCase{.scenario = 0, .id = BenchmarkCaseId{.scenario = "game"}});

        if (options.keyframe_screenshots) {
            info("Benchmark: keyframe screenshots on");
        }
    } else {
        driver.scenarios_ = builtin_benchmark_scenarios(has_game_path);

        auto planned = plan_benchmark_cases(scenario_infos(driver.scenarios_), options);
        if (!planned) {
            return std::unexpected(planned.error());
        }
        driver.cases_ = std::move(*planned);

        if (options.keyframe_screenshots) {
            warn("--benchmark-screenshots only applies to --benchmark=; ignored for the suite");
            options.keyframe_screenshots = false;
        }

        info("Benchmark suite: {} run(s) of {} measured frames into {}", driver.cases_.size(), options.frame_count,
             options.output_path.string());
        for (auto const &scenario: driver.scenarios_) {
            info("  {}: {}", scenario.info.name, scenario.description);
        }
    }

    driver.options_ = std::move(options);
    return driver;
}

auto BenchmarkDriver::start_case(Application &application) -> void {
    auto const &current = cases_[case_index_];
    auto const &scenario = scenarios_[current.scenario];

    set_fixed_random_seed(options_.seed);

    std::vector<CameraKeyframe> keyframes;

    if (scenario.source == BenchmarkSceneSource::game) {
        application.terrain_enabled = true;
        application.game_hooks_enabled = true;

        if (options_.mode == BenchmarkMode::suite) {
            application.game->on_populate(*application.editor_scene, *application.renderer, application.engine_models);
            application.mark_editor_scene_clean();

            release_scenario_materials(application);
        }

        keyframes = application.game->benchmark_camera_path();
    } else {
        application.terrain_enabled = false;
        application.game_hooks_enabled = false;

        if (auto const waited = application.renderer->wait_idle(); !waited) {
            warn("Benchmark: could not wait for the GPU before repopulating");
        }
        application.editor_scene->get_registry().clear();
        release_scenario_materials(application);
        scenario.populate(BenchmarkScenarioContext{
                .scene = *application.editor_scene,
                .renderer = *application.renderer,
                .engine_models = application.engine_models,
                .load = current.id.load,
                .owned_materials = scenario_materials_,
        });
        application.mark_editor_scene_clean();

        keyframes = scenario.camera_path(current.id.load);
    }

    render_scale_percent_ = scenario.load_target == BenchmarkLoadTarget::render_scale ? current.id.load : 100U;

    auto run_options = options_;
    if (options_.mode == BenchmarkMode::suite) {
        run_options.warmup_frame_count = std::max(run_options.warmup_frame_count, minimum_case_warmup_frames);
        run_options.max_warmup_frame_count =
                std::max(run_options.max_warmup_frame_count, run_options.warmup_frame_count);
    }

    info("Benchmark case {}/{}: {} (repeat {}), {} keyframes", case_index_ + 1, cases_.size(), current.id.key(),
         current.id.repeat, keyframes.size());

    run_.emplace(std::move(run_options), std::move(keyframes));

    thermals_start_ = sample_thermals();

    last_events_ = perf_events::snapshot();
    auto const memory = MemoryTracker::stats();
    last_allocations_ = memory.total_allocations;
    last_allocated_bytes_ = memory.total_allocated_bytes;
}

auto BenchmarkDriver::release_scenario_materials(Application &application) -> void {
    for (auto const material: scenario_materials_) {
        application.renderer->release_material(material);
    }
    scenario_materials_.clear();
}

auto BenchmarkDriver::begin_frame(Application &application) -> void {
    if (finished()) {
        return;
    }

    if (!run_) {
        start_case(application);
    }

    application.elapsed_time = run_->simulated_time();

    auto const keyframe = run_->camera();
    application.camera.look_at(keyframe.position, keyframe.target);

    if (run_->options().keyframe_screenshots && run_->at_keyframe()) {
        application.renderer->request_screenshot(ScreenshotSource::viewport);
    }
}

auto BenchmarkDriver::render_size(BenchmarkRenderSize panel) const noexcept -> BenchmarkRenderSize {
    auto const base = options_.render_size.value_or(panel);
    if (render_scale_percent_ == 100) {
        return base;
    }

    auto const scale = [this](std::uint32_t side) {
        return std::max(1U,
                        static_cast<std::uint32_t>(static_cast<std::uint64_t>(side) * render_scale_percent_ / 100U));
    };
    return BenchmarkRenderSize{.width = scale(base.width), .height = scale(base.height)};
}

auto BenchmarkDriver::environment(Application const &application, VulkanContext const &context,
                                  BenchmarkRenderSize render_extent) const -> BenchmarkEnvironment {
    BenchmarkEnvironment environment;

    VkPhysicalDeviceDriverProperties driver_properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           .pNext = &driver_properties};
    vkGetPhysicalDeviceProperties2(context.physical_device, &properties);

    auto const &device = properties.properties;
    environment.device_name = device.deviceName;
    environment.device_type = device_type_name(device.deviceType);
    environment.vendor_id = device.vendorID;
    environment.device_id = device.deviceID;
    environment.driver_version = std::format("{} {}", driver_properties.driverName, driver_properties.driverInfo);
    environment.api_version =
            std::format("{}.{}.{}", VK_API_VERSION_MAJOR(device.apiVersion), VK_API_VERSION_MINOR(device.apiVersion),
                        VK_API_VERSION_PATCH(device.apiVersion));

    environment.render_width = render_extent.width;
    environment.render_height = render_extent.height;
    environment.swapchain_width = context.swapchain.extent().width;
    environment.swapchain_height = context.swapchain.extent().height;
    environment.present_mode = present_mode_name(context.swapchain.present_mode());
    environment.requested_present_mode = context.present_mode ? present_mode_name(*context.present_mode) : "";
    environment.swapchain_images = context.swapchain.image_count();
    environment.frames_in_flight = frames_in_flight;

    auto const &renderer = *application.renderer;
    environment.cluster_grid = renderer.cluster_grid();
    environment.occlusion_culling = renderer.occlusion_culling() && renderer.occlusion_culling_supported();
    environment.meshlet_occlusion =
            environment.occlusion_culling && renderer.meshlet_culling() && renderer.meshlet_occlusion_culling();

    probe_host_environment(environment);
    add_environment_warnings(environment);
    return environment;
}

auto BenchmarkDriver::end_frame(Application &application, VulkanContext const &context, CpuFrameTimes const &cpu,
                                BenchmarkRenderSize render_extent) -> Status {
    if (!run_) {
        return finished() ? Status::finished : Status::running;
    }

    auto const &renderer = *application.renderer;

    auto const recorded = renderer.recorded_frame_count();
    auto const frame_serial = recorded != last_frame_serial_ ? recorded : 0U;
    last_frame_serial_ = recorded;

    auto const events_now = perf_events::snapshot();
    auto const memory = MemoryTracker::stats();

    auto const *terrain = application.active_terrain();
    auto const &frame_stats = renderer.last_frame_stats();
    auto const &cluster_stats = renderer.last_cluster_stats();

    run_->on_frame_drawn(BenchmarkFrameInput{
            .gpu = &renderer.last_frame_timings(),
            .frame_serial = frame_serial,
            .cpu = cpu,
            .streaming_idle = application.renderer->texture_streamer().pending_count() == 0 &&
                              (terrain == nullptr || terrain->streaming_idle()),
            .counters =
                    BenchmarkCounters{
                            .occlusion_valid = frame_stats.occlusion_stats_valid,
                            .frustum_visible_instances = frame_stats.frustum_visible_instance_count,
                            .early_instances = frame_stats.early_instance_count,
                            .occlusion_candidates = frame_stats.occlusion_candidate_count,
                            .late_instances = frame_stats.late_instance_count,
                            .occluded_instances = frame_stats.occluded_instance_count,
                            .meshlet_valid = frame_stats.meshlet_occlusion_stats_valid,
                            .deferred_meshlets = frame_stats.deferred_meshlet_count,
                            .occluded_meshlets = frame_stats.occluded_meshlet_count,
                            .cluster_valid = cluster_stats.valid,
                            .occupied_clusters = cluster_stats.occupied_clusters,
                            .overflowing_clusters = cluster_stats.overflowing_clusters,
                            .maximum_lights = cluster_stats.maximum_lights,
                            .stored_lights = cluster_stats.stored_lights,
                    },
            .workload =
                    BenchmarkWorkload{
                            .submitted_triangles = frame_stats.submitted_triangle_count,
                            .submitted_instances = frame_stats.submitted_instance_count,
                            .indirect_commands = frame_stats.indirect_command_count,
                            .model_submissions = frame_stats.model_submission_count,
                            .mesh_submissions = frame_stats.mesh_submission_count,
                            .point_lights = frame_stats.point_light_count,
                            .spot_lights = frame_stats.spot_light_count,
                    },
            .events = events_now.since(last_events_),
            .allocations = memory.total_allocations - last_allocations_,
            .allocated_bytes = memory.total_allocated_bytes - last_allocated_bytes_,
    });

    last_events_ = events_now;
    last_allocations_ = memory.total_allocations;
    last_allocated_bytes_ = memory.total_allocated_bytes;

    if (!run_->finished()) {
        return Status::running;
    }

    return finish_case(application, context, render_extent);
}

auto BenchmarkDriver::finish_case(Application &application, VulkanContext const &context,
                                  BenchmarkRenderSize render_extent) -> Status {
    auto const &current = cases_[case_index_];
    auto const environment_now = environment(application, context, render_extent);

    auto result = run_->result(current.id, environment_now);
    result.thermals_start = thermals_start_;
    result.thermals_end = sample_thermals();
    log_case_summary(result);

    auto status = Status::running;

    if (options_.mode == BenchmarkMode::single) {
        if (auto const written = run_->write(environment_now); written) {
            info("Benchmark written to {} (per-frame samples alongside, .frames.csv)", options_.output_path.string());
            status = Status::finished;
        } else {
            error("Could not write benchmark results: {}", written.error());
            status = Status::failed;
        }
    } else {
        auto const csv_path = options_.output_path / "frames" / (current.id.file_stem() + ".csv");
        if (auto const written = write_text_file(csv_path, run_->to_csv()); !written) {
            error("Could not write {}: {}", csv_path.string(), written.error());
        }
    }

    results_.push_back(std::move(result));
    run_.reset();
    ++case_index_;

    if (options_.mode == BenchmarkMode::single || !finished()) {
        return status;
    }

    auto const json_path = options_.output_path / "suite.json";
    auto const report_path = options_.output_path / "report.md";

    auto const json_written = write_text_file(json_path, suite_to_json(options_, environment_now, results_));
    auto const report_written =
            write_text_file(report_path, suite_report_markdown(options_, environment_now, results_));

    if (!json_written || !report_written) {
        error("Could not write the suite results: {}", !json_written ? json_written.error() : report_written.error());
        return Status::failed;
    }

    info("Benchmark suite written to {} (report.md, suite.json, frames/)", options_.output_path.string());
    for (auto const &warning: environment_now.warnings) {
        warn("Benchmark: {}", warning);
    }

    return Status::finished;
}
