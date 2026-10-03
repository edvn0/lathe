#include "app/benchmark.hxx"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <set>
#include <span>
#include <utility>

#include "core/json.hxx"
#include "rendering/renderer.hxx"

namespace {

    [[nodiscard]] auto parse_count(std::string_view flag,
                                   std::string_view value) -> std::expected<std::uint32_t, std::string> {
        std::uint32_t result = 0;
        auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);

        if (error != std::errc{} || end != value.data() + value.size()) {
            return std::unexpected(std::format("{}: '{}' is not a non-negative integer", flag, value));
        }

        return result;
    }

    [[nodiscard]] auto parse_positive_float(std::string_view flag,
                                            std::string_view value) -> std::expected<float, std::string> {
        auto result = 0.0F;
        auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);

        if (error != std::errc{} || end != value.data() + value.size() || !(result > 0.0F) || !std::isfinite(result)) {
            return std::unexpected(std::format("{}: '{}' is not a positive number", flag, value));
        }

        return result;
    }

    [[nodiscard]] auto split(std::string_view text, char separator) -> std::vector<std::string_view> {
        std::vector<std::string_view> parts;
        while (!text.empty()) {
            auto const at = text.find(separator);
            parts.push_back(text.substr(0, at));
            if (at == std::string_view::npos) {
                break;
            }
            text = text.substr(at + 1);
        }
        return parts;
    }

    [[nodiscard]] auto sanitise_file_name(std::string_view text) -> std::string {
        std::string out;
        out.reserve(text.size());
        for (auto const character: text) {
            auto const keep = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
                              (character >= '0' && character <= '9') || character == '-';
            out.push_back(keep ? character : '_');
        }
        return out;
    }

    constexpr std::array<std::string_view, 11> counter_names{
            "frustum_visible_instances", "early_instances",   "occlusion_candidates", "late_instances",
            "occluded_instances",        "deferred_meshlets", "occluded_meshlets",    "occupied_clusters",
            "overflowing_clusters",      "maximum_lights",    "stored_lights",
    };

} // namespace

auto parse_render_size(std::string_view text) -> std::expected<BenchmarkRenderSize, std::string> {
    auto const x = text.find('x');
    if (x == std::string_view::npos) {
        return std::unexpected(std::format("'{}' is not WIDTHxHEIGHT", text));
    }

    auto const width = parse_count("width", text.substr(0, x));
    auto const height = parse_count("height", text.substr(x + 1));
    if (!width || !height) {
        return std::unexpected(std::format("'{}' is not WIDTHxHEIGHT", text));
    }
    if (*width == 0 || *height == 0 || *width > 16384 || *height > 16384) {
        return std::unexpected(std::format("'{}': each side must be between 1 and 16384", text));
    }

    return BenchmarkRenderSize{.width = *width, .height = *height};
}

auto parse_vsync_option(std::span<char const *const> args) -> std::expected<std::optional<bool>, std::string> {
    std::optional<bool> vsync;

    for (auto const *raw: args) {
        std::string_view const arg = raw;
        if (constexpr std::string_view prefix = "--vsync="; arg.starts_with(prefix)) {
            auto const value = arg.substr(prefix.size());
            if (value == "on") {
                vsync = true;
            } else if (value == "off") {
                vsync = false;
            } else {
                return std::unexpected(std::format("--vsync: '{}' (expected on or off)", value));
            }
        }
    }

    return vsync;
}

auto parse_present_mode_option(std::span<char const *const> args)
        -> std::expected<std::optional<PresentModeChoice>, std::string> {
    std::optional<PresentModeChoice> mode;

    for (auto const *raw: args) {
        std::string_view const arg = raw;
        if (constexpr std::string_view prefix = "--present-mode="; arg.starts_with(prefix)) {
            auto const value = arg.substr(prefix.size());
            if (value == "immediate") {
                mode = PresentModeChoice::immediate;
            } else if (value == "mailbox") {
                mode = PresentModeChoice::mailbox;
            } else if (value == "fifo") {
                mode = PresentModeChoice::fifo;
            } else if (value == "fifo_relaxed") {
                mode = PresentModeChoice::fifo_relaxed;
            } else {
                return std::unexpected(
                        std::format("--present-mode: '{}' (expected immediate, mailbox, fifo or fifo_relaxed)", value));
            }
        }
    }

    return mode;
}

auto parse_swapchain_images_option(std::span<char const *const> args)
        -> std::expected<std::optional<std::uint32_t>, std::string> {
    std::optional<std::uint32_t> count;

    for (auto const *raw: args) {
        std::string_view const arg = raw;
        if (constexpr std::string_view prefix = "--swapchain-images="; arg.starts_with(prefix)) {
            auto const value = parse_count("--swapchain-images", arg.substr(prefix.size()));
            if (!value) {
                return std::unexpected(value.error());
            }
            if (*value < 2 || *value > 8) {
                return std::unexpected(std::string{"--swapchain-images must be between 2 and 8"});
            }
            count = *value;
        }
    }

    return count;
}

auto parse_benchmark_options(std::span<char const *const> args)
        -> std::expected<std::optional<BenchmarkOptions>, std::string> {
    BenchmarkOptions options;
    bool single = false;
    bool suite = false;
    bool repeats_given = false;

    struct CountFlag {
        std::string_view prefix;
        std::uint32_t *value;
        bool allow_zero;
        bool *given;
    };

    std::array const count_flags{
            CountFlag{.prefix = "--benchmark-frames=", .value = &options.frame_count, .allow_zero = false},
            CountFlag{.prefix = "--benchmark-warmup=", .value = &options.warmup_frame_count, .allow_zero = true},
            CountFlag{
                    .prefix = "--benchmark-max-warmup=", .value = &options.max_warmup_frame_count, .allow_zero = true},
            CountFlag{.prefix = "--seed=", .value = &options.seed, .allow_zero = true},
            CountFlag{.prefix = "--benchmark-repeats=",
                      .value = &options.repeats,
                      .allow_zero = false,
                      .given = &repeats_given},
    };

    for (auto const *raw: args) {
        std::string_view const arg = raw;

        if (arg == "--benchmark-screenshots") {
            options.keyframe_screenshots = true;
            continue;
        }

        if (constexpr std::string_view prefix = "--benchmark="; arg.starts_with(prefix)) {
            auto const path = arg.substr(prefix.size());
            if (path.empty()) {
                return std::unexpected(std::string{"--benchmark= needs an output path"});
            }
            options.output_path = std::filesystem::path{path};
            single = true;
            continue;
        }

        if (constexpr std::string_view prefix = "--benchmark-suite="; arg.starts_with(prefix)) {
            auto const path = arg.substr(prefix.size());
            if (path.empty()) {
                return std::unexpected(std::string{"--benchmark-suite= needs an output directory"});
            }
            options.output_path = std::filesystem::path{path};
            suite = true;
            continue;
        }

        if (constexpr std::string_view prefix = "--benchmark-target-hz="; arg.starts_with(prefix)) {
            auto const hz = parse_positive_float("--benchmark-target-hz", arg.substr(prefix.size()));
            if (!hz) {
                return std::unexpected(hz.error());
            }
            options.target_hz = *hz;
            continue;
        }

        if (constexpr std::string_view prefix = "--benchmark-render-size="; arg.starts_with(prefix)) {
            auto const size = parse_render_size(arg.substr(prefix.size()));
            if (!size) {
                return std::unexpected(std::format("--benchmark-render-size: {}", size.error()));
            }
            options.render_size = *size;
            continue;
        }

        if (constexpr std::string_view prefix = "--benchmark-scenarios="; arg.starts_with(prefix)) {
            options.scenarios.clear();
            for (auto const name: split(arg.substr(prefix.size()), ',')) {
                if (name.empty()) {
                    return std::unexpected(std::string{"--benchmark-scenarios: empty scenario name"});
                }
                options.scenarios.emplace_back(name);
            }
            continue;
        }

        if (constexpr std::string_view prefix = "--benchmark-sweep="; arg.starts_with(prefix)) {
            auto const spec = arg.substr(prefix.size());
            auto const colon = spec.find(':');
            if (colon == std::string_view::npos || colon == 0 || colon + 1 == spec.size()) {
                return std::unexpected(std::format("--benchmark-sweep: '{}' is not scenario:load,load,...", spec));
            }

            BenchmarkSweep sweep{.scenario = std::string{spec.substr(0, colon)}};
            for (auto const value: split(spec.substr(colon + 1), ',')) {
                auto const load = parse_count("--benchmark-sweep", value);
                if (!load) {
                    return std::unexpected(load.error());
                }
                if (*load == 0) {
                    return std::unexpected(std::string{"--benchmark-sweep: loads must be at least 1"});
                }
                sweep.loads.push_back(*load);
            }

            std::erase_if(options.sweeps,
                          [&](BenchmarkSweep const &other) { return other.scenario == sweep.scenario; });
            options.sweeps.push_back(std::move(sweep));
            continue;
        }

        for (auto const &flag: count_flags) {
            if (!arg.starts_with(flag.prefix)) {
                continue;
            }

            auto const name = flag.prefix.substr(0, flag.prefix.size() - 1);
            auto const value = parse_count(name, arg.substr(flag.prefix.size()));

            if (!value) {
                return std::unexpected(value.error());
            }
            if (*value == 0 && !flag.allow_zero) {
                return std::unexpected(std::format("{} must be at least 1", name));
            }

            *flag.value = *value;
            if (flag.given != nullptr) {
                *flag.given = true;
            }
        }
    }

    if (single && suite) {
        return std::unexpected(std::string{"--benchmark= and --benchmark-suite= are exclusive"});
    }

    if (!single && !suite) {
        return std::nullopt;
    }

    options.mode = suite ? BenchmarkMode::suite : BenchmarkMode::single;
    options.max_warmup_frame_count = std::max(options.max_warmup_frame_count, options.warmup_frame_count);

    if (suite) {
        if (!repeats_given) {
            options.repeats = benchmark_default_suite_repeats;
        }
        if (!options.render_size) {
            options.render_size = benchmark_default_suite_render_size;
        }
    }

    return options;
}

// ---- BenchmarkCaseId

auto BenchmarkCaseId::key() const -> std::string {
    if (load_axis.empty()) {
        return scenario;
    }
    return std::format("{}@{}={}", scenario, load_axis, load);
}

auto BenchmarkCaseId::file_stem() const -> std::string {
    if (load_axis.empty()) {
        return std::format("{}_r{}", sanitise_file_name(scenario), repeat);
    }
    return std::format("{}_{}_{}_r{}", sanitise_file_name(scenario), sanitise_file_name(load_axis), load, repeat);
}

// ---- BenchmarkRun

BenchmarkRun::BenchmarkRun(BenchmarkOptions options, std::vector<CameraKeyframe> keyframes) :
    options_(std::move(options)), keyframes_(std::move(keyframes)) {
    samples_.reserve(options_.frame_count);
}

auto BenchmarkRun::camera() const noexcept -> CameraKeyframe {
    if (phase_ != Phase::measuring) {
        return sample_camera_path(keyframes_, 0.0F);
    }

    auto const t = static_cast<float>(measured_frames_) / static_cast<float>(options_.frame_count);
    return sample_camera_path(keyframes_, t);
}

auto BenchmarkRun::at_keyframe() const noexcept -> bool {
    if (phase_ != Phase::measuring || keyframes_.empty()) {
        return false;
    }

    // The keyframe segment frame `i` is in; a keyframe is crossed where it changes.
    auto const segment = [&](std::uint32_t frame) {
        return static_cast<std::uint64_t>(frame) * keyframes_.size() / options_.frame_count;
    };

    return measured_frames_ == 0 || segment(measured_frames_) != segment(measured_frames_ - 1);
}

auto BenchmarkRun::on_frame_drawn(FrameTimings const &timings, bool streaming_idle,
                                  BenchmarkCounters const &counters) -> void {
    on_frame_drawn(BenchmarkFrameInput{.gpu = &timings, .streaming_idle = streaming_idle, .counters = counters});
}

auto BenchmarkRun::pending_gpu_rows() const noexcept -> bool {
    return std::ranges::any_of(
            samples_, [](BenchmarkFrameSample const &sample) { return sample.serial != 0 && !sample.gpu_valid; });
}

auto BenchmarkRun::attach_gpu(FrameTimings const &timings, std::size_t row) -> void {
    auto &sample = samples_[row];
    sample.gpu_valid = true;
    sample.gpu_frame_ms = timings.full_frame_ms;

    for (auto const &pass: timings.passes) {
        auto found = std::ranges::find_if(stages_, [&](StageSamples const &stage) { return stage.id == pass.name_id; });
        if (found == stages_.end()) {
            // First seen: earlier frames count as 0 ms for it.
            stages_.push_back(StageSamples{
                    .id = pass.name_id, .name = pass.label, .samples_ms = std::vector<float>(samples_.size(), 0.0F)});
            found = std::prev(stages_.end());
        }
        found->samples_ms[row] = pass.milliseconds.value_or(0.0F);
    }
}

auto BenchmarkRun::on_frame_drawn(BenchmarkFrameInput const &input) -> void {
    if (phase_ == Phase::finished) {
        return;
    }

    if (phase_ == Phase::warmup) {
        ++warmup_frames_;

        auto const warmed_up = warmup_frames_ >= options_.warmup_frame_count && input.streaming_idle;
        auto const gave_up = warmup_frames_ >= options_.max_warmup_frame_count;

        if (warmed_up || gave_up) {
            phase_ = Phase::measuring;
            streaming_settled_ = input.streaming_idle;
        }

        // Timings of warmup frames have no row to land in; remember them so they are not taken for a later frame's.
        if (input.gpu != nullptr && input.gpu->valid) {
            last_gpu_serial_ = std::max(last_gpu_serial_, input.gpu->frame_serial);
        }
        return;
    }

    if (phase_ == Phase::measuring) {
        auto const t = static_cast<float>(measured_frames_) / static_cast<float>(options_.frame_count);

        samples_.push_back(BenchmarkFrameSample{
                .index = measured_frames_,
                .serial = input.frame_serial,
                .path_t = t,
                .cpu = input.cpu,
                .workload = input.workload,
                .events = input.events,
                .allocations = input.allocations,
                .allocated_bytes = input.allocated_bytes,
        });
        for (auto &stage: stages_) {
            stage.samples_ms.push_back(0.0F);
        }

        auto const record = [this](std::size_t first, bool valid, std::initializer_list<std::uint32_t> values) {
            if (!valid) {
                return;
            }

            auto index = first;
            for (auto const value: values) {
                counter_sums_[index] += value;
                ++counter_samples_[index];
                counter_final_[index] = value;
                ++index;
            }
        };
        auto const &counters = input.counters;
        record(0, counters.occlusion_valid,
               {counters.frustum_visible_instances, counters.early_instances, counters.occlusion_candidates,
                counters.late_instances, counters.occluded_instances});
        record(5, counters.meshlet_valid, {counters.deferred_meshlets, counters.occluded_meshlets});
        record(7, counters.cluster_valid,
               {counters.occupied_clusters, counters.overflowing_clusters, counters.maximum_lights,
                counters.stored_lights});

        ++measured_frames_;
    } else {
        ++drain_frames_;
    }

    if (input.gpu != nullptr && input.gpu->valid) {
        auto const &timings = *input.gpu;

        if (timings.frame_serial == 0) {
            // Untagged: belongs to the frame just drawn.
            if (phase_ == Phase::measuring && !samples_.back().gpu_valid) {
                attach_gpu(timings, samples_.size() - 1);
            }
        } else if (timings.frame_serial > last_gpu_serial_) {
            last_gpu_serial_ = timings.frame_serial;

            for (auto row = samples_.size(); row-- > 0;) {
                if (samples_[row].serial == timings.frame_serial) {
                    attach_gpu(timings, row);
                    break;
                }
                if (samples_[row].serial != 0 && samples_[row].serial < timings.frame_serial) {
                    break;
                }
            }
        }
    }

    if (phase_ == Phase::measuring && measured_frames_ >= options_.frame_count) {
        phase_ = Phase::draining;
    }

    if (phase_ == Phase::draining && (!pending_gpu_rows() || drain_frames_ >= options_.drain_frame_limit)) {
        phase_ = Phase::finished;
    }
}

auto BenchmarkRun::analyse() const -> FrameAnalysis {
    return analyse_frames(samples_, AnalysisOptions{.target_hz = options_.target_hz});
}

auto BenchmarkRun::stages() const -> std::vector<BenchmarkStage> {
    std::vector<BenchmarkStage> result;
    result.reserve(stages_.size() + 1);

    std::vector<float> values;
    values.reserve(samples_.size());

    for (auto const &sample: samples_) {
        if (sample.gpu_valid) {
            values.push_back(sample.gpu_frame_ms);
        }
    }
    result.push_back(BenchmarkStage{.id = "full_frame", .name = "Full Frame", .summary = summarise_timings(values)});

    for (auto const &stage: stages_) {
        values.clear();
        for (std::size_t row = 0; row < samples_.size(); ++row) {
            if (samples_[row].gpu_valid) {
                values.push_back(stage.samples_ms[row]);
            }
        }
        result.push_back(BenchmarkStage{.id = stage.id, .name = stage.name, .summary = summarise_timings(values)});
    }

    return result;
}

auto BenchmarkRun::result(BenchmarkCaseId id, BenchmarkEnvironment const &environment) const -> BenchmarkCaseResult {
    auto const frames_with_timings = std::ranges::count_if(samples_, &BenchmarkFrameSample::gpu_valid);

    return BenchmarkCaseResult{
            .id = std::move(id),
            .keyframes = static_cast<std::uint32_t>(keyframes_.size()),
            .warmup_frames = warmup_frames_,
            .streaming_settled = streaming_settled_,
            .frames_with_timings = static_cast<std::uint32_t>(frames_with_timings),
            .render_width = environment.render_width,
            .render_height = environment.render_height,
            .analysis = analyse(),
            .stages = stages(),
    };
}

namespace {

    auto write_summary(JsonWriter &writer, std::string_view key, TimingSummary const &summary) -> void {
        writer.begin_object(key, true)
                .value("count", summary.count)
                .value("mean_ms", summary.mean_ms)
                .value("stddev_ms", summary.stddev_ms)
                .value("min_ms", summary.min_ms)
                .value("median_ms", summary.median_ms)
                .value("p90_ms", summary.p90_ms)
                .value("p95_ms", summary.p95_ms)
                .value("p99_ms", summary.p99_ms)
                .value("p999_ms", summary.p999_ms)
                .value("max_ms", summary.max_ms)
                .end_object();
    }

    auto write_stages(JsonWriter &writer, std::span<BenchmarkStage const> stages) -> void {
        writer.begin_array("stages");
        for (auto const &stage: stages) {
            // The first keys match schema 1, which tools/perf/compare_benchmarks.py reads.
            writer.begin_object({}, true)
                    .value("id", stage.id)
                    .value("name", stage.name)
                    .value("mean_ms", stage.summary.mean_ms)
                    .value("median_ms", stage.summary.median_ms)
                    .value("p95_ms", stage.summary.p95_ms)
                    .value("min_ms", stage.summary.min_ms)
                    .value("max_ms", stage.summary.max_ms)
                    .value("p99_ms", stage.summary.p99_ms)
                    .value("stddev_ms", stage.summary.stddev_ms)
                    .end_object();
        }
        writer.end_array();
    }

    auto write_optional(JsonWriter &writer, std::string_view key, std::optional<float> value) -> void {
        if (value) {
            writer.value(key, *value, 1);
        } else {
            writer.null(key);
        }
    }

    auto write_thermals(JsonWriter &writer, std::string_view key, ThermalSample const &sample) -> void {
        writer.begin_object(key, true);
        write_optional(writer, "gpu_temperature_c", sample.gpu_temperature_c);
        write_optional(writer, "gpu_power_w", sample.gpu_power_w);
        write_optional(writer, "gpu_clock_mhz", sample.gpu_clock_mhz);
        write_optional(writer, "cpu_temperature_c", sample.cpu_temperature_c);
        write_optional(writer, "cpu_clock_mhz", sample.cpu_clock_mhz);
        writer.end_object();
    }

    [[nodiscard]] auto event_names(PerfEventCounts const &counts) -> std::string {
        std::string names;
        for (std::size_t event = 0; event < perf_event_count; ++event) {
            if (counts.values[event] != 0) {
                if (!names.empty()) {
                    names += ", ";
                }
                names += perf_event_name(static_cast<PerfEvent>(event));
            }
        }
        return names;
    }

} // namespace

auto write_environment_json(JsonWriter &writer, std::string_view key, BenchmarkEnvironment const &environment) -> void {
    writer.begin_object(key)
            .value("device", environment.device_name)
            .value("device_type", environment.device_type)
            .value("vendor_id", environment.vendor_id)
            .value("device_id", environment.device_id)
            .value("driver_version", environment.driver_version)
            .value("api_version", environment.api_version);
    writer.begin_array("render_extent", true)
            .value({}, environment.render_width)
            .value({}, environment.render_height)
            .end_array();
    writer.begin_array("swapchain_extent", true)
            .value({}, environment.swapchain_width)
            .value({}, environment.swapchain_height)
            .end_array();
    writer.value("present_mode", environment.present_mode)
            .value("requested_present_mode", environment.requested_present_mode)
            .value("swapchain_images", environment.swapchain_images)
            .value("frames_in_flight", environment.frames_in_flight)
            .value("occlusion_culling", environment.occlusion_culling)
            .value("meshlet_occlusion", environment.meshlet_occlusion);
    writer.begin_array("cluster_grid", true)
            .value({}, environment.cluster_grid.tiles_x)
            .value({}, environment.cluster_grid.tiles_y)
            .value({}, environment.cluster_grid.depth_slices)
            .value({}, environment.cluster_grid.light_capacity)
            .end_array();
    writer.value("build_type", environment.build_type)
            .value("source_revision", environment.source_revision)
            .value("memory_tracking", environment.memory_tracking)
            .value("cpu", environment.cpu_name)
            .value("cpu_threads", environment.cpu_threads)
            .value("cpu_governor", environment.cpu_governor)
            .value("os", environment.os)
            .value("timestamp_utc", environment.timestamp_utc);
    writer.begin_array("warnings");
    for (auto const &warning: environment.warnings) {
        writer.value({}, warning);
    }
    writer.end_array();
    writer.end_object();
}

auto write_analysis_json(JsonWriter &writer, std::string_view key, FrameAnalysis const &analysis) -> void {
    writer.begin_object(key);
    writer.value("frames", analysis.frames);
    write_summary(writer, "displayed", analysis.displayed);
    write_summary(writer, "cpu_frame", analysis.cpu_frame);
    write_summary(writer, "cpu_busy", analysis.cpu_busy);
    write_summary(writer, "cpu_wait", analysis.cpu_wait);
    write_summary(writer, "gpu_frame", analysis.gpu_frame);

    writer.begin_object("cpu_phases");
    for (std::size_t phase = 0; phase < cpu_phase_count; ++phase) {
        write_summary(writer, cpu_phase_name(static_cast<CpuPhase>(phase)), analysis.cpu_phases[phase]);
    }
    writer.end_object();

    auto const &budget = analysis.budget;
    auto const fraction = [](std::uint32_t part, std::uint32_t whole) {
        return whole == 0 ? 0.0 : static_cast<double>(part) / static_cast<double>(whole);
    };
    writer.begin_object("budget")
            .value("target_hz", budget.target_hz, 2)
            .value("budget_ms", budget.budget_ms)
            .value("frames", budget.frames)
            .value("displayed_over_budget", budget.displayed_over_budget)
            .value("displayed_over_budget_fraction", fraction(budget.displayed_over_budget, budget.frames))
            .value("cpu_busy_over_budget", budget.cpu_busy_over_budget)
            .value("gpu_over_budget", budget.gpu_over_budget)
            .value("frames_with_gpu", budget.frames_with_gpu)
            .end_object();

    writer.value("average_fps", analysis.average_fps, 1)
            .value("one_percent_low_fps", analysis.one_percent_low_fps, 1)
            .value("hitch_threshold_ms", analysis.hitch_threshold_ms)
            .value("hitch_count", static_cast<std::uint64_t>(analysis.hitches.size()));

    // The worst hitches, by displayed time; all of them are in the CSV.
    constexpr std::size_t listed_hitches = 32;
    auto worst = analysis.hitches;
    std::ranges::sort(worst, std::greater{}, &HitchFrame::displayed_ms);
    if (worst.size() > listed_hitches) {
        worst.resize(listed_hitches);
    }

    writer.begin_array("hitches");
    for (auto const &hitch: worst) {
        writer.begin_object({}, true)
                .value("frame", hitch.index)
                .value("displayed_ms", hitch.displayed_ms)
                .value("cpu_busy_ms", hitch.cpu_busy_ms);
        if (hitch.gpu_frame_ms) {
            writer.value("gpu_frame_ms", *hitch.gpu_frame_ms);
        } else {
            writer.null("gpu_frame_ms");
        }
        writer.begin_array("events");
        for (std::size_t event = 0; event < perf_event_count; ++event) {
            if (hitch.events.values[event] != 0) {
                writer.value({}, perf_event_name(static_cast<PerfEvent>(event)));
            }
        }
        writer.end_array();
        writer.end_object();
    }
    writer.end_array();

    writer.begin_object("events");
    for (std::size_t event = 0; event < perf_event_count; ++event) {
        auto const &correlation = analysis.events[event];
        writer.begin_object(perf_event_name(static_cast<PerfEvent>(event)), true)
                .value("total", correlation.total)
                .value("frames_with", correlation.frames_with)
                .value("mean_displayed_with_ms", correlation.mean_displayed_with_ms)
                .value("mean_displayed_without_ms", correlation.mean_displayed_without_ms)
                .value("hitches_with", correlation.hitches_with)
                .end_object();
    }
    writer.end_object();

    writer.begin_object("bound", true)
            .value("frames_with_gpu", analysis.bound.frames_with_gpu)
            .value("gpu_bound", analysis.bound.gpu_bound)
            .value("cpu_bound", analysis.bound.cpu_bound)
            .value("presentation_bound", analysis.bound.presentation_bound)
            .value("gpu_bound_fraction", analysis.bound.gpu_bound_fraction(), 3)
            .value("cpu_bound_fraction", analysis.bound.cpu_bound_fraction(), 3)
            .value("presentation_bound_fraction", analysis.bound.presentation_bound_fraction(), 3)
            .end_object();

    auto const &workload = analysis.workload;
    writer.begin_object("workload", true)
            .value("submitted_triangles", workload.submitted_triangles, 1)
            .value("submitted_instances", workload.submitted_instances, 1)
            .value("indirect_commands", workload.indirect_commands, 1)
            .value("model_submissions", workload.model_submissions, 1)
            .value("mesh_submissions", workload.mesh_submissions, 1)
            .value("point_lights", workload.point_lights, 1)
            .value("spot_lights", workload.spot_lights, 1)
            .end_object();

    writer.begin_object("allocations", true)
            .value("mean_per_frame", analysis.mean_allocations, 2)
            .value("max_per_frame", analysis.max_allocations)
            .value("mean_bytes_per_frame", analysis.mean_allocated_bytes, 1)
            .end_object();

    writer.end_object();
}

auto write_case_json(JsonWriter &writer, std::string_view key, BenchmarkCaseResult const &result) -> void {
    writer.begin_object(key)
            .value("key", result.id.key())
            .value("scenario", result.id.scenario)
            .value("load_axis", result.id.load_axis)
            .value("load", result.id.load)
            .value("repeat", result.id.repeat)
            .value("keyframes", result.keyframes)
            .value("warmup_frames", result.warmup_frames)
            .value("streaming_settled", result.streaming_settled)
            .value("frames", result.analysis.frames)
            .value("frames_with_timings", result.frames_with_timings);
    writer.begin_array("render_extent", true)
            .value({}, result.render_width)
            .value({}, result.render_height)
            .end_array();
    write_stages(writer, result.stages);
    write_analysis_json(writer, "analysis", result.analysis);
    writer.begin_object("thermals");
    write_thermals(writer, "start", result.thermals_start);
    write_thermals(writer, "end", result.thermals_end);
    writer.end_object();
    writer.end_object();
}

auto BenchmarkRun::to_json(BenchmarkEnvironment const &environment) const -> std::string {
    JsonWriter writer;
    auto const all_stages = stages();
    auto const analysis = analyse();

    writer.begin_object().value("schema", 2).value("kind", "single").value("device", environment.device_name);
    writer.begin_array("render_extent", true)
            .value({}, environment.render_width)
            .value({}, environment.render_height)
            .end_array();
    writer.begin_array("cluster_grid", true)
            .value({}, environment.cluster_grid.tiles_x)
            .value({}, environment.cluster_grid.tiles_y)
            .value({}, environment.cluster_grid.depth_slices)
            .value({}, environment.cluster_grid.light_capacity)
            .end_array();
    writer.value("occlusion_culling", environment.occlusion_culling)
            .value("meshlet_occlusion", environment.meshlet_occlusion)
            .value("seed", options_.seed)
            .value("keyframes", static_cast<std::uint64_t>(keyframes_.size()))
            .value("frames", measured_frames_)
            .value("frames_with_timings", static_cast<std::uint64_t>(all_stages.front().summary.count))
            .value("warmup_frames", warmup_frames_)
            .value("streaming_settled", streaming_settled_);

    write_stages(writer, all_stages);

    // Per-frame means over the frames the counter was read back for, and the last such frame's value. Counters whose
    // group was never valid (occlusion off, no lights) are null.
    writer.begin_object("counters");
    for (std::size_t i = 0; i < counter_count; ++i) {
        writer.begin_object(counter_names[i], true);
        if (counter_samples_[i] == 0) {
            writer.null("mean").null("final");
        } else {
            writer.value("mean", counter_sums_[i] / counter_samples_[i]).value("final", counter_final_[i]);
        }
        writer.end_object();
    }
    writer.end_object();

    write_environment_json(writer, "environment", environment);
    write_analysis_json(writer, "analysis", analysis);

    // GPU time of every measured frame that has one, in path order.
    std::vector<float> full_frame;
    full_frame.reserve(samples_.size());
    for (auto const &sample: samples_) {
        if (sample.gpu_valid) {
            full_frame.push_back(sample.gpu_frame_ms);
        }
    }
    writer.numbers("full_frame_ms", full_frame);

    writer.end_object();
    return writer.str();
}

auto BenchmarkRun::to_csv() const -> std::string {
    std::string csv;
    csv.reserve(samples_.size() * 256);

    csv += "frame,serial,path_t,displayed_ms,present_interval_ms,cpu_frame_ms,cpu_busy_ms,cpu_wait_ms";
    for (std::size_t phase = 0; phase < cpu_phase_count; ++phase) {
        csv += std::format(",cpu_{}_ms", cpu_phase_name(static_cast<CpuPhase>(phase)));
    }
    csv += ",gpu_frame_ms";
    for (auto const &stage: stages_) {
        csv += std::format(",gpu_{}_ms", sanitise_file_name(stage.id));
    }
    csv += ",submitted_triangles,submitted_instances,indirect_commands,model_submissions,mesh_submissions,"
           "point_lights,spot_lights";
    for (std::size_t event = 0; event < perf_event_count; ++event) {
        csv += std::format(",event_{}", perf_event_name(static_cast<PerfEvent>(event)));
    }
    csv += ",allocations,allocated_bytes\n";

    for (std::size_t row = 0; row < samples_.size(); ++row) {
        auto const &sample = samples_[row];
        csv += std::format("{},{},{:.5f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}", sample.index, sample.serial,
                           sample.path_t, sample.displayed_ms(), sample.cpu.present_interval_ms, sample.cpu.frame_ms,
                           sample.cpu.busy_ms(), sample.cpu.wait_ms());
        for (auto const phase_ms: sample.cpu.phase_ms) {
            csv += std::format(",{:.4f}", phase_ms);
        }

        if (sample.gpu_valid) {
            csv += std::format(",{:.4f}", sample.gpu_frame_ms);
            for (auto const &stage: stages_) {
                csv += std::format(",{:.4f}", stage.samples_ms[row]);
            }
        } else {
            csv += ',';
            csv.append(stages_.size(), ',');
        }

        auto const &workload = sample.workload;
        csv += std::format(",{},{},{},{},{},{},{}", workload.submitted_triangles, workload.submitted_instances,
                           workload.indirect_commands, workload.model_submissions, workload.mesh_submissions,
                           workload.point_lights, workload.spot_lights);
        for (auto const count: sample.events.values) {
            csv += std::format(",{}", count);
        }
        csv += std::format(",{},{}\n", sample.allocations, sample.allocated_bytes);
    }

    return csv;
}

auto write_text_file(std::filesystem::path const &path, std::string_view contents) -> std::expected<void, std::string> {
    if (auto const parent = path.parent_path(); !parent.empty()) {
        std::error_code error;
        std::filesystem::create_directories(parent, error);

        if (error) {
            return std::unexpected(std::format("could not create {}: {}", parent.string(), error.message()));
        }
    }

    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    if (!file) {
        return std::unexpected(std::format("could not open {} for writing", path.string()));
    }

    file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!file) {
        return std::unexpected(std::format("could not write {}", path.string()));
    }

    return {};
}

auto BenchmarkRun::write(BenchmarkEnvironment const &environment) const -> std::expected<void, std::string> {
    if (auto written = write_text_file(options_.output_path, to_json(environment)); !written) {
        return written;
    }

    auto csv_path = options_.output_path;
    csv_path.replace_extension(".frames.csv");
    return write_text_file(csv_path, to_csv());
}

// ---- Suite

auto plan_benchmark_cases(std::span<BenchmarkScenarioInfo const> scenarios,
                          BenchmarkOptions const &options) -> std::expected<std::vector<BenchmarkCase>, std::string> {
    auto const known = [&](std::string_view name) {
        return std::ranges::find(scenarios, name, &BenchmarkScenarioInfo::name) != scenarios.end();
    };

    auto const list_names = [&] {
        std::string names;
        for (auto const &scenario: scenarios) {
            names += names.empty() ? "" : ", ";
            names += scenario.name;
        }
        return names;
    };

    for (auto const &name: options.scenarios) {
        if (!known(name)) {
            return std::unexpected(std::format("unknown scenario '{}' (available: {})", name, list_names()));
        }
    }
    for (auto const &sweep: options.sweeps) {
        if (!known(sweep.scenario)) {
            return std::unexpected(std::format("--benchmark-sweep: unknown scenario '{}' (available: {})",
                                               sweep.scenario, list_names()));
        }
    }

    // One entry per (scenario, load), in scenario order then load order.
    struct Level {
        std::size_t scenario;
        std::uint32_t load;
    };
    std::vector<Level> levels;

    for (std::size_t index = 0; index < scenarios.size(); ++index) {
        auto const &scenario = scenarios[index];

        if (!options.scenarios.empty() &&
            std::ranges::find(options.scenarios, scenario.name) == options.scenarios.end()) {
            continue;
        }

        auto loads = scenario.default_loads;
        if (auto const sweep = std::ranges::find(options.sweeps, scenario.name, &BenchmarkSweep::scenario);
            sweep != options.sweeps.end()) {
            if (scenario.load_axis.empty()) {
                return std::unexpected(std::format("--benchmark-sweep: scenario '{}' has no load axis", scenario.name));
            }
            loads = sweep->loads;
        }
        if (scenario.load_axis.empty() || loads.empty()) {
            loads = {0};
        }

        for (auto const load: loads) {
            levels.push_back(Level{.scenario = index, .load = load});
        }
    }

    if (levels.empty()) {
        return std::unexpected(std::string{"no benchmark cases to run"});
    }

    std::vector<BenchmarkCase> cases;
    cases.reserve(levels.size() * options.repeats);

    for (std::uint32_t repeat = 0; repeat < options.repeats; ++repeat) {
        for (auto const &level: levels) {
            auto const &scenario = scenarios[level.scenario];
            cases.push_back(BenchmarkCase{
                    .scenario = level.scenario,
                    .id = BenchmarkCaseId{.scenario = scenario.name,
                                          .load_axis = scenario.load_axis,
                                          .load = level.load,
                                          .repeat = repeat},
            });
        }
    }

    return cases;
}

auto BenchmarkAggregate::metric(std::string_view name) const noexcept -> RepeatStats const * {
    for (auto const &[metric_name, stats]: metrics) {
        if (metric_name == name) {
            return &stats;
        }
    }
    return nullptr;
}

auto aggregate_cases(std::span<BenchmarkCaseResult const> results) -> std::vector<BenchmarkAggregate> {
    std::vector<BenchmarkAggregate> aggregates;

    // Group by case key, in order of first appearance.
    std::vector<std::vector<BenchmarkCaseResult const *>> groups;
    for (auto const &result: results) {
        auto const key = result.id.key();
        auto found = std::ranges::find(aggregates, key, &BenchmarkAggregate::key);
        if (found == aggregates.end()) {
            aggregates.push_back(BenchmarkAggregate{.key = key,
                                                    .scenario = result.id.scenario,
                                                    .load_axis = result.id.load_axis,
                                                    .load = result.id.load});
            groups.emplace_back();
            found = std::prev(aggregates.end());
        }
        groups[static_cast<std::size_t>(found - aggregates.begin())].push_back(&result);
    }

    for (std::size_t index = 0; index < aggregates.size(); ++index) {
        auto &aggregate = aggregates[index];
        auto const &group = groups[index];
        aggregate.repeats = static_cast<std::uint32_t>(group.size());

        auto const add = [&](std::string name, auto projection) {
            std::vector<double> values;
            values.reserve(group.size());
            for (auto const *result: group) {
                values.push_back(static_cast<double>(projection(*result)));
            }
            aggregate.metrics.emplace_back(std::move(name), summarise_repeats(values));
        };

        add("displayed_median_ms", [](BenchmarkCaseResult const &r) { return r.analysis.displayed.median_ms; });
        add("displayed_p99_ms", [](BenchmarkCaseResult const &r) { return r.analysis.displayed.p99_ms; });
        add("displayed_p999_ms", [](BenchmarkCaseResult const &r) { return r.analysis.displayed.p999_ms; });
        add("cpu_busy_median_ms", [](BenchmarkCaseResult const &r) { return r.analysis.cpu_busy.median_ms; });
        add("cpu_busy_p99_ms", [](BenchmarkCaseResult const &r) { return r.analysis.cpu_busy.p99_ms; });
        add("gpu_median_ms", [](BenchmarkCaseResult const &r) { return r.analysis.gpu_frame.median_ms; });
        add("gpu_p99_ms", [](BenchmarkCaseResult const &r) { return r.analysis.gpu_frame.p99_ms; });
        add("budget_miss_fraction", [](BenchmarkCaseResult const &r) {
            auto const &budget = r.analysis.budget;
            return budget.frames == 0
                           ? 0.0
                           : static_cast<double>(budget.displayed_over_budget) / static_cast<double>(budget.frames);
        });
        add("hitches", [](BenchmarkCaseResult const &r) { return static_cast<double>(r.analysis.hitches.size()); });
        add("gpu_bound_fraction", [](BenchmarkCaseResult const &r) { return r.analysis.bound.gpu_bound_fraction(); });
        add("cpu_bound_fraction", [](BenchmarkCaseResult const &r) { return r.analysis.bound.cpu_bound_fraction(); });
        add("presentation_bound_fraction",
            [](BenchmarkCaseResult const &r) { return r.analysis.bound.presentation_bound_fraction(); });

        // Every stage seen in any repeat, 0 in a repeat without it.
        std::vector<std::string> stage_ids;
        for (auto const *result: group) {
            for (auto const &stage: result->stages) {
                if (stage.id != "full_frame" && std::ranges::find(stage_ids, stage.id) == stage_ids.end()) {
                    stage_ids.push_back(stage.id);
                }
            }
        }
        for (auto const &stage_id: stage_ids) {
            add(std::format("stage_{}_median_ms", stage_id), [&stage_id](BenchmarkCaseResult const &r) {
                auto const stage = std::ranges::find(r.stages, stage_id, &BenchmarkStage::id);
                return stage == r.stages.end() ? 0.0F : stage->summary.median_ms;
            });
        }
    }

    return aggregates;
}

auto scaling_of(std::span<BenchmarkAggregate const> aggregates) -> std::vector<BenchmarkScaling> {
    std::vector<BenchmarkScaling> scaling;

    for (auto const &aggregate: aggregates) {
        if (aggregate.load_axis.empty()) {
            continue;
        }

        auto found = std::ranges::find(scaling, aggregate.scenario, &BenchmarkScaling::scenario);
        if (found == scaling.end()) {
            scaling.push_back(BenchmarkScaling{.scenario = aggregate.scenario, .load_axis = aggregate.load_axis});
            found = std::prev(scaling.end());
        }

        auto const median = [&](std::string_view name) {
            auto const *stats = aggregate.metric(name);
            return stats == nullptr ? 0.0 : stats->median;
        };

        found->loads.push_back(aggregate.load);
        found->gpu_median_ms.push_back(median("gpu_median_ms"));
        found->cpu_busy_median_ms.push_back(median("cpu_busy_median_ms"));
        found->displayed_p99_ms.push_back(median("displayed_p99_ms"));
    }

    std::erase_if(scaling, [](BenchmarkScaling const &entry) {
        return std::set<std::uint32_t>(entry.loads.begin(), entry.loads.end()).size() < 2;
    });

    for (auto &entry: scaling) {
        std::vector<double> loads{entry.loads.begin(), entry.loads.end()};
        entry.gpu = fit_scaling(loads, entry.gpu_median_ms);
        entry.cpu_busy = fit_scaling(loads, entry.cpu_busy_median_ms);
        entry.displayed_p99 = fit_scaling(loads, entry.displayed_p99_ms);
    }

    return scaling;
}

namespace {

    auto write_fit(JsonWriter &writer, std::string_view key, ScalingFit const &fit, double budget_ms) -> void {
        writer.begin_object(key, true)
                .value("intercept_ms", fit.intercept_ms)
                .value("slope_ms_per_unit", fit.slope_ms_per_unit, 8)
                .value("r_squared", fit.r_squared, 4);
        if (fit.exponent) {
            writer.value("exponent", *fit.exponent, 3);
        } else {
            writer.null("exponent");
        }
        if (auto const load = fit.load_at(budget_ms)) {
            writer.value("load_at_budget", *load, 0);
        } else {
            writer.null("load_at_budget");
        }
        writer.end_object();
    }

} // namespace

auto suite_to_json(BenchmarkOptions const &options, BenchmarkEnvironment const &environment,
                   std::span<BenchmarkCaseResult const> results) -> std::string {
    JsonWriter writer;
    auto const budget_ms = 1000.0 / static_cast<double>(options.target_hz);

    writer.begin_object().value("schema", 2).value("kind", "suite");

    writer.begin_object("options")
            .value("frames", options.frame_count)
            .value("warmup_frames", options.warmup_frame_count)
            .value("max_warmup_frames", options.max_warmup_frame_count)
            .value("seed", options.seed)
            .value("repeats", options.repeats)
            .value("target_hz", options.target_hz, 2)
            .value("budget_ms", budget_ms)
            .value("timestep_s", static_cast<double>(benchmark_timestep), 6)
            .end_object();

    write_environment_json(writer, "environment", environment);

    writer.begin_array("cases");
    for (auto const &result: results) {
        write_case_json(writer, {}, result);
    }
    writer.end_array();

    auto const aggregates = aggregate_cases(results);
    writer.begin_array("aggregates");
    for (auto const &aggregate: aggregates) {
        writer.begin_object()
                .value("key", aggregate.key)
                .value("scenario", aggregate.scenario)
                .value("load_axis", aggregate.load_axis)
                .value("load", aggregate.load)
                .value("repeats", aggregate.repeats);
        writer.begin_object("metrics");
        for (auto const &[name, stats]: aggregate.metrics) {
            writer.begin_object(name, true)
                    .value("median", stats.median)
                    .value("min", stats.min)
                    .value("max", stats.max)
                    .value("spread", stats.spread());
            writer.begin_array("values");
            for (auto const value: stats.values) {
                writer.value({}, value);
            }
            writer.end_array();
            writer.end_object();
        }
        writer.end_object();
        writer.end_object();
    }
    writer.end_array();

    writer.begin_array("scaling");
    for (auto const &entry: scaling_of(aggregates)) {
        writer.begin_object().value("scenario", entry.scenario).value("load_axis", entry.load_axis);
        writer.begin_array("loads", true);
        for (auto const load: entry.loads) {
            writer.value({}, load);
        }
        writer.end_array();

        auto const series = [&](std::string_view key, std::vector<double> const &values) {
            writer.begin_array(key, true);
            for (auto const value: values) {
                writer.value({}, value);
            }
            writer.end_array();
        };
        series("gpu_median_ms", entry.gpu_median_ms);
        series("cpu_busy_median_ms", entry.cpu_busy_median_ms);
        series("displayed_p99_ms", entry.displayed_p99_ms);

        writer.begin_object("fits");
        write_fit(writer, "gpu", entry.gpu, budget_ms);
        write_fit(writer, "cpu_busy", entry.cpu_busy, budget_ms);
        write_fit(writer, "displayed_p99", entry.displayed_p99, budget_ms);
        writer.end_object();
        writer.end_object();
    }
    writer.end_array();

    writer.end_object();
    return writer.str();
}

namespace {

    [[nodiscard]] auto format_stats(RepeatStats const *stats, int precision = 2) -> std::string {
        if (stats == nullptr || stats->values.empty()) {
            return "--";
        }
        if (stats->values.size() == 1) {
            return std::format("{:.{}f}", stats->median, precision);
        }
        return std::format("{:.{}f} ({:.{}f}-{:.{}f})", stats->median, precision, stats->min, precision, stats->max,
                           precision);
    }

    // The dominant limiter across a case's repeats, e.g. "GPU 100%" or "presentation 97%".
    [[nodiscard]] auto limiter_text(BenchmarkAggregate const &aggregate) -> std::string {
        constexpr std::array<std::pair<std::string_view, std::string_view>, 3> limiters{{
                {"GPU", "gpu_bound_fraction"},
                {"CPU", "cpu_bound_fraction"},
                {"presentation", "presentation_bound_fraction"},
        }};

        std::string_view best_name;
        auto best = -1.0;
        for (auto const &[name, metric]: limiters) {
            if (auto const *stats = aggregate.metric(metric); stats != nullptr && stats->median > best) {
                best = stats->median;
                best_name = name;
            }
        }
        return best < 0.0 ? std::string{"--"} : std::format("{} {:.0f}%", best_name, best * 100.0);
    }

    [[nodiscard]] auto describe_fit(ScalingFit const &fit, std::string_view axis, std::uint32_t largest_load,
                                    double budget_ms) -> std::string {
        if (fit.points < 2) {
            return "not enough points";
        }

        // Marginal cost per 1000 objects reads better than per object, per layer better than per 1000 layers.
        auto const unit = largest_load >= 5000 ? 1000U : largest_load >= 500 ? 100U : 1U;
        auto text = unit == 1 ? std::format("{:+.4f} ms per {}", fit.slope_ms_per_unit, axis)
                              : std::format("{:+.4f} ms per {} {}", fit.slope_ms_per_unit * unit, unit, axis);
        text += std::format(" (fit R² {:.3f}", fit.r_squared);
        if (fit.exponent) {
            text += std::format(", cost ~ load^{:.2f}", *fit.exponent);
        }
        text += ")";
        if (auto const load = fit.load_at(budget_ms)) {
            text += std::format("; reaches the {:.2f} ms budget at ~{:.0f} {}", budget_ms, *load, axis);
        }
        return text;
    }

} // namespace

auto suite_report_markdown(BenchmarkOptions const &options, BenchmarkEnvironment const &environment,
                           std::span<BenchmarkCaseResult const> results) -> std::string {
    auto const budget_ms = 1000.0 / static_cast<double>(options.target_hz);
    auto const aggregates = aggregate_cases(results);

    std::string out;
    out += "# Lathe benchmark suite\n\n";
    out += std::format("- **Device**: {} ({}), driver {}, Vulkan {}\n", environment.device_name,
                       environment.device_type, environment.driver_version, environment.api_version);
    out += std::format("- **CPU**: {} ({} threads), governor `{}`; {}\n", environment.cpu_name, environment.cpu_threads,
                       environment.cpu_governor.empty() ? "unknown" : environment.cpu_governor, environment.os);
    out += std::format("- **Build**: {} at `{}`, present mode `{}` with {} swapchain images, {} frames in flight\n",
                       environment.build_type,
                       environment.source_revision.empty() ? "unknown" : environment.source_revision,
                       environment.present_mode, environment.swapchain_images, environment.frames_in_flight);
    out += std::format("- **Method**: {} measured frames per run at a fixed {:.4f} s step, seed {}, {} repeat(s) per "
                       "case run interleaved; budget {:.2f} ms ({:.0f} Hz)\n",
                       options.frame_count, static_cast<double>(benchmark_timestep), options.seed, options.repeats,
                       budget_ms, static_cast<double>(options.target_hz));
    out += std::format("- **When**: {}\n", environment.timestamp_utc);

    for (auto const &warning: environment.warnings) {
        out += std::format("- :warning: {}\n", warning);
    }

    // Cases whose frames were mostly paced by the swapchain: their displayed times say little about the engine.
    std::vector<std::string> presentation_paced;
    for (auto const &aggregate: aggregates) {
        if (auto const *paced = aggregate.metric("presentation_bound_fraction");
            paced != nullptr && paced->median > 0.5) {
            presentation_paced.push_back(aggregate.key);
        }
    }
    if (!presentation_paced.empty()) {
        out += std::format("- :warning: {} of {} cases were mostly **presentation-bound**: the CPU sat in acquire or "
                           "present while the GPU had time to spare, so their displayed times are set by the swapchain "
                           "or compositor, not the engine. Read GPU p50 and CPU busy for those; to measure displayed "
                           "times, try `--present-mode=immediate`, `--swapchain-images=4`, or an X11 session.\n",
                           presentation_paced.size(), aggregates.size());
    }

    out += "\n## Cases\n\n";
    out += "Median across repeats, with the range in brackets. *Displayed* is present-to-present time, what a player "
           "sees. *CPU busy* excludes time blocked on the GPU or the swapchain. *Over budget* allows 1% for refresh "
           "jitter. *Limited by* is what held most frames back: the GPU, the CPU, or presentation (the swapchain "
           "made the CPU wait while the GPU had time to spare).\n\n";
    out += "| Case | Displayed p50 (ms) | Displayed p99 (ms) | p99.9 (ms) | GPU p50 (ms) | CPU busy p50 (ms) | Over "
           "budget | Hitches | Limited by |\n";
    out += "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";

    for (auto const &aggregate: aggregates) {
        auto const *miss = aggregate.metric("budget_miss_fraction");
        out += std::format(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n", aggregate.key,
                format_stats(aggregate.metric("displayed_median_ms")),
                format_stats(aggregate.metric("displayed_p99_ms")), format_stats(aggregate.metric("displayed_p999_ms")),
                format_stats(aggregate.metric("gpu_median_ms")), format_stats(aggregate.metric("cpu_busy_median_ms")),
                miss == nullptr ? "--" : std::format("{:.1f}%", miss->median * 100.0),
                format_stats(aggregate.metric("hitches"), 0), limiter_text(aggregate));
    }

    if (auto const scaling = scaling_of(aggregates); !scaling.empty()) {
        out += "\n## Scaling\n\n";
        out += "Least-squares fits of the per-load medians. The exponent comes from a log-log fit: about 1 is linear, "
               "below 1 means a fixed cost dominates, above 1 means each unit gets more expensive.\n";

        for (auto const &entry: scaling) {
            out += std::format("\n### {} (by {})\n\n", entry.scenario, entry.load_axis);
            out += std::format("| {} | GPU p50 (ms) | CPU busy p50 (ms) | Displayed p99 (ms) |\n", entry.load_axis);
            out += "|---:|---:|---:|---:|\n";
            for (std::size_t i = 0; i < entry.loads.size(); ++i) {
                out += std::format("| {} | {:.3f} | {:.3f} | {:.3f} |\n", entry.loads[i], entry.gpu_median_ms[i],
                                   entry.cpu_busy_median_ms[i], entry.displayed_p99_ms[i]);
            }
            auto const largest = std::ranges::max(entry.loads);
            out += std::format("\n- GPU: {}\n", describe_fit(entry.gpu, entry.load_axis, largest, budget_ms));
            out += std::format("- CPU busy: {}\n", describe_fit(entry.cpu_busy, entry.load_axis, largest, budget_ms));
        }
    }

    // Events that coincide with hitches, over every run.
    std::array<std::uint64_t, perf_event_count> hitches_with{};
    std::uint64_t total_hitches = 0;
    std::uint64_t unexplained = 0;
    for (auto const &result: results) {
        total_hitches += result.analysis.hitches.size();
        for (auto const &hitch: result.analysis.hitches) {
            unexplained += hitch.events.any() ? 0U : 1U;
        }
        for (std::size_t event = 0; event < perf_event_count; ++event) {
            hitches_with[event] += result.analysis.events[event].hitches_with;
        }
    }

    out += "\n## Hitches\n\n";
    if (total_hitches == 0) {
        out += "No displayed frame exceeded both twice its run's median and the budget.\n";
    } else {
        out += std::format("{} hitch(es) over all runs (a displayed frame over both twice its run's median and the "
                           "budget). Events in the hitch frame or the one before it:\n\n",
                           total_hitches);
        out += "| Event | Hitches with it |\n|---|---:|\n";
        for (std::size_t event = 0; event < perf_event_count; ++event) {
            if (hitches_with[event] != 0) {
                out += std::format("| `{}` | {} |\n", perf_event_name(static_cast<PerfEvent>(event)),
                                   hitches_with[event]);
            }
        }
        out += std::format("| (none recorded) | {} |\n", unexplained);

        out += "\nWorst per case:\n\n";
        for (auto const &result: results) {
            if (result.analysis.hitches.empty()) {
                continue;
            }
            auto const worst = std::ranges::max_element(result.analysis.hitches, {}, &HitchFrame::displayed_ms);
            auto const events = event_names(worst->events);
            out += std::format("- {} (repeat {}): frame {} took {:.2f} ms{}\n", result.id.key(), result.id.repeat,
                               worst->index, worst->displayed_ms,
                               events.empty() ? std::string{} : std::format(" with {}", events));
        }
    }

    auto const unsettled = std::ranges::count_if(results, [](auto const &r) { return !r.streaming_settled; });
    if (unsettled != 0) {
        out += std::format("\n:warning: {} run(s) started measuring before streaming settled "
                           "(--benchmark-max-warmup reached).\n",
                           unsettled);
    }

    out += "\nPer-frame data for every run is in `frames/`; `suite.json` has the full numbers.\n";
    return out;
}
