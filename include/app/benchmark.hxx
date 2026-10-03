#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "app/benchmark_analysis.hxx"
#include "app/frame_clock.hxx"
#include "core/perf_events.hxx"
#include "rendering/cluster_grid.hxx"
#include "scene/camera_path.hxx"

struct FrameTimings;
class JsonWriter;

// Repeatable performance measurement, built into the engine (docs/perf-benchmark.md).
//
// Two ways to run it:
//
//   --benchmark=<out.json>         one run of the game's scene along IGame::benchmark_camera_path(). Writes the JSON
//                                  (compatible with earlier versions plus an "analysis" section) and the per-frame
//                                  samples next to it as <out>.frames.csv.
//   --benchmark-suite=<dir>        every scenario (or --benchmark-scenarios=) at each of its load levels, each
//                                  --benchmark-repeats times, interleaved. Writes <dir>/suite.json, <dir>/report.md and
//                                  <dir>/frames/<case>.csv.
//
// Shared flags:
//   --benchmark-frames=<n>         measured frames per run, one lap of the camera path (600)
//   --benchmark-warmup=<n>         minimum frames at the first keyframe before measuring (60)
//   --benchmark-max-warmup=<n>     measure anyway after this many, even if streaming hasn't settled (1200)
//   --seed=<n>                     scene seed (1337), see core/random.hxx
//   --benchmark-target-hz=<hz>     refresh rate frames are judged against (144)
//   --benchmark-render-size=WxH    fixed render resolution (suite default 1920x1080; single run: the Viewport panel)
//   --benchmark-screenshots        screenshot the first frame at or past each keyframe (single run only)
//   --benchmark-scenarios=a,b      suite: which scenarios (default: all)
//   --benchmark-sweep=name:1,2,4   suite: load levels for a scenario, replacing its defaults (repeatable)
//   --benchmark-repeats=<n>        suite: runs of each case (3)
//   --vsync=on|off                 off by default while benchmarking, so presentation doesn't cap the frame rate
//
// --benchmark-compare=<base>,<head> compares two results instead of running (app/benchmark_compare.hxx).

enum class BenchmarkMode : std::uint8_t {
    single,
    suite,
};

struct BenchmarkRenderSize {
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    auto operator==(BenchmarkRenderSize const &) const -> bool = default;
};

// Load levels for one scenario, from --benchmark-sweep=.
struct BenchmarkSweep {
    std::string scenario;
    std::vector<std::uint32_t> loads;
};

struct BenchmarkOptions {
    BenchmarkMode mode = BenchmarkMode::single;

    // single: the JSON file; suite: the output directory.
    std::filesystem::path output_path;

    std::uint32_t frame_count = 600;
    std::uint32_t warmup_frame_count = 60;
    std::uint32_t max_warmup_frame_count = 1200;
    std::uint32_t seed = 1337;
    bool keyframe_screenshots = false;

    float target_hz = 144.0F;
    std::optional<BenchmarkRenderSize> render_size;

    std::vector<std::string> scenarios;
    std::vector<BenchmarkSweep> sweeps;
    std::uint32_t repeats = 1;

    // Frames drawn after the lap, with the camera parked, to collect the GPU timings of its last frames.
    std::uint32_t drain_frame_limit = 8;
};

inline constexpr BenchmarkRenderSize benchmark_default_suite_render_size{.width = 1920, .height = 1080};
inline constexpr std::uint32_t benchmark_default_suite_repeats = 3;

// nullopt without --benchmark= or --benchmark-suite= (the other flags are then ignored); an error for a malformed
// or zero value, or for both modes at once.
[[nodiscard]]
auto parse_benchmark_options(std::span<char const *const> args)
        -> std::expected<std::optional<BenchmarkOptions>, std::string>;

// --present-mode=immediate|mailbox|fifo|fifo_relaxed. Benchmarks prefer immediate when none is given: MAILBOX still
// lets some compositors pace acquisition to the refresh rate.
enum class PresentModeChoice : std::uint8_t {
    immediate,
    mailbox,
    fifo,
    fifo_relaxed,
};

[[nodiscard]]
auto parse_present_mode_option(std::span<char const *const> args)
        -> std::expected<std::optional<PresentModeChoice>, std::string>;

// --swapchain-images=<n> (2..8), nullopt when absent.
[[nodiscard]]
auto parse_swapchain_images_option(std::span<char const *const> args)
        -> std::expected<std::optional<std::uint32_t>, std::string>;

// --vsync=on|off, nullopt when absent.
[[nodiscard]]
auto parse_vsync_option(std::span<char const *const> args) -> std::expected<std::optional<bool>, std::string>;

// "1920x1080".
[[nodiscard]]
auto parse_render_size(std::string_view text) -> std::expected<BenchmarkRenderSize, std::string>;

// Simulated seconds per frame, so frame N shows the same scene in every run.
inline constexpr float benchmark_timestep = 1.0F / 60.0F;

// Culling and clustering counters for one frame, copied from FrameStats / ClusterStats (which lag the frame by the
// frames in flight). The `*_valid` flags say whether a group was read back; invalid groups are left out of the means.
struct BenchmarkCounters {
    bool occlusion_valid = false;
    std::uint32_t frustum_visible_instances = 0;
    std::uint32_t early_instances = 0;
    std::uint32_t occlusion_candidates = 0;
    std::uint32_t late_instances = 0;
    std::uint32_t occluded_instances = 0;

    bool meshlet_valid = false;
    std::uint32_t deferred_meshlets = 0;
    std::uint32_t occluded_meshlets = 0;

    bool cluster_valid = false;
    std::uint32_t occupied_clusters = 0;
    std::uint32_t overflowing_clusters = 0;
    std::uint32_t maximum_lights = 0;
    std::uint32_t stored_lights = 0;
};

// Hardware, driver and machine state a result was produced on (app/benchmark_environment.hxx fills most of it).
struct ThermalSample {
    std::optional<float> gpu_temperature_c;
    std::optional<float> gpu_power_w;
    std::optional<float> gpu_clock_mhz;
    std::optional<float> cpu_temperature_c;
    std::optional<float> cpu_clock_mhz;
};

struct BenchmarkEnvironment {
    std::string device_name;
    std::string device_type;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::string driver_version;
    std::string api_version;

    std::uint32_t render_width = 0;
    std::uint32_t render_height = 0;
    std::uint32_t swapchain_width = 0;
    std::uint32_t swapchain_height = 0;
    std::string present_mode;
    // --present-mode= or the benchmark's default; empty when the vsync setting decided.
    std::string requested_present_mode;
    std::uint32_t swapchain_images = 0;
    std::uint32_t frames_in_flight = 0;
    ClusterGridSettings cluster_grid{};

    // Renderer::occlusion_culling() for the run, so on/off results aren't mistaken for one another.
    bool occlusion_culling = false;

    // Renderer::meshlet_occlusion_culling() for the run, and only true when it was active (occlusion culling and
    // meshlet culling on too).
    bool meshlet_occlusion = false;

    std::string build_type;
    std::string source_revision;
    bool memory_tracking = false;

    std::string cpu_name;
    std::uint32_t cpu_threads = 0;
    std::string cpu_governor;
    std::string os;
    std::string timestamp_utc;

    // Things about the machine that make numbers less trustworthy (powersave governor, vsync on, software rasterizer).
    std::vector<std::string> warnings;
};

// A frame graph pass's GPU time over a run.
struct BenchmarkStage {
    std::string id;
    std::string name;
    TimingSummary summary;
};

struct BenchmarkCaseId {
    std::string scenario;
    std::string load_axis; // empty when the scenario has none
    std::uint32_t load = 0;
    std::uint32_t repeat = 0;

    // "lights@point_lights=512", "game"; the repeat is not part of it.
    [[nodiscard]] auto key() const -> std::string;
    // key() made safe for a file name, plus the repeat: "lights_point_lights_512_r0".
    [[nodiscard]] auto file_stem() const -> std::string;
};

// Everything one run produced.
struct BenchmarkCaseResult {
    BenchmarkCaseId id;
    std::uint32_t keyframes = 0;
    std::uint32_t warmup_frames = 0;
    bool streaming_settled = true;
    std::uint32_t frames_with_timings = 0;
    std::uint32_t render_width = 0;
    std::uint32_t render_height = 0;
    FrameAnalysis analysis;
    std::vector<BenchmarkStage> stages; // full_frame first
    ThermalSample thermals_start{};
    ThermalSample thermals_end{};
};

// What main.cxx hands over after each drawn frame.
struct BenchmarkFrameInput {
    // The renderer's latest retired GPU timings; their frame_serial says which frame they belong to.
    FrameTimings const *gpu = nullptr;

    // Renderer::recorded_frame_count() after this frame, or 0 if this frame recorded nothing.
    std::uint64_t frame_serial = 0;

    CpuFrameTimes cpu{};
    bool streaming_idle = true;
    BenchmarkCounters counters{};
    BenchmarkWorkload workload{};
    PerfEventCounts events{};
    std::uint64_t allocations = 0;
    std::uint64_t allocated_bytes = 0;
};

// One run: warm up parked at the first keyframe, fly one lap measuring every frame, then drain the GPU timings of
// the last frames.
class BenchmarkRun {
public:
    BenchmarkRun(BenchmarkOptions options, std::vector<CameraKeyframe> keyframes);

    // The camera for the frame about to be drawn: the first keyframe while warming up, then one lap.
    [[nodiscard]]
    auto camera() const noexcept -> CameraKeyframe;

    // Call after each drawn frame. Warmup ends only once `streaming_idle`.
    auto on_frame_drawn(BenchmarkFrameInput const &input) -> void;

    // Shorthand for untagged GPU timings (frame_serial 0 attaches them to the frame just drawn).
    auto on_frame_drawn(FrameTimings const &timings, bool streaming_idle,
                        BenchmarkCounters const &counters = {}) -> void;

    // Simulated seconds for the frame about to be drawn: 0 while warming up, then measured_frames * timestep. Warmup
    // length depends on when streaming settles, so shader time must not accumulate through it.
    [[nodiscard]]
    auto simulated_time() const noexcept -> float {
        return phase_ == Phase::warmup ? 0.0F : static_cast<float>(measured_frames_) * benchmark_timestep;
    }

    // True if the frame about to be drawn is the first measured one at or past a keyframe.
    [[nodiscard]]
    auto at_keyframe() const noexcept -> bool;

    [[nodiscard]] auto measuring() const noexcept -> bool { return phase_ == Phase::measuring; }

    // The lap is done and the GPU timings of its frames are in (or stopped coming).
    [[nodiscard]]
    auto finished() const noexcept -> bool {
        return phase_ == Phase::finished;
    }

    [[nodiscard]]
    auto options() const noexcept -> BenchmarkOptions const & {
        return options_;
    }

    [[nodiscard]] auto samples() const noexcept -> std::span<BenchmarkFrameSample const> { return samples_; }

    [[nodiscard]] auto analyse() const -> FrameAnalysis;
    [[nodiscard]] auto stages() const -> std::vector<BenchmarkStage>;

    [[nodiscard]] auto result(BenchmarkCaseId id, BenchmarkEnvironment const &environment) const -> BenchmarkCaseResult;

    // The single-run JSON: the earlier keys (stages, counters, full_frame_ms; tools/perf/compare_benchmarks.py reads
    // them) plus "environment" and "analysis".
    [[nodiscard]]
    auto to_json(BenchmarkEnvironment const &environment) const -> std::string;

    [[nodiscard]] auto write(BenchmarkEnvironment const &environment) const -> std::expected<void, std::string>;

    // One row per measured frame: CPU phases, displayed interval, GPU frame and pass times, workload, events.
    [[nodiscard]] auto to_csv() const -> std::string;

private:
    enum class Phase : std::uint8_t { warmup, measuring, draining, finished };

    auto attach_gpu(FrameTimings const &timings, std::size_t row) -> void;
    [[nodiscard]] auto pending_gpu_rows() const noexcept -> bool;

    BenchmarkOptions options_;
    std::vector<CameraKeyframe> keyframes_;

    Phase phase_ = Phase::warmup;
    bool streaming_settled_ = true;
    std::uint32_t warmup_frames_ = 0;
    std::uint32_t measured_frames_ = 0;
    std::uint32_t drain_frames_ = 0;
    std::uint64_t last_gpu_serial_ = 0;

    std::vector<BenchmarkFrameSample> samples_;

    // Per counter (BenchmarkCounters field order, see counter_names in benchmark.cxx): the sum over the measured frames
    // whose group was valid, how many those were, and the value of the last such frame.
    static constexpr std::size_t counter_count = 11;
    std::array<double, counter_count> counter_sums_{};
    std::array<std::uint32_t, counter_count> counter_samples_{};
    std::array<std::uint32_t, counter_count> counter_final_{};

    // One entry per frame graph pass seen (keyed by its stable name_id, in order of first appearance), each with one
    // value per sample; only rows whose gpu_valid is set mean anything. A pass absent from a frame counts as 0 ms.
    struct StageSamples {
        std::string id;
        std::string name;
        std::vector<float> samples_ms;
    };
    std::vector<StageSamples> stages_;
};

// Writes `result` as one JSON object (without per-frame arrays) under `key`.
auto write_case_json(JsonWriter &writer, std::string_view key, BenchmarkCaseResult const &result) -> void;
auto write_environment_json(JsonWriter &writer, std::string_view key, BenchmarkEnvironment const &environment) -> void;
auto write_analysis_json(JsonWriter &writer, std::string_view key, FrameAnalysis const &analysis) -> void;

// ---- Suite

// One run of the plan. Cases run repeat-major (every case once, then every case again, ...), so slow drift such as
// heat soak spreads over all of them instead of landing on whichever ran last.
struct BenchmarkCase {
    std::size_t scenario = 0;
    BenchmarkCaseId id;
};

struct BenchmarkScenarioInfo {
    std::string name;
    std::string load_axis;
    std::vector<std::uint32_t> default_loads;
};

[[nodiscard]]
auto plan_benchmark_cases(std::span<BenchmarkScenarioInfo const> scenarios,
                          BenchmarkOptions const &options) -> std::expected<std::vector<BenchmarkCase>, std::string>;

// The repeats of one case, aggregated, and how each load level of a scenario scales.
struct BenchmarkAggregate {
    std::string key;
    std::string scenario;
    std::string load_axis;
    std::uint32_t load = 0;
    std::uint32_t repeats = 0;

    // Metric name -> stats across repeats; see aggregate_metric_names().
    std::vector<std::pair<std::string, RepeatStats>> metrics;

    [[nodiscard]] auto metric(std::string_view name) const noexcept -> RepeatStats const *;
};

struct BenchmarkScaling {
    std::string scenario;
    std::string load_axis;
    std::vector<std::uint32_t> loads;
    std::vector<double> gpu_median_ms;
    std::vector<double> cpu_busy_median_ms;
    std::vector<double> displayed_p99_ms;
    ScalingFit gpu;
    ScalingFit cpu_busy;
    ScalingFit displayed_p99;
};

[[nodiscard]]
auto aggregate_cases(std::span<BenchmarkCaseResult const> results) -> std::vector<BenchmarkAggregate>;

[[nodiscard]]
auto scaling_of(std::span<BenchmarkAggregate const> aggregates) -> std::vector<BenchmarkScaling>;

[[nodiscard]]
auto suite_to_json(BenchmarkOptions const &options, BenchmarkEnvironment const &environment,
                   std::span<BenchmarkCaseResult const> results) -> std::string;

// Human-readable summary of a suite: per-case table, scaling, hitch attribution and warnings.
[[nodiscard]]
auto suite_report_markdown(BenchmarkOptions const &options, BenchmarkEnvironment const &environment,
                           std::span<BenchmarkCaseResult const> results) -> std::string;

// Creates parent directories and writes `contents`.
[[nodiscard]]
auto write_text_file(std::filesystem::path const &path, std::string_view contents) -> std::expected<void, std::string>;
