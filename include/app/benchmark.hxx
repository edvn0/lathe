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
class CommandLine;
class JsonWriter;

enum class BenchmarkMode : std::uint8_t {
    single,
    suite,
};

struct BenchmarkRenderSize {
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    auto operator==(BenchmarkRenderSize const &) const -> bool = default;
};

struct BenchmarkSweep {
    std::string scenario;
    std::vector<std::uint32_t> loads;
};

struct BenchmarkOptions {
    BenchmarkMode mode = BenchmarkMode::single;

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

    std::uint32_t drain_frame_limit = 8;
};

inline constexpr BenchmarkRenderSize benchmark_default_suite_render_size{.width = 1920, .height = 1080};
inline constexpr std::uint32_t benchmark_default_suite_repeats = 3;

enum class PresentModeChoice : std::uint8_t {
    immediate,
    mailbox,
    fifo,
    fifo_relaxed,
};

struct PresentationArguments {
    explicit PresentationArguments(CommandLine &cli);

    PresentationArguments(PresentationArguments const &) = delete;
    auto operator=(PresentationArguments const &) -> PresentationArguments & = delete;

    std::optional<bool> vsync;
    std::optional<PresentModeChoice> present_mode;
    std::optional<std::uint32_t> swapchain_images;
};

class BenchmarkArguments {
public:
    explicit BenchmarkArguments(CommandLine &cli);

    BenchmarkArguments(BenchmarkArguments const &) = delete;
    auto operator=(BenchmarkArguments const &) -> BenchmarkArguments & = delete;

    [[nodiscard]]
    auto options() const -> std::expected<std::optional<BenchmarkOptions>, std::string>;

private:
    BenchmarkOptions options_;
    std::optional<std::filesystem::path> single_;
    std::optional<std::filesystem::path> suite_;
    std::optional<std::uint32_t> repeats_;
};

[[nodiscard]]
auto parse_render_size(std::string_view text) -> std::expected<BenchmarkRenderSize, std::string>;

inline constexpr float benchmark_timestep = 1.0F / 60.0F;

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
    std::string requested_present_mode;
    std::uint32_t swapchain_images = 0;
    std::uint32_t frames_in_flight = 0;
    ClusterGridSettings cluster_grid{};

    bool occlusion_culling = false;

    bool meshlet_occlusion = false;

    std::string build_type;
    std::string source_revision;
    bool memory_tracking = false;

    std::string cpu_name;
    std::uint32_t cpu_threads = 0;
    std::string cpu_governor;
    std::string os;
    std::string timestamp_utc;

    std::vector<std::string> warnings;
};

struct BenchmarkStage {
    std::string id;
    std::string name;
    TimingSummary summary;
};

struct BenchmarkCaseId {
    std::string scenario;
    std::string load_axis;
    std::uint32_t load = 0;
    std::uint32_t repeat = 0;

    [[nodiscard]] auto key() const -> std::string;
    [[nodiscard]] auto file_stem() const -> std::string;
};

struct BenchmarkCaseResult {
    BenchmarkCaseId id;
    std::uint32_t keyframes = 0;
    std::uint32_t warmup_frames = 0;
    bool streaming_settled = true;
    std::uint32_t frames_with_timings = 0;
    std::uint32_t render_width = 0;
    std::uint32_t render_height = 0;
    FrameAnalysis analysis;
    std::vector<BenchmarkStage> stages;
    ThermalSample thermals_start{};
    ThermalSample thermals_end{};
};

struct BenchmarkFrameInput {
    FrameTimings const *gpu = nullptr;

    std::uint64_t frame_serial = 0;

    CpuFrameTimes cpu{};
    bool streaming_idle = true;
    BenchmarkCounters counters{};
    BenchmarkWorkload workload{};
    PerfEventCounts events{};
    std::uint64_t allocations = 0;
    std::uint64_t allocated_bytes = 0;
};

class BenchmarkRun {
public:
    BenchmarkRun(BenchmarkOptions options, std::vector<CameraKeyframe> keyframes);

    [[nodiscard]]
    auto camera() const noexcept -> CameraKeyframe;

    auto on_frame_drawn(BenchmarkFrameInput const &input) -> void;

    auto on_frame_drawn(FrameTimings const &timings, bool streaming_idle,
                        BenchmarkCounters const &counters = {}) -> void;

    [[nodiscard]]
    auto simulated_time() const noexcept -> float {
        return phase_ == Phase::warmup ? 0.0F : static_cast<float>(measured_frames_) * benchmark_timestep;
    }

    [[nodiscard]]
    auto at_keyframe() const noexcept -> bool;

    [[nodiscard]] auto measuring() const noexcept -> bool { return phase_ == Phase::measuring; }

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

    [[nodiscard]]
    auto to_json(BenchmarkEnvironment const &environment) const -> std::string;

    [[nodiscard]] auto write(BenchmarkEnvironment const &environment) const -> std::expected<void, std::string>;

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

    static constexpr std::size_t counter_count = 11;
    std::array<double, counter_count> counter_sums_{};
    std::array<std::uint32_t, counter_count> counter_samples_{};
    std::array<std::uint32_t, counter_count> counter_final_{};

    struct StageSamples {
        std::string id;
        std::string name;
        std::vector<float> samples_ms;
    };
    std::vector<StageSamples> stages_;
};

auto write_case_json(JsonWriter &writer, std::string_view key, BenchmarkCaseResult const &result) -> void;
auto write_environment_json(JsonWriter &writer, std::string_view key, BenchmarkEnvironment const &environment) -> void;
auto write_analysis_json(JsonWriter &writer, std::string_view key, FrameAnalysis const &analysis) -> void;

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

struct BenchmarkAggregate {
    std::string key;
    std::string scenario;
    std::string load_axis;
    std::uint32_t load = 0;
    std::uint32_t repeats = 0;

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

[[nodiscard]]
auto suite_report_markdown(BenchmarkOptions const &options, BenchmarkEnvironment const &environment,
                           std::span<BenchmarkCaseResult const> results) -> std::string;

[[nodiscard]]
auto write_text_file(std::filesystem::path const &path, std::string_view contents) -> std::expected<void, std::string>;
