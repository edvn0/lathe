#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "app/frame_clock.hxx"
#include "core/perf_events.hxx"

// Statistics over a benchmark run's per-frame samples (docs/perf-benchmark.md, "Reading the results"). Pure functions
// of the samples, so they are unit tested without a GPU.

struct TimingSummary {
    std::uint32_t count = 0;
    float mean_ms = 0.0F;
    float stddev_ms = 0.0F;
    float min_ms = 0.0F;
    float median_ms = 0.0F;
    float p90_ms = 0.0F;
    float p95_ms = 0.0F;
    float p99_ms = 0.0F;
    float p999_ms = 0.0F;
    float max_ms = 0.0F;
};

// Nearest-rank percentiles (the smallest sample with at least that fraction of samples at or below it) and the
// population standard deviation; all zero when empty.
[[nodiscard]]
auto summarise_timings(std::span<float const> samples_ms) -> TimingSummary;

// What the frame submitted, from the renderer's FrameStats for that same frame (not lagged).
struct BenchmarkWorkload {
    std::uint32_t submitted_triangles = 0;
    std::uint32_t submitted_instances = 0;
    std::uint32_t indirect_commands = 0;
    std::uint32_t model_submissions = 0;
    std::uint32_t mesh_submissions = 0;
    std::uint32_t point_lights = 0;
    std::uint32_t spot_lights = 0;
};

// One measured frame. The GPU part arrives frames_in_flight frames later and is matched by `serial`.
struct BenchmarkFrameSample {
    std::uint32_t index = 0;

    // Renderer::recorded_frame_count() after this frame was recorded; 0 if nothing was recorded (e.g. the swapchain
    // was recreated instead), in which case GPU data never arrives for it.
    std::uint64_t serial = 0;

    // Position along the camera path, [0, 1).
    float path_t = 0.0F;

    CpuFrameTimes cpu{};

    bool gpu_valid = false;
    float gpu_frame_ms = 0.0F;

    BenchmarkWorkload workload{};

    // What happened during this frame (perf_events deltas) and its heap traffic (MemoryTracker deltas; zero in builds
    // without memory tracking).
    PerfEventCounts events{};
    std::uint64_t allocations = 0;
    std::uint64_t allocated_bytes = 0;

    // The interval the player saw: present to present when known, otherwise the CPU frame.
    [[nodiscard]] auto displayed_ms() const noexcept -> float {
        return cpu.present_interval_ms > 0.0F ? cpu.present_interval_ms : cpu.frame_ms;
    }
};

struct AnalysisOptions {
    // The refresh rate frames are judged against; the budget is 1000 / target_hz ms.
    float target_hz = 144.0F;

    // A hitch is a displayed frame longer than this many times the run's median, and over budget.
    float hitch_factor = 2.0F;
};

struct BudgetStats {
    float target_hz = 0.0F;
    float budget_ms = 0.0F;
    std::uint32_t frames = 0;
    std::uint32_t displayed_over_budget = 0;
    std::uint32_t cpu_busy_over_budget = 0;
    std::uint32_t gpu_over_budget = 0;
    std::uint32_t frames_with_gpu = 0;
};

struct HitchFrame {
    std::uint32_t index = 0;
    float displayed_ms = 0.0F;
    float cpu_busy_ms = 0.0F;
    std::optional<float> gpu_frame_ms;

    // Events in this frame and the one before it: work done in frame N often shows up in N's or N+1's present.
    PerfEventCounts events{};
};

struct EventCorrelation {
    std::uint64_t total = 0;
    std::uint32_t frames_with = 0;
    float mean_displayed_with_ms = 0.0F;
    float mean_displayed_without_ms = 0.0F;
    std::uint32_t hitches_with = 0;
};

// Which side limits the frame rate. A frame is GPU-bound when its GPU time is at least its CPU busy time: the CPU
// would have had time to spare.
struct BoundStats {
    std::uint32_t frames_with_gpu = 0;
    std::uint32_t gpu_bound = 0;
    std::uint32_t cpu_bound = 0;

    [[nodiscard]] auto gpu_bound_fraction() const noexcept -> float {
        return frames_with_gpu == 0 ? 0.0F : static_cast<float>(gpu_bound) / static_cast<float>(frames_with_gpu);
    }
};

struct WorkloadSummary {
    double submitted_triangles = 0.0;
    double submitted_instances = 0.0;
    double indirect_commands = 0.0;
    double model_submissions = 0.0;
    double mesh_submissions = 0.0;
    double point_lights = 0.0;
    double spot_lights = 0.0;
};

struct FrameAnalysis {
    std::uint32_t frames = 0;

    TimingSummary displayed{};
    TimingSummary cpu_frame{};
    TimingSummary cpu_busy{};
    TimingSummary cpu_wait{};
    TimingSummary gpu_frame{};
    std::array<TimingSummary, cpu_phase_count> cpu_phases{};

    BudgetStats budget{};

    // 1000 / mean and 1000 / p99 of the displayed interval: the usual "average" and "1% low" frame rates.
    float average_fps = 0.0F;
    float one_percent_low_fps = 0.0F;

    float hitch_threshold_ms = 0.0F;
    std::vector<HitchFrame> hitches;

    std::array<EventCorrelation, perf_event_count> events{};

    BoundStats bound{};

    WorkloadSummary workload{};
    double mean_allocations = 0.0;
    std::uint64_t max_allocations = 0;
    double mean_allocated_bytes = 0.0;
};

[[nodiscard]]
auto analyse_frames(std::span<BenchmarkFrameSample const> samples,
                    AnalysisOptions const &options = {}) -> FrameAnalysis;

// One metric across the repeats of a case: the median of the per-repeat values and their range. With three or more
// repeats, a change between two builds whose ranges don't overlap is unlikely to be noise.
struct RepeatStats {
    std::vector<double> values;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;

    // (max - min) / median: the run-to-run noise of this metric on this machine.
    [[nodiscard]] auto spread() const noexcept -> double { return median > 0.0 ? (max - min) / median : 0.0; }
};

[[nodiscard]]
auto summarise_repeats(std::span<double const> values) -> RepeatStats;

// Least-squares fits of a cost metric against the load axis (objects, lights, ...). `linear` gives the marginal
// cost per unit of load; `exponent` comes from a log-log fit (cost ~ load^k): about 1 means linear scaling, below 1
// means a fixed cost dominates, above 1 means each unit gets more expensive as load grows.
struct ScalingFit {
    std::size_t points = 0;
    double intercept_ms = 0.0;
    double slope_ms_per_unit = 0.0;
    double r_squared = 0.0;
    std::optional<double> exponent;

    // The load at which the linear fit reaches `budget_ms`, if it is rising and does so beyond the origin.
    [[nodiscard]] auto load_at(double budget_ms) const noexcept -> std::optional<double>;
};

// `loads` and `costs_ms` are paired. Needs at least two distinct loads; otherwise points < 2 and the rest is zero.
[[nodiscard]]
auto fit_scaling(std::span<double const> loads, std::span<double const> costs_ms) -> ScalingFit;
