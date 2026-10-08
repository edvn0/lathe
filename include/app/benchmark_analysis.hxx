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

[[nodiscard]]
auto summarise_timings(std::span<float const> samples_ms) -> TimingSummary;

struct BenchmarkWorkload {
    std::uint32_t submitted_triangles = 0;
    std::uint32_t submitted_instances = 0;
    std::uint32_t indirect_commands = 0;
    std::uint32_t model_submissions = 0;
    std::uint32_t mesh_submissions = 0;
    std::uint32_t point_lights = 0;
    std::uint32_t spot_lights = 0;
};

struct BenchmarkFrameSample {
    std::uint32_t index = 0;

    std::uint64_t serial = 0;

    float path_t = 0.0F;

    CpuFrameTimes cpu{};

    bool gpu_valid = false;
    float gpu_frame_ms = 0.0F;

    BenchmarkWorkload workload{};

    PerfEventCounts events{};
    std::uint64_t allocations = 0;
    std::uint64_t allocated_bytes = 0;

    [[nodiscard]] auto displayed_ms() const noexcept -> float {
        return cpu.present_interval_ms > 0.0F ? cpu.present_interval_ms : cpu.frame_ms;
    }
};

struct AnalysisOptions {
    float target_hz = 144.0F;

    float hitch_factor = 2.0F;

    float budget_tolerance = 0.01F;

    float presentation_wait_share = 0.25F;
    float gpu_idle_share = 0.9F;
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

    PerfEventCounts events{};
};

struct EventCorrelation {
    std::uint64_t total = 0;
    std::uint32_t frames_with = 0;
    float mean_displayed_with_ms = 0.0F;
    float mean_displayed_without_ms = 0.0F;
    std::uint32_t hitches_with = 0;
};

struct BoundStats {
    std::uint32_t frames_with_gpu = 0;
    std::uint32_t gpu_bound = 0;
    std::uint32_t cpu_bound = 0;
    std::uint32_t presentation_bound = 0;

    [[nodiscard]] auto fraction(std::uint32_t count) const noexcept -> float {
        return frames_with_gpu == 0 ? 0.0F : static_cast<float>(count) / static_cast<float>(frames_with_gpu);
    }
    [[nodiscard]] auto gpu_bound_fraction() const noexcept -> float { return fraction(gpu_bound); }
    [[nodiscard]] auto cpu_bound_fraction() const noexcept -> float { return fraction(cpu_bound); }
    [[nodiscard]] auto presentation_bound_fraction() const noexcept -> float { return fraction(presentation_bound); }
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

struct RepeatStats {
    std::vector<double> values;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;

    [[nodiscard]] auto spread() const noexcept -> double { return median > 0.0 ? (max - min) / median : 0.0; }
};

[[nodiscard]]
auto summarise_repeats(std::span<double const> values) -> RepeatStats;

struct ScalingFit {
    std::size_t points = 0;
    double intercept_ms = 0.0;
    double slope_ms_per_unit = 0.0;
    double r_squared = 0.0;
    std::optional<double> exponent;

    [[nodiscard]] auto load_at(double budget_ms) const noexcept -> std::optional<double>;
};

[[nodiscard]]
auto fit_scaling(std::span<double const> loads, std::span<double const> costs_ms) -> ScalingFit;
