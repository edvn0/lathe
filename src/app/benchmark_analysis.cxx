#include "app/benchmark_analysis.hxx"

#include <algorithm>
#include <cmath>
#include <numeric>

auto summarise_timings(std::span<float const> samples_ms) -> TimingSummary {
    if (samples_ms.empty()) {
        return {};
    }

    std::vector<float> sorted{samples_ms.begin(), samples_ms.end()};
    std::ranges::sort(sorted);

    auto const percentile = [&](double fraction) {
        auto const rank = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(sorted.size())));
        return sorted[std::clamp(rank, std::size_t{1}, sorted.size()) - 1];
    };

    auto const count = static_cast<double>(sorted.size());
    auto const sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    auto const mean = sum / count;

    auto squares = 0.0;
    for (auto const sample: sorted) {
        auto const difference = static_cast<double>(sample) - mean;
        squares += difference * difference;
    }

    return TimingSummary{
            .count = static_cast<std::uint32_t>(sorted.size()),
            .mean_ms = static_cast<float>(mean),
            .stddev_ms = static_cast<float>(std::sqrt(squares / count)),
            .min_ms = sorted.front(),
            .median_ms = percentile(0.5),
            .p90_ms = percentile(0.90),
            .p95_ms = percentile(0.95),
            .p99_ms = percentile(0.99),
            .p999_ms = percentile(0.999),
            .max_ms = sorted.back(),
    };
}

namespace {

    template<typename Projection>
    [[nodiscard]] auto summarise_by(std::span<BenchmarkFrameSample const> samples,
                                    Projection projection) -> TimingSummary {
        std::vector<float> values;
        values.reserve(samples.size());
        for (auto const &sample: samples) {
            values.push_back(projection(sample));
        }
        return summarise_timings(values);
    }

}

auto analyse_frames(std::span<BenchmarkFrameSample const> samples, AnalysisOptions const &options) -> FrameAnalysis {
    FrameAnalysis analysis;
    analysis.frames = static_cast<std::uint32_t>(samples.size());

    analysis.displayed = summarise_by(samples, [](auto const &sample) { return sample.displayed_ms(); });
    analysis.cpu_frame = summarise_by(samples, [](auto const &sample) { return sample.cpu.frame_ms; });
    analysis.cpu_busy = summarise_by(samples, [](auto const &sample) { return sample.cpu.busy_ms(); });
    analysis.cpu_wait = summarise_by(samples, [](auto const &sample) { return sample.cpu.wait_ms(); });

    for (std::size_t phase = 0; phase < cpu_phase_count; ++phase) {
        analysis.cpu_phases[phase] =
                summarise_by(samples, [phase](auto const &sample) { return sample.cpu.phase_ms[phase]; });
    }

    std::vector<float> gpu_values;
    gpu_values.reserve(samples.size());
    for (auto const &sample: samples) {
        if (sample.gpu_valid) {
            gpu_values.push_back(sample.gpu_frame_ms);
        }
    }
    analysis.gpu_frame = summarise_timings(gpu_values);

    if (samples.empty()) {
        return analysis;
    }

    auto const target_hz = options.target_hz > 0.0F ? options.target_hz : 144.0F;
    auto const budget_ms = 1000.0F / target_hz;
    analysis.budget = BudgetStats{.target_hz = target_hz, .budget_ms = budget_ms, .frames = analysis.frames};

    if (analysis.displayed.mean_ms > 0.0F) {
        analysis.average_fps = 1000.0F / analysis.displayed.mean_ms;
    }
    if (analysis.displayed.p99_ms > 0.0F) {
        analysis.one_percent_low_fps = 1000.0F / analysis.displayed.p99_ms;
    }

    analysis.hitch_threshold_ms = std::max(options.hitch_factor * analysis.displayed.median_ms, budget_ms);
    auto const over_budget_ms = budget_ms * (1.0F + options.budget_tolerance);

    std::array<double, perf_event_count> with_sum{};
    std::array<double, perf_event_count> without_sum{};

    for (std::size_t i = 0; i < samples.size(); ++i) {
        auto const &sample = samples[i];
        auto const displayed = sample.displayed_ms();

        analysis.budget.displayed_over_budget += displayed > over_budget_ms ? 1U : 0U;
        analysis.budget.cpu_busy_over_budget += sample.cpu.busy_ms() > over_budget_ms ? 1U : 0U;

        if (sample.gpu_valid) {
            ++analysis.budget.frames_with_gpu;
            analysis.budget.gpu_over_budget += sample.gpu_frame_ms > over_budget_ms ? 1U : 0U;

            ++analysis.bound.frames_with_gpu;
            auto const presentation_wait = sample.cpu.phase(CpuPhase::acquire) + sample.cpu.phase(CpuPhase::present);
            auto const presentation_bound = presentation_wait >= options.presentation_wait_share * displayed &&
                                            sample.gpu_frame_ms < options.gpu_idle_share * displayed;
            if (presentation_bound) {
                ++analysis.bound.presentation_bound;
            } else if (sample.gpu_frame_ms >= sample.cpu.busy_ms()) {
                ++analysis.bound.gpu_bound;
            } else {
                ++analysis.bound.cpu_bound;
            }
        }

        auto const is_hitch = displayed > analysis.hitch_threshold_ms;

        auto window = sample.events;
        if (i > 0) {
            for (std::size_t event = 0; event < perf_event_count; ++event) {
                window.values[event] += samples[i - 1].events.values[event];
            }
        }

        if (is_hitch) {
            analysis.hitches.push_back(HitchFrame{
                    .index = sample.index,
                    .displayed_ms = displayed,
                    .cpu_busy_ms = sample.cpu.busy_ms(),
                    .gpu_frame_ms = sample.gpu_valid ? std::optional<float>{sample.gpu_frame_ms} : std::nullopt,
                    .events = window,
            });
        }

        for (std::size_t event = 0; event < perf_event_count; ++event) {
            auto &correlation = analysis.events[event];
            correlation.total += sample.events.values[event];

            if (sample.events.values[event] != 0) {
                ++correlation.frames_with;
                with_sum[event] += displayed;
            } else {
                without_sum[event] += displayed;
            }

            if (is_hitch && window.values[event] != 0) {
                ++correlation.hitches_with;
            }
        }

        auto &workload = analysis.workload;
        workload.submitted_triangles += sample.workload.submitted_triangles;
        workload.submitted_instances += sample.workload.submitted_instances;
        workload.indirect_commands += sample.workload.indirect_commands;
        workload.model_submissions += sample.workload.model_submissions;
        workload.mesh_submissions += sample.workload.mesh_submissions;
        workload.point_lights += sample.workload.point_lights;
        workload.spot_lights += sample.workload.spot_lights;

        analysis.mean_allocations += static_cast<double>(sample.allocations);
        analysis.mean_allocated_bytes += static_cast<double>(sample.allocated_bytes);
        analysis.max_allocations = std::max(analysis.max_allocations, sample.allocations);
    }

    auto const count = static_cast<double>(samples.size());

    for (std::size_t event = 0; event < perf_event_count; ++event) {
        auto &correlation = analysis.events[event];
        auto const without = samples.size() - correlation.frames_with;
        if (correlation.frames_with != 0) {
            correlation.mean_displayed_with_ms = static_cast<float>(with_sum[event] / correlation.frames_with);
        }
        if (without != 0) {
            correlation.mean_displayed_without_ms =
                    static_cast<float>(without_sum[event] / static_cast<double>(without));
        }
    }

    auto &workload = analysis.workload;
    workload.submitted_triangles /= count;
    workload.submitted_instances /= count;
    workload.indirect_commands /= count;
    workload.model_submissions /= count;
    workload.mesh_submissions /= count;
    workload.point_lights /= count;
    workload.spot_lights /= count;

    analysis.mean_allocations /= count;
    analysis.mean_allocated_bytes /= count;

    return analysis;
}

auto summarise_repeats(std::span<double const> values) -> RepeatStats {
    RepeatStats stats;
    stats.values.assign(values.begin(), values.end());

    if (values.empty()) {
        return stats;
    }

    std::vector<double> sorted{values.begin(), values.end()};
    std::ranges::sort(sorted);

    auto const middle = sorted.size() / 2;
    stats.median = sorted.size() % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) * 0.5;
    stats.min = sorted.front();
    stats.max = sorted.back();

    return stats;
}

namespace {

    struct LineFit {
        double intercept = 0.0;
        double slope = 0.0;
        double r_squared = 0.0;
        bool valid = false;
    };

    [[nodiscard]] auto fit_line(std::span<double const> xs, std::span<double const> ys) -> LineFit {
        auto const count = static_cast<double>(xs.size());
        if (xs.size() < 2) {
            return {};
        }

        auto const mean_x = std::accumulate(xs.begin(), xs.end(), 0.0) / count;
        auto const mean_y = std::accumulate(ys.begin(), ys.end(), 0.0) / count;

        auto covariance = 0.0;
        auto variance_x = 0.0;
        auto variance_y = 0.0;
        for (std::size_t i = 0; i < xs.size(); ++i) {
            covariance += (xs[i] - mean_x) * (ys[i] - mean_y);
            variance_x += (xs[i] - mean_x) * (xs[i] - mean_x);
            variance_y += (ys[i] - mean_y) * (ys[i] - mean_y);
        }

        if (variance_x <= 0.0) {
            return {};
        }

        auto const slope = covariance / variance_x;
        auto const r_squared = variance_y > 0.0 ? (covariance * covariance) / (variance_x * variance_y) : 1.0;

        return LineFit{.intercept = mean_y - slope * mean_x, .slope = slope, .r_squared = r_squared, .valid = true};
    }

}

auto fit_scaling(std::span<double const> loads, std::span<double const> costs_ms) -> ScalingFit {
    auto const count = std::min(loads.size(), costs_ms.size());
    auto const xs = loads.first(count);
    auto const ys = costs_ms.first(count);

    auto const linear = fit_line(xs, ys);
    if (!linear.valid) {
        return ScalingFit{.points = count};
    }

    ScalingFit fit{
            .points = count,
            .intercept_ms = linear.intercept,
            .slope_ms_per_unit = linear.slope,
            .r_squared = linear.r_squared,
    };

    std::vector<double> log_x;
    std::vector<double> log_y;
    for (std::size_t i = 0; i < count; ++i) {
        if (xs[i] > 0.0 && ys[i] > 0.0) {
            log_x.push_back(std::log(xs[i]));
            log_y.push_back(std::log(ys[i]));
        }
    }

    if (auto const power = fit_line(log_x, log_y); power.valid) {
        fit.exponent = power.slope;
    }

    return fit;
}

auto ScalingFit::load_at(double budget_ms) const noexcept -> std::optional<double> {
    if (points < 2 || slope_ms_per_unit <= 0.0) {
        return std::nullopt;
    }

    auto const load = (budget_ms - intercept_ms) / slope_ms_per_unit;
    return load > 0.0 ? std::optional<double>{load} : std::nullopt;
}
