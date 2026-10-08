#include <doctest/doctest.h>

#include <array>
#include <vector>

#include "app/benchmark_analysis.hxx"

namespace {

    [[nodiscard]] auto frame(std::uint32_t index, float displayed_ms, float busy_ms = 2.0F,
                             std::optional<float> gpu_ms = std::nullopt) -> BenchmarkFrameSample {
        BenchmarkFrameSample sample{.index = index};
        sample.cpu.frame_ms = displayed_ms;
        sample.cpu.present_interval_ms = displayed_ms;
        sample.cpu.phase_ms[static_cast<std::size_t>(CpuPhase::record)] = busy_ms;
        sample.cpu.phase_ms[static_cast<std::size_t>(CpuPhase::slot_wait)] = displayed_ms - busy_ms;
        if (gpu_ms) {
            sample.gpu_valid = true;
            sample.gpu_frame_ms = *gpu_ms;
        }
        return sample;
    }

}

TEST_CASE("timing summaries report tail percentiles and spread") {
    std::vector<float> samples;
    for (int i = 1000; i >= 1; --i) {
        samples.push_back(static_cast<float>(i));
    }

    auto const summary = summarise_timings(samples);
    CHECK(summary.count == 1000);
    CHECK(summary.median_ms == 500.0F);
    CHECK(summary.p90_ms == 900.0F);
    CHECK(summary.p99_ms == 990.0F);
    CHECK(summary.p999_ms == 999.0F);
    CHECK(summary.max_ms == 1000.0F);

    std::array const known{2.0F, 4.0F, 4.0F, 4.0F, 5.0F, 5.0F, 7.0F, 9.0F};
    CHECK(summarise_timings(known).stddev_ms == doctest::Approx(2.0F));
    CHECK(summarise_timings({}).count == 0);
}

TEST_CASE("cpu frame times split work from waiting") {
    CpuFrameTimes times{};
    times.frame_ms = 10.0F;
    times.phase_ms[static_cast<std::size_t>(CpuPhase::slot_wait)] = 3.0F;
    times.phase_ms[static_cast<std::size_t>(CpuPhase::acquire)] = 1.0F;
    times.phase_ms[static_cast<std::size_t>(CpuPhase::record)] = 2.0F;

    CHECK(times.wait_ms() == doctest::Approx(4.0F));
    CHECK(times.busy_ms() == doctest::Approx(6.0F));
}

TEST_CASE("analysis finds hitches, attributes them to events and judges the budget") {
    std::vector<BenchmarkFrameSample> samples;
    samples.reserve(100);
    for (std::uint32_t i = 0; i < 100; ++i) {
        samples.push_back(frame(i, 5.0F, 2.0F, 4.0F));
    }

    samples[39].events.values[static_cast<std::size_t>(PerfEvent::texture_upload)] = 2;
    samples[40] = frame(40, 30.0F, 25.0F, 4.0F);

    samples[70] = frame(70, 20.0F, 18.0F, 4.0F);

    samples[90] = frame(90, 5.0F, 2.0F, 6.0F);

    auto const analysis = analyse_frames(samples, AnalysisOptions{.target_hz = 144.0F});

    CHECK(analysis.frames == 100);
    CHECK(analysis.displayed.median_ms == 5.0F);
    CHECK(analysis.budget.budget_ms == doctest::Approx(1000.0F / 144.0F));
    CHECK(analysis.budget.displayed_over_budget == 2);
    CHECK(analysis.budget.cpu_busy_over_budget == 2);

    CHECK(analysis.hitch_threshold_ms == doctest::Approx(10.0F));
    REQUIRE(analysis.hitches.size() == 2);
    CHECK(analysis.hitches[0].index == 40);
    CHECK(analysis.hitches[0].events[PerfEvent::texture_upload] == 2);
    CHECK_FALSE(analysis.hitches[1].events.any());

    auto const &uploads = analysis.events[static_cast<std::size_t>(PerfEvent::texture_upload)];
    CHECK(uploads.total == 2);
    CHECK(uploads.frames_with == 1);
    CHECK(uploads.hitches_with == 1);

    CHECK(analysis.bound.frames_with_gpu == 100);
    CHECK(analysis.bound.cpu_bound == 2);
    CHECK(analysis.bound.gpu_bound == 98);

    CHECK(analysis.average_fps == doctest::Approx(1000.0F / analysis.displayed.mean_ms));
    CHECK(analysis.one_percent_low_fps == doctest::Approx(1000.0F / analysis.displayed.p99_ms));
}

TEST_CASE("frames without GPU timings are left out of GPU statistics") {
    std::vector<BenchmarkFrameSample> samples{frame(0, 5.0F, 2.0F, 3.0F), frame(1, 5.0F), frame(2, 5.0F, 2.0F, 5.0F)};

    auto const analysis = analyse_frames(samples);
    CHECK(analysis.gpu_frame.count == 2);
    CHECK(analysis.gpu_frame.mean_ms == doctest::Approx(4.0F));
    CHECK(analysis.bound.frames_with_gpu == 2);
}

TEST_CASE("repeat statistics use the median and the range") {
    std::array const odd{3.0, 1.0, 2.0};
    auto const odd_stats = summarise_repeats(odd);
    CHECK(odd_stats.median == 2.0);
    CHECK(odd_stats.min == 1.0);
    CHECK(odd_stats.max == 3.0);
    CHECK(odd_stats.spread() == doctest::Approx(1.0));

    std::array const even{4.0, 1.0, 2.0, 3.0};
    CHECK(summarise_repeats(even).median == 2.5);
}

TEST_CASE("scaling fits give marginal cost, exponent and the load that fills the budget") {
    std::array const loads{1000.0, 2000.0, 4000.0, 8000.0};
    std::array const linear_cost{2.0, 3.0, 5.0, 9.0};

    auto const fit = fit_scaling(loads, linear_cost);
    CHECK(fit.points == 4);
    CHECK(fit.slope_ms_per_unit == doctest::Approx(0.001));
    CHECK(fit.intercept_ms == doctest::Approx(1.0));
    CHECK(fit.r_squared == doctest::Approx(1.0));
    REQUIRE(fit.exponent.has_value());
    CHECK(*fit.exponent < 1.0);
    REQUIRE(fit.load_at(6.0).has_value());
    CHECK(*fit.load_at(6.0) == doctest::Approx(5000.0));

    std::array const quadratic{1.0, 4.0, 16.0, 64.0};
    auto const steep = fit_scaling(loads, quadratic);
    REQUIRE(steep.exponent.has_value());
    CHECK(*steep.exponent == doctest::Approx(2.0));

    std::array const one_load{1000.0, 1000.0};
    std::array const two_costs{1.0, 2.0};
    CHECK_FALSE(fit_scaling(one_load, two_costs).load_at(6.0).has_value());
}

TEST_CASE("frames the swapchain paced are presentation-bound, and refresh jitter is not a budget miss") {
    std::vector<BenchmarkFrameSample> samples;
    samples.reserve(10);

    for (std::uint32_t i = 0; i < 8; ++i) {
        auto sample = frame(i, 6.95F, 0.8F, 3.4F);
        sample.cpu.phase_ms[static_cast<std::size_t>(CpuPhase::slot_wait)] = 0.15F;
        sample.cpu.phase_ms[static_cast<std::size_t>(CpuPhase::acquire)] = 6.0F;
        samples.push_back(sample);
    }

    samples.push_back(frame(8, 10.5F, 0.4F, 10.4F));

    auto busy_gpu = frame(9, 10.0F, 0.4F, 9.5F);
    busy_gpu.cpu.phase_ms[static_cast<std::size_t>(CpuPhase::slot_wait)] = 0.0F;
    busy_gpu.cpu.phase_ms[static_cast<std::size_t>(CpuPhase::acquire)] = 9.6F;
    samples.push_back(busy_gpu);

    auto const analysis = analyse_frames(samples, AnalysisOptions{.target_hz = 144.0F});
    CHECK(analysis.bound.presentation_bound == 8);
    CHECK(analysis.bound.gpu_bound == 2);
    CHECK(analysis.bound.cpu_bound == 0);
    CHECK(analysis.bound.presentation_bound_fraction() == doctest::Approx(0.8F));

    CHECK(analysis.budget.displayed_over_budget == 2);

    auto const strict = analyse_frames(samples, AnalysisOptions{.target_hz = 144.0F, .budget_tolerance = 0.0F});
    CHECK(strict.budget.displayed_over_budget == 10);
}
