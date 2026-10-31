#include <doctest/doctest.h>

#include "core/gpu_timings.hxx"

TEST_SUITE("unit") {
    TEST_CASE("first sample seeds the average") {
        GpuTimings timings;
        timings.begin_frame();
        timings.record("gtao", "GTAO", 0, 2.0F);

        auto const *entry = timings.find("gtao");
        REQUIRE(entry != nullptr);
        CHECK(entry->active);
        CHECK(entry->ema_ms == doctest::Approx(2.0F));
        CHECK(entry->last_ms == doctest::Approx(2.0F));
    }

    TEST_CASE("average moves toward new samples by the smoothing factor") {
        GpuTimings timings;
        timings.begin_frame();
        timings.record("p", "P", 0, 1.0F);
        timings.begin_frame();
        timings.record("p", "P", 0, 2.0F);

        CHECK(timings.find("p")->ema_ms == doctest::Approx(1.0F + GpuTimings::smoothing));
    }

    TEST_CASE("a pass that goes idle is inactive and restarts its average on return") {
        GpuTimings timings;
        timings.begin_frame();
        timings.record("p", "P", 0, 1.0F);
        timings.begin_frame();
        CHECK_FALSE(timings.find("p")->active);

        timings.begin_frame();
        timings.record("p", "P", 0, 5.0F);
        CHECK(timings.find("p")->ema_ms == doctest::Approx(5.0F));
    }

    TEST_CASE("a pass without a measurement stays idle") {
        GpuTimings timings;
        timings.begin_frame();
        timings.record("p", "P", 0, std::nullopt);
        CHECK_FALSE(timings.find("p")->active);
        CHECK(timings.total_ms() == doctest::Approx(0.0F));
    }

    TEST_CASE("totals sum active passes per lane") {
        GpuTimings timings;
        timings.begin_frame();
        timings.record("a", "A", 0, 1.0F);
        timings.record("b", "B", 0, 2.0F);
        timings.record("c", "C", 1, 4.0F);

        CHECK(timings.lane_total_ms(0) == doctest::Approx(3.0F));
        CHECK(timings.lane_total_ms(1) == doctest::Approx(4.0F));
        CHECK(timings.total_ms() == doctest::Approx(7.0F));
    }

    TEST_CASE("history keeps the newest samples in order and reports min and max") {
        GpuTimings timings;
        for (auto frame = 0U; frame < GpuTimings::history_length + 10; ++frame) {
            timings.begin_frame();
            timings.record("p", "P", 0, static_cast<float>(frame));
        }

        auto const *entry = timings.find("p");
        REQUIRE(entry->samples() == GpuTimings::history_length);
        CHECK(entry->sample(0) == doctest::Approx(10.0F));
        CHECK(entry->sample(GpuTimings::history_length - 1) == doctest::Approx(GpuTimings::history_length + 9.0F));
        CHECK(entry->min_ms() == doctest::Approx(10.0F));
        CHECK(entry->max_ms() == doctest::Approx(GpuTimings::history_length + 9.0F));
    }
}
