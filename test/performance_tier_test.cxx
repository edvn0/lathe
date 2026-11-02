#include <doctest/doctest.h>

#include "gpu/performance_tier.hxx"

TEST_SUITE("unit") {
    constexpr auto gib = std::uint64_t{1024} * 1024 * 1024;

    TEST_CASE("performance tier: a discrete GPU is high with enough memory and medium without") {
        CHECK(performance_tier::detect(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 8 * gib) == PerformanceTier::high);
        CHECK(performance_tier::detect(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 4 * gib) == PerformanceTier::medium);
        CHECK(performance_tier::detect(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU, 6 * gib) == PerformanceTier::high);
    }

    TEST_CASE("performance tier: integrated, CPU and unknown devices are low whatever their memory") {
        CHECK(performance_tier::detect(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 32 * gib) == PerformanceTier::low);
        CHECK(performance_tier::detect(VK_PHYSICAL_DEVICE_TYPE_CPU, 32 * gib) == PerformanceTier::low);
        CHECK(performance_tier::detect(VK_PHYSICAL_DEVICE_TYPE_OTHER, 32 * gib) == PerformanceTier::low);
    }

    TEST_CASE("performance tier: lower tiers trace shadows at a lower resolution and update the far cascade less") {
        CHECK(performance_tier::profile(PerformanceTier::high).rt_shadow_resolution_shift == 0);
        CHECK(performance_tier::profile(PerformanceTier::medium).rt_shadow_resolution_shift == 1);
        CHECK(performance_tier::profile(PerformanceTier::low).rt_shadow_resolution_shift == 2);
        CHECK(performance_tier::profile(PerformanceTier::low).far_cascade_update_interval >
              performance_tier::profile(PerformanceTier::high).far_cascade_update_interval);
    }

    TEST_CASE("performance tier: names parse back and auto is not a tier") {
        for (auto const tier: {PerformanceTier::low, PerformanceTier::medium, PerformanceTier::high}) {
            CHECK(performance_tier::parse(performance_tier::name(tier)) == tier);
        }
        CHECK_FALSE(performance_tier::parse("auto").has_value());
        CHECK_FALSE(performance_tier::parse("ultra").has_value());
    }
}
