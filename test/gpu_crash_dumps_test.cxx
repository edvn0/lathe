#include <doctest/doctest.h>

#include "gpu/gpu_crash_dumps.hxx"

TEST_SUITE("unit") {
    TEST_CASE("gpu_crash_dumps: the mode a setting names") {
        using gpu_crash_dumps::Mode;
        CHECK(gpu_crash_dumps::mode_from("off", false) == Mode::off);
        CHECK(gpu_crash_dumps::mode_from("standard", false) == Mode::standard);
        CHECK(gpu_crash_dumps::mode_from("full", false) == Mode::full);
        CHECK_FALSE(gpu_crash_dumps::mode_from("verbose", false).has_value());
        CHECK_FALSE(gpu_crash_dumps::mode_from("", true).has_value());
    }

    TEST_CASE("gpu_crash_dumps: auto is the cheap mode, and everything after a device-lost restart") {
        using gpu_crash_dumps::Mode;
        CHECK(gpu_crash_dumps::mode_from("auto", false) == Mode::standard);
        CHECK(gpu_crash_dumps::mode_from("auto", true) == Mode::full);
        // An explicit setting is not changed by a restart.
        CHECK(gpu_crash_dumps::mode_from("off", true) == Mode::off);
        CHECK(gpu_crash_dumps::mode_from("standard", true) == Mode::standard);
    }

    TEST_CASE("gpu_crash_dumps: nothing is on before start") {
        CHECK_FALSE(gpu_crash_dumps::active());
        CHECK(gpu_crash_dumps::device_flags() == 0U);
        CHECK_FALSE(gpu_crash_dumps::prefers_spirv());
    }
}
