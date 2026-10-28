#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/cvar.hxx"

// The benchmark runner's settings (benchmark_options() turns them into BenchmarkOptions) and the comparison tool's
// (benchmark_compare_options()).
namespace cvars {

    extern CVar<std::filesystem::path> benchmark;
    extern CVar<std::filesystem::path> benchmark_suite;
    extern CVar<std::uint32_t> benchmark_frames;
    extern CVar<std::uint32_t> benchmark_warmup;
    extern CVar<std::uint32_t> benchmark_max_warmup;
    extern CVar<std::uint32_t> seed;
    extern CVar<float> benchmark_target_hz;
    extern CVar<std::string> benchmark_render_size;
    extern CVar<bool> benchmark_screenshots;
    extern CVar<bool> benchmark_frame_checksum;
    extern CVar<std::string> benchmark_scenarios;
    extern CVar<std::vector<std::string>> benchmark_sweep;
    extern CVar<std::uint32_t> benchmark_repeats;

    extern CVar<std::string> benchmark_compare;
    extern CVar<std::filesystem::path> benchmark_report;
    extern CVar<double> benchmark_threshold;
    extern CVar<double> benchmark_fail_threshold;

}
