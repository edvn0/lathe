#include "benchmark/benchmark_cvars.hxx"

#include <limits>

#include "benchmark/benchmark.hxx"
#include "benchmark/benchmark_compare.hxx"
#include "core/command_line.hxx"

namespace {

    BenchmarkOptions const defaults;
    BenchmarkCompareOptions const compare_defaults;

    auto check_render_size(std::string_view text) -> std::expected<void, std::string> {
        if (auto const size = parse_render_size(text); !size) {
            return std::unexpected(size.error());
        }

        return {};
    }

    auto check_scenarios(std::string_view text) -> std::expected<void, std::string> {
        for (auto const name: CommandLine::split(text, ',')) {
            if (name.empty()) {
                return std::unexpected(std::string{"empty scenario name"});
            }
        }

        return {};
    }

    auto check_sweep(std::string_view spec) -> std::expected<void, std::string> {
        if (auto const sweep = parse_benchmark_sweep(spec); !sweep) {
            return std::unexpected(sweep.error());
        }

        return {};
    }

    auto check_compare(std::string_view text) -> std::expected<void, std::string> {
        auto const comma = text.find(',');

        if (comma == std::string_view::npos || comma == 0 || comma + 1 == text.size()) {
            return std::unexpected(std::string{"needs <base.json>,<head.json>"});
        }

        return {};
    }

}

namespace cvars {

    CVar<std::filesystem::path> benchmark{{
            .name = "benchmark",
            .group = "Benchmark",
            .help = "Run the game's scene along its benchmark camera path and write the JSON",
            .hint = "OUT.json",
            .cli_only = true,
    }};
    CVar<std::filesystem::path> benchmark_suite{{
            .name = "benchmark_suite",
            .group = "Benchmark",
            .help = "Run every scenario at each of its load levels, writing suite.json and report.md",
            .hint = "DIR",
            .cli_only = true,
    }};
    CVar<std::uint32_t> benchmark_frames{{
            .name = "benchmark_frames",
            .group = "Benchmark",
            .help = "Measured frames per run, one lap of the camera path (600)",
            .hint = "N",
            .default_value = defaults.frame_count,
            .min = 1,
    }};
    CVar<std::uint32_t> benchmark_warmup{{
            .name = "benchmark_warmup",
            .group = "Benchmark",
            .help = "Minimum frames at the first keyframe before measuring (60)",
            .hint = "N",
            .default_value = defaults.warmup_frame_count,
    }};
    CVar<std::uint32_t> benchmark_max_warmup{{
            .name = "benchmark_max_warmup",
            .group = "Benchmark",
            .help = "Measure anyway after this many frames, even if streaming hasn't settled (1200)",
            .hint = "N",
            .default_value = defaults.max_warmup_frame_count,
    }};
    CVar<std::uint32_t> seed{{
            .name = "seed",
            .group = "Benchmark",
            .help = "Scene seed (1337)",
            .hint = "N",
            .default_value = defaults.seed,
    }};
    CVar<float> benchmark_target_hz{{
            .name = "benchmark_target_hz",
            .group = "Benchmark",
            .help = "Refresh rate frames are judged against (144)",
            .hint = "HZ",
            .default_value = defaults.target_hz,
            .min = std::numeric_limits<float>::min(),
    }};
    CVar<std::string> benchmark_render_size{{
            .name = "benchmark_render_size",
            .group = "Benchmark",
            .help = "Fixed render resolution (suite default 1920x1080; single run: the Viewport panel)",
            .hint = "WxH",
            .validator = check_render_size,
    }};
    CVar<bool> benchmark_screenshots{{
            .name = "benchmark_screenshots",
            .group = "Benchmark",
            .help = "Screenshot the first frame at or past each keyframe of every run (slows the frames)",
    }};
    CVar<bool> benchmark_frame_checksum{{
            .name = "benchmark_frame_checksum",
            .group = "Benchmark",
            .help = "Hash the data prepare_frame writes for every frame into the results, to check that a change "
                    "leaves it exactly as it was (slow: the timings are not comparable)",
    }};
    CVar<std::string> benchmark_scenarios{{
            .name = "benchmark_scenarios",
            .group = "Benchmark",
            .help = "Suite: which scenarios to run (default: all)",
            .hint = "a,b",
            .validator = check_scenarios,
    }};
    CVar<std::vector<std::string>> benchmark_sweep{{
            .name = "benchmark_sweep",
            .group = "Benchmark",
            .help = "Suite: load levels for a scenario, replacing its defaults (repeatable)",
            .hint = "name:1,2,4",
            .validator = check_sweep,
    }};
    CVar<std::uint32_t> benchmark_repeats{{
            .name = "benchmark_repeats",
            .group = "Benchmark",
            .help = "Suite: runs of each case (3)",
            .hint = "N",
            .min = 1,
    }};

    CVar<std::string> benchmark_compare{{
            .name = "benchmark_compare",
            .group = "Benchmark comparison",
            .help = "Compare two benchmark results and print a Markdown report, without starting the renderer",
            .hint = "BASE.json,HEAD.json",
            .validator = check_compare,
            .cli_only = true,
    }};
    CVar<std::filesystem::path> benchmark_report{{
            .name = "benchmark_report",
            .group = "Benchmark comparison",
            .help = "Also write the comparison report to a file",
            .hint = "OUT.md",
            .cli_only = true,
    }};
    CVar<double> benchmark_threshold{{
            .name = "benchmark_threshold",
            .group = "Benchmark comparison",
            .help = "Flag a metric whose median moved more than this (10)",
            .hint = "PERCENT",
            .default_value = compare_defaults.threshold_percent,
            .min = 0.0,
    }};
    CVar<double> benchmark_fail_threshold{{
            .name = "benchmark_fail_threshold",
            .group = "Benchmark comparison",
            .help = "Exit 1 when a headline metric regresses more than this (20)",
            .hint = "PERCENT",
            .default_value = compare_defaults.fail_threshold_percent,
            .min = 0.0,
    }};

}
