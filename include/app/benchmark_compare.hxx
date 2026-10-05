#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

class CommandLine;
class JsonValue;

// --benchmark-compare=<base.json>,<head.json>: compares two benchmark results (single runs, schema 1 or 2, or suites)
// and prints a Markdown report, without starting the renderer.
//
//   --benchmark-report=<out.md>         also write the report to a file
//   --benchmark-threshold=<percent>     flag a metric whose median moved more than this (10)
//   --benchmark-fail-threshold=<pct>    exit 1 when a headline metric regresses more than this (20)
//
// With repeats on both sides (a suite's aggregates), a change only counts as a regression when the two builds'
// ranges across repeats don't overlap; otherwise it is reported as within noise. Single runs have no spread, so
// their changes are judged on the thresholds alone.
struct BenchmarkCompareOptions {
    std::filesystem::path base;
    std::filesystem::path head;
    std::optional<std::filesystem::path> report;
    double threshold_percent = 10.0;
    double fail_threshold_percent = 20.0;

    // Timings below this are reported but never flagged: a 0.1 ms pass moving 30% is noise.
    double minimum_flagged_ms = 0.5;
};

// The --benchmark-compare options. Registers on construction; call options() after the CommandLine has parsed. Not
// movable: the CommandLine holds references to the members.
class BenchmarkCompareArguments {
public:
    explicit BenchmarkCompareArguments(CommandLine &cli);

    BenchmarkCompareArguments(BenchmarkCompareArguments const &) = delete;
    auto operator=(BenchmarkCompareArguments const &) -> BenchmarkCompareArguments & = delete;

    // nullopt without --benchmark-compare.
    [[nodiscard]]
    auto options() const -> std::optional<BenchmarkCompareOptions>;

private:
    BenchmarkCompareOptions options_;
    std::optional<std::filesystem::path> report_;
    bool enabled_ = false;
};

enum class CompareVerdict : std::uint8_t {
    unchanged,
    within_noise, // moved past the threshold, but the repeat ranges overlap
    improved,
    regressed,
    not_judged, // a displayed time of a presentation-paced case: it measures the swapchain, not the engine
};

struct MetricComparison {
    std::string name;
    std::optional<double> base;
    std::optional<double> head;
    std::uint32_t base_samples = 0;
    std::uint32_t head_samples = 0;
    std::optional<double> change_percent;
    CompareVerdict verdict = CompareVerdict::unchanged;
    bool headline = false;
};

struct CaseComparison {
    std::string key;
    std::vector<MetricComparison> metrics;
};

struct BenchmarkComparison {
    std::vector<std::string> notes; // environment mismatches and the like
    std::vector<CaseComparison> cases;
    std::string markdown;

    // A headline metric regressed past the fail threshold (and outside the noise, when there are repeats).
    bool failed = false;
};

[[nodiscard]]
auto compare_benchmark_results(JsonValue const &base, JsonValue const &head,
                               BenchmarkCompareOptions const &options) -> BenchmarkComparison;

// Reads both files, prints the report to stdout (and --benchmark-report=), returns the process exit code.
[[nodiscard]]
auto run_benchmark_compare(BenchmarkCompareOptions const &options) -> int;
