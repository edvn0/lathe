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

struct BenchmarkCompareOptions {
    std::filesystem::path base;
    std::filesystem::path head;
    std::optional<std::filesystem::path> report;
    double threshold_percent = 10.0;
    double fail_threshold_percent = 20.0;

    double minimum_flagged_ms = 0.5;
};

class BenchmarkCompareArguments {
public:
    explicit BenchmarkCompareArguments(CommandLine &cli);

    BenchmarkCompareArguments(BenchmarkCompareArguments const &) = delete;
    auto operator=(BenchmarkCompareArguments const &) -> BenchmarkCompareArguments & = delete;

    [[nodiscard]]
    auto options() const -> std::optional<BenchmarkCompareOptions>;

private:
    BenchmarkCompareOptions options_;
    std::optional<std::filesystem::path> report_;
    bool enabled_ = false;
};

enum class CompareVerdict : std::uint8_t {
    unchanged,
    within_noise,
    improved,
    regressed,
    not_judged,
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
    std::vector<std::string> notes;
    std::vector<CaseComparison> cases;
    std::string markdown;

    bool failed = false;
};

[[nodiscard]]
auto compare_benchmark_results(JsonValue const &base, JsonValue const &head,
                               BenchmarkCompareOptions const &options) -> BenchmarkComparison;

[[nodiscard]]
auto run_benchmark_compare(BenchmarkCompareOptions const &options) -> int;
