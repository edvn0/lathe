#include "app/benchmark_compare.hxx"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>

#include "app/benchmark.hxx"
#include "core/json.hxx"

namespace {

    [[nodiscard]] auto parse_percent(std::string_view flag,
                                     std::string_view value) -> std::expected<double, std::string> {
        auto result = 0.0;
        auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
        if (error != std::errc{} || end != value.data() + value.size() || !(result >= 0.0)) {
            return std::unexpected(std::format("{}: '{}' is not a non-negative number", flag, value));
        }
        return result;
    }

    // The headline metrics decide pass/fail; the rest are context.
    constexpr std::array<std::string_view, 4> headline_metrics{
            "displayed_p99_ms",
            "displayed_median_ms",
            "gpu_median_ms",
            "cpu_busy_median_ms",
    };

    [[nodiscard]] auto is_headline(std::string_view name) -> bool {
        return std::ranges::find(headline_metrics, name) != headline_metrics.end();
    }

    [[nodiscard]] auto display_name(std::string_view name) -> std::string {
        if (name == "displayed_median_ms") {
            return "Displayed p50 (ms)";
        }
        if (name == "displayed_p99_ms") {
            return "Displayed p99 (ms)";
        }
        if (name == "displayed_p999_ms") {
            return "Displayed p99.9 (ms)";
        }
        if (name == "gpu_median_ms") {
            return "GPU p50 (ms)";
        }
        if (name == "gpu_p95_ms") {
            return "GPU p95 (ms)";
        }
        if (name == "gpu_p99_ms") {
            return "GPU p99 (ms)";
        }
        if (name == "cpu_busy_median_ms") {
            return "CPU busy p50 (ms)";
        }
        if (name == "cpu_busy_p99_ms") {
            return "CPU busy p99 (ms)";
        }
        if (name == "budget_miss_fraction") {
            return "Over budget (fraction)";
        }
        if (name == "hitches") {
            return "Hitches";
        }
        if (name == "gpu_bound_fraction") {
            return "GPU-bound (fraction)";
        }
        constexpr std::string_view stage_prefix = "stage_";
        constexpr std::string_view stage_suffix = "_median_ms";
        if (name.starts_with(stage_prefix) && name.ends_with(stage_suffix)) {
            auto const id = name.substr(stage_prefix.size(), name.size() - stage_prefix.size() - stage_suffix.size());
            return std::format("pass `{}` p50 (ms)", id);
        }
        return std::string{name};
    }

    struct Metric {
        std::string name;
        double median = 0.0;
        double min = 0.0;
        double max = 0.0;
        std::uint32_t samples = 1;
    };

    struct Case {
        std::string key;
        std::vector<Metric> metrics;

        [[nodiscard]] auto find(std::string_view name) const -> Metric const * {
            auto const found = std::ranges::find(metrics, name, &Metric::name);
            return found == metrics.end() ? nullptr : &*found;
        }
    };

    struct Result {
        std::string kind;
        std::string device;
        std::string render_extent;
        std::string present_mode;
        std::string build_type;
        std::string revision;
        std::string frames;
        std::string seed;
        std::vector<Case> cases;

        [[nodiscard]] auto find(std::string_view key) const -> Case const * {
            auto const found = std::ranges::find(cases, key, &Case::key);
            return found == cases.end() ? nullptr : &*found;
        }
    };

    [[nodiscard]] auto number_text(JsonValue const &value) -> std::string {
        return value.is_number() ? std::format("{}", static_cast<std::int64_t>(value.as_number())) : std::string{};
    }

    [[nodiscard]] auto extent_text(JsonValue const &value) -> std::string {
        auto const items = value.items();
        if (items.size() != 2) {
            return {};
        }
        return std::format("{}x{}", static_cast<std::int64_t>(items[0].as_number()),
                           static_cast<std::int64_t>(items[1].as_number()));
    }

    auto add_single_metric(Case &target, std::string name, JsonValue const &value) -> void {
        if (!value.is_number()) {
            return;
        }
        auto const number = value.as_number();
        target.metrics.push_back(Metric{.name = std::move(name), .median = number, .min = number, .max = number});
    }

    [[nodiscard]] auto normalise(JsonValue const &root) -> Result {
        Result result;
        auto const &environment = root["environment"];

        result.present_mode = environment["present_mode"].as_string();
        result.build_type = environment["build_type"].as_string();
        result.revision = environment["source_revision"].as_string();

        if (root["kind"].as_string() == "suite") {
            result.kind = "suite";
            result.device = environment["device"].as_string();
            result.render_extent = extent_text(environment["render_extent"]);
            result.frames = number_text(root["options"]["frames"]);
            result.seed = number_text(root["options"]["seed"]);

            for (auto const &aggregate: root["aggregates"].items()) {
                Case entry{.key = aggregate["key"].as_string()};
                for (auto const &member: aggregate["metrics"].members()) {
                    auto const &stats = member.value;
                    entry.metrics.push_back(Metric{
                            .name = member.key,
                            .median = stats["median"].as_number(),
                            .min = stats["min"].as_number(),
                            .max = stats["max"].as_number(),
                            .samples = static_cast<std::uint32_t>(stats["values"].items().size()),
                    });
                }
                result.cases.push_back(std::move(entry));
            }
            return result;
        }

        // A single run: schema 1 has only GPU stages, schema 2 adds "analysis".
        result.kind = "single";
        result.device = root["device"].as_string();
        result.render_extent = extent_text(root["render_extent"]);
        result.frames = number_text(root["frames"]);
        result.seed = number_text(root["seed"]);

        Case entry{.key = "game"};
        auto const &analysis = root["analysis"];
        add_single_metric(entry, "displayed_median_ms", analysis["displayed"]["median_ms"]);
        add_single_metric(entry, "displayed_p99_ms", analysis["displayed"]["p99_ms"]);
        add_single_metric(entry, "displayed_p999_ms", analysis["displayed"]["p999_ms"]);
        add_single_metric(entry, "cpu_busy_median_ms", analysis["cpu_busy"]["median_ms"]);
        add_single_metric(entry, "cpu_busy_p99_ms", analysis["cpu_busy"]["p99_ms"]);

        for (auto const &stage: root["stages"].items()) {
            auto const id = stage["id"].as_string();
            if (id == "full_frame") {
                add_single_metric(entry, "gpu_median_ms", stage["median_ms"]);
                add_single_metric(entry, "gpu_p95_ms", stage["p95_ms"]);
            } else {
                add_single_metric(entry, std::format("stage_{}_median_ms", id), stage["median_ms"]);
            }
        }
        add_single_metric(entry, "hitches", analysis["hitch_count"]);

        result.cases.push_back(std::move(entry));
        return result;
    }

    [[nodiscard]] auto format_value(Metric const *metric) -> std::string {
        if (metric == nullptr) {
            return "--";
        }
        auto const precision = metric->name.ends_with("_ms") ? 3 : 2;
        if (metric->samples < 2) {
            return std::format("{:.{}f}", metric->median, precision);
        }
        return std::format("{:.{}f} ({:.{}f}-{:.{}f})", metric->median, precision, metric->min, precision, metric->max,
                           precision);
    }

    [[nodiscard]] auto compare_metric(std::string const &name, Metric const *base, Metric const *head,
                                      BenchmarkCompareOptions const &options) -> MetricComparison {
        MetricComparison comparison{.name = name, .headline = is_headline(name)};

        if (base != nullptr) {
            comparison.base = base->median;
            comparison.base_samples = base->samples;
        }
        if (head != nullptr) {
            comparison.head = head->median;
            comparison.head_samples = head->samples;
        }
        if (base == nullptr || head == nullptr || base->median <= 0.0) {
            return comparison;
        }

        auto const change = (head->median - base->median) / base->median * 100.0;
        comparison.change_percent = change;

        // Only timings are judged; counts and fractions are shown for context.
        if (!name.ends_with("_ms")) {
            return comparison;
        }

        auto const big_enough = std::max(base->median, head->median) >= options.minimum_flagged_ms;
        if (std::abs(change) <= options.threshold_percent || !big_enough) {
            return comparison;
        }

        if (base->samples >= 2 && head->samples >= 2) {
            auto const separated = change > 0.0 ? head->min > base->max : head->max < base->min;
            if (!separated) {
                comparison.verdict = CompareVerdict::within_noise;
                return comparison;
            }
        }

        comparison.verdict = change > 0.0 ? CompareVerdict::regressed : CompareVerdict::improved;
        return comparison;
    }

    [[nodiscard]] auto verdict_text(MetricComparison const &metric) -> std::string_view {
        switch (metric.verdict) {
            case CompareVerdict::regressed:
                return ":red_circle: slower";
            case CompareVerdict::improved:
                return ":green_circle: faster";
            case CompareVerdict::within_noise:
                return "within noise";
            case CompareVerdict::unchanged:
                break;
        }
        return "";
    }

    [[nodiscard]] auto describe(std::string_view label, Result const &result) -> std::string {
        return std::format("- **{}**: {} ({}), {}, present `{}`, {} build at `{}`, {} frames, seed {}", label,
                           result.device.empty() ? "unknown device" : result.device, result.kind,
                           result.render_extent.empty() ? "?" : result.render_extent,
                           result.present_mode.empty() ? "?" : result.present_mode,
                           result.build_type.empty() ? "unknown" : result.build_type,
                           result.revision.empty() ? "?" : result.revision, result.frames, result.seed);
    }

} // namespace

auto parse_benchmark_compare_options(std::span<char const *const> args)
        -> std::expected<std::optional<BenchmarkCompareOptions>, std::string> {
    BenchmarkCompareOptions options;
    bool enabled = false;

    for (auto const *raw: args) {
        std::string_view const arg = raw;

        if (constexpr std::string_view prefix = "--benchmark-compare="; arg.starts_with(prefix)) {
            auto const value = arg.substr(prefix.size());
            auto const comma = value.find(',');
            if (comma == std::string_view::npos || comma == 0 || comma + 1 == value.size()) {
                return std::unexpected(std::string{"--benchmark-compare= needs <base.json>,<head.json>"});
            }
            options.base = std::filesystem::path{value.substr(0, comma)};
            options.head = std::filesystem::path{value.substr(comma + 1)};
            enabled = true;
        } else if (constexpr std::string_view report_prefix = "--benchmark-report="; arg.starts_with(report_prefix)) {
            options.report = std::filesystem::path{arg.substr(report_prefix.size())};
        } else if (constexpr std::string_view threshold_prefix = "--benchmark-threshold=";
                   arg.starts_with(threshold_prefix)) {
            auto const value = parse_percent("--benchmark-threshold", arg.substr(threshold_prefix.size()));
            if (!value) {
                return std::unexpected(value.error());
            }
            options.threshold_percent = *value;
        } else if (constexpr std::string_view fail_prefix = "--benchmark-fail-threshold=";
                   arg.starts_with(fail_prefix)) {
            auto const value = parse_percent("--benchmark-fail-threshold", arg.substr(fail_prefix.size()));
            if (!value) {
                return std::unexpected(value.error());
            }
            options.fail_threshold_percent = *value;
        }
    }

    if (!enabled) {
        return std::nullopt;
    }
    return options;
}

auto compare_benchmark_results(JsonValue const &base_root, JsonValue const &head_root,
                               BenchmarkCompareOptions const &options) -> BenchmarkComparison {
    BenchmarkComparison comparison;

    auto const base = normalise(base_root);
    auto const head = normalise(head_root);

    auto const mismatch = [&](std::string_view what, std::string const &a, std::string const &b) {
        if (!a.empty() && !b.empty() && a != b) {
            comparison.notes.push_back(
                    std::format("`{}` differs ({} vs {}): the numbers may not be comparable.", what, a, b));
        }
    };
    mismatch("device", base.device, head.device);
    mismatch("render extent", base.render_extent, head.render_extent);
    mismatch("present mode", base.present_mode, head.present_mode);
    mismatch("build type", base.build_type, head.build_type);
    mismatch("frames", base.frames, head.frames);
    mismatch("seed", base.seed, head.seed);
    if (base.kind != head.kind) {
        comparison.notes.push_back(std::format("Comparing a {} result with a {} result: only shared metrics compare.",
                                               base.kind, head.kind));
    }
    if (head.device.find("llvmpipe") != std::string::npos) {
        comparison.notes.emplace_back("Head ran on a software rasterizer (lavapipe): don't read much into the timings.");
    }

    // Cases in head order, then those only in base.
    std::vector<std::string> keys;
    keys.reserve(head.cases.size() + base.cases.size());
    for (auto const &entry: head.cases) {
        keys.push_back(entry.key);
    }
    for (auto const &entry: base.cases) {
        if (std::ranges::find(keys, entry.key) == keys.end()) {
            keys.push_back(entry.key);
        }
    }

    for (auto const &key: keys) {
        auto const *base_case = base.find(key);
        auto const *head_case = head.find(key);
        CaseComparison case_comparison{.key = key};

        std::vector<std::string> names;
        for (auto const *source: {head_case, base_case}) {
            if (source == nullptr) {
                continue;
            }
            for (auto const &metric: source->metrics) {
                if (std::ranges::find(names, metric.name) == names.end()) {
                    names.push_back(metric.name);
                }
            }
        }

        for (auto const &name: names) {
            auto metric = compare_metric(name, base_case != nullptr ? base_case->find(name) : nullptr,
                                         head_case != nullptr ? head_case->find(name) : nullptr, options);

            if (metric.headline && metric.verdict == CompareVerdict::regressed && metric.change_percent &&
                *metric.change_percent > options.fail_threshold_percent) {
                comparison.failed = true;
            }
            case_comparison.metrics.push_back(std::move(metric));
        }

        comparison.cases.push_back(std::move(case_comparison));
    }

    // ---- Markdown
    auto &out = comparison.markdown;
    out += "### Benchmark: base vs head\n\n";
    out += describe("base", base) + "\n";
    out += describe("head", head) + "\n";
    for (auto const &note: comparison.notes) {
        out += std::format("- :warning: {}\n", note);
    }

    auto const repeats = base.kind == "suite" && head.kind == "suite";

    for (auto const &case_comparison: comparison.cases) {
        auto const *base_case = base.find(case_comparison.key);
        auto const *head_case = head.find(case_comparison.key);

        out += std::format("\n#### {}{}\n\n", case_comparison.key,
                           base_case == nullptr   ? " (new)"
                           : head_case == nullptr ? " (removed)"
                                                  : "");
        out += "| Metric | Base | Head | Change | |\n|---|---:|---:|---:|---|\n";

        std::size_t quiet = 0;
        for (auto const &metric: case_comparison.metrics) {
            // In suites, list headline metrics, anything flagged and anything that only one side has; count the rest.
            auto const interesting = !repeats || metric.headline || metric.verdict != CompareVerdict::unchanged ||
                                     !metric.base || !metric.head;
            if (!interesting) {
                ++quiet;
                continue;
            }

            auto const name =
                    metric.headline ? std::format("**{}**", display_name(metric.name)) : display_name(metric.name);
            auto const change = metric.change_percent ? std::format("{:+.1f}%", *metric.change_percent)
                                                      : std::string{metric.base ? "removed" : "new"};
            out += std::format("| {} | {} | {} | {} | {} |\n", name,
                               format_value(base_case != nullptr ? base_case->find(metric.name) : nullptr),
                               format_value(head_case != nullptr ? head_case->find(metric.name) : nullptr), change,
                               verdict_text(metric));
        }
        if (quiet != 0) {
            out += std::format("\n{} other metric(s) moved less than {:.0f}%.\n", quiet, options.threshold_percent);
        }
    }

    out += std::format("\nFlagged: a timing whose median moved more than {:.0f}% (timings under {} ms never are)",
                       options.threshold_percent, options.minimum_flagged_ms);
    out += repeats ? "; with repeats on both sides, only when the ranges across repeats don't overlap. Ranges are "
                     "shown in brackets.\n"
                   : ". Single runs have no spread, so run a suite with repeats before trusting a small change.\n";
    out += comparison.failed ? std::format("\n**Result: a headline metric regressed by more than {:.0f}%.**\n",
                                           options.fail_threshold_percent)
                             : std::format("\nResult: no headline metric regressed by more than {:.0f}%.\n",
                                           options.fail_threshold_percent);

    return comparison;
}

namespace {

    [[nodiscard]] auto load_json(std::filesystem::path const &path) -> std::expected<JsonValue, std::string> {
        std::ifstream file{path, std::ios::binary};
        if (!file) {
            return std::unexpected(std::format("could not open {}", path.string()));
        }
        std::string text{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};

        auto parsed = parse_json(text);
        if (!parsed) {
            return std::unexpected(
                    std::format("{}: {} at byte {}", path.string(), parsed.error().message, parsed.error().offset));
        }
        return std::move(*parsed);
    }

} // namespace

auto run_benchmark_compare(BenchmarkCompareOptions const &options) -> int {
    auto const base = load_json(options.base);
    auto const head = load_json(options.head);

    if (!base || !head) {
        auto const message = !base ? base.error() : head.error();
        std::fprintf(stderr, "benchmark compare: %s\n", message.c_str()); // NOLINT(modernize-use-std-print)
        return 2;
    }

    auto const comparison = compare_benchmark_results(*base, *head, options);
    std::fwrite(comparison.markdown.data(), 1, comparison.markdown.size(), stdout);

    if (options.report) {
        if (auto const written = write_text_file(*options.report, comparison.markdown); !written) {
            std::fprintf(stderr, "benchmark compare: %s\n", written.error().c_str()); // NOLINT(modernize-use-std-print)
            return 2;
        }
    }

    return comparison.failed ? 1 : 0;
}
