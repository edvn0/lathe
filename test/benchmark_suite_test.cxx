#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <expected>
#include <format>
#include <optional>
#include <string>
#include <vector>

#include "app/benchmark.hxx"
#include "app/benchmark_compare.hxx"
#include "core/json.hxx"
#include "rendering/renderer.hxx"

namespace {

    [[nodiscard]] auto line_path() -> std::vector<CameraKeyframe> {
        return {
                {.position = {0.0F, 2.0F, 0.0F}, .target = {0.0F, 0.0F, -1.0F}},
                {.position = {10.0F, 2.0F, 0.0F}, .target = {1.0F, 0.0F, 0.0F}},
        };
    }

    // GPU timings whose full frame time equals their serial, so a test can tell which frame they landed on.
    [[nodiscard]] auto gpu_of(std::uint64_t serial) -> FrameTimings {
        FrameTimings timings{.full_frame_ms = static_cast<float>(serial), .valid = true, .frame_serial = serial};
        timings.passes.push_back(
                frame_graph::PassTiming{.name_id = "forward_pass", .label = "Forward", .milliseconds = 1.0F});
        return timings;
    }

    [[nodiscard]] auto
    parse_args(std::vector<char const *> const &args) -> std::expected<std::optional<BenchmarkOptions>, std::string> {
        return parse_benchmark_options(args);
    }

    [[nodiscard]] auto case_result(std::string scenario, std::string axis, std::uint32_t load, std::uint32_t repeat,
                                   float displayed_ms, float gpu_ms, float busy_ms) -> BenchmarkCaseResult {
        BenchmarkCaseResult result{
                .id = BenchmarkCaseId{
                        .scenario = std::move(scenario), .load_axis = std::move(axis), .load = load, .repeat = repeat}};
        result.analysis.frames = 100;
        result.analysis.displayed.median_ms = displayed_ms;
        result.analysis.displayed.p99_ms = displayed_ms * 1.5F;
        result.analysis.gpu_frame.median_ms = gpu_ms;
        result.analysis.cpu_busy.median_ms = busy_ms;
        result.analysis.budget.frames = 100;
        result.stages.push_back(BenchmarkStage{.id = "full_frame", .name = "Full Frame"});
        result.stages.push_back(
                BenchmarkStage{.id = "forward_pass", .name = "Forward", .summary = TimingSummary{.median_ms = gpu_ms}});
        return result;
    }

} // namespace

TEST_CASE("suite options: defaults, sweeps, scenarios and errors") {
    SUBCASE("suite mode gets repeats and a fixed render size by default") {
        auto const options = parse_args({"--benchmark-suite=perf/run"});
        REQUIRE(options.has_value());
        REQUIRE(options->has_value());
        CHECK((*options)->mode == BenchmarkMode::suite);
        CHECK((*options)->output_path == "perf/run");
        CHECK((*options)->repeats == benchmark_default_suite_repeats);
        CHECK((*options)->render_size == benchmark_default_suite_render_size);
    }

    SUBCASE("single mode keeps one run and the panel's size") {
        auto const options = parse_args({"--benchmark=a.json"});
        REQUIRE(options.has_value());
        REQUIRE(options->has_value());
        CHECK((*options)->mode == BenchmarkMode::single);
        CHECK((*options)->repeats == 1);
        CHECK_FALSE((*options)->render_size.has_value());
    }

    SUBCASE("explicit flags") {
        auto const options = parse_args({"--benchmark-suite=out", "--benchmark-repeats=5", "--benchmark-target-hz=240",
                                         "--benchmark-render-size=2560x1440", "--benchmark-scenarios=lights,draw_calls",
                                         "--benchmark-sweep=lights:16,32", "--benchmark-sweep=lights:8"});
        REQUIRE(options.has_value());
        REQUIRE(options->has_value());
        auto const &value = **options;
        CHECK(value.repeats == 5);
        CHECK(value.target_hz == 240.0F);
        CHECK(value.render_size == BenchmarkRenderSize{.width = 2560, .height = 1440});
        CHECK(value.scenarios == std::vector<std::string>{"lights", "draw_calls"});
        REQUIRE(value.sweeps.size() == 1); // the later sweep replaces the earlier
        CHECK(value.sweeps[0].loads == std::vector<std::uint32_t>{8});
    }

    SUBCASE("errors") {
        CHECK_FALSE(parse_args({"--benchmark=a.json", "--benchmark-suite=b"}).has_value());
        CHECK_FALSE(parse_args({"--benchmark-suite=b", "--benchmark-repeats=0"}).has_value());
        CHECK_FALSE(parse_args({"--benchmark-suite=b", "--benchmark-target-hz=-1"}).has_value());
        CHECK_FALSE(parse_args({"--benchmark-suite=b", "--benchmark-render-size=1920"}).has_value());
        CHECK_FALSE(parse_args({"--benchmark-suite=b", "--benchmark-sweep=lights"}).has_value());
        CHECK_FALSE(parse_args({"--benchmark-suite=b", "--benchmark-sweep=lights:0"}).has_value());
    }

    SUBCASE("present mode and swapchain images") {
        std::array<char const *, 2> args{"--present-mode=immediate", "--swapchain-images=4"};
        CHECK(parse_present_mode_option(args).value() == PresentModeChoice::immediate);
        CHECK(parse_swapchain_images_option(args).value() == 4U);

        std::array<char const *, 1> bad_mode{"--present-mode=tearing"};
        std::array<char const *, 1> too_many{"--swapchain-images=9"};
        std::array<char const *, 1> none{"--seed=1"};
        CHECK_FALSE(parse_present_mode_option(bad_mode).has_value());
        CHECK_FALSE(parse_swapchain_images_option(too_many).has_value());
        CHECK_FALSE(parse_present_mode_option(none).value().has_value());
    }

    SUBCASE("vsync") {
        std::array<char const *, 1> on{"--vsync=on"};
        std::array<char const *, 1> bad{"--vsync=maybe"};
        std::array<char const *, 1> none{"--seed=1"};
        CHECK(parse_vsync_option(on).value() == true);
        CHECK_FALSE(parse_vsync_option(bad).has_value());
        CHECK_FALSE(parse_vsync_option(none).value().has_value());
    }
}

TEST_CASE("the suite plan runs every case once per repeat, repeat-major") {
    std::vector<BenchmarkScenarioInfo> const scenarios{
            {.name = "game"},
            {.name = "lights", .load_axis = "point_lights", .default_loads = {64, 512}},
            {.name = "overdraw", .load_axis = "layers", .default_loads = {4}},
    };

    BenchmarkOptions options{.mode = BenchmarkMode::suite, .repeats = 2};
    auto const plan = plan_benchmark_cases(scenarios, options);
    REQUIRE(plan.has_value());
    REQUIRE(plan->size() == 8);

    std::vector<std::string> order;
    for (auto const &entry: *plan) {
        order.push_back(std::format("{}#{}", entry.id.key(), entry.id.repeat));
    }
    CHECK(order == std::vector<std::string>{"game#0", "lights@point_lights=64#0", "lights@point_lights=512#0",
                                            "overdraw@layers=4#0", "game#1", "lights@point_lights=64#1",
                                            "lights@point_lights=512#1", "overdraw@layers=4#1"});
    CHECK((*plan)[2].id.file_stem() == "lights_point_lights_512_r0");

    SUBCASE("filtered and swept") {
        options.repeats = 1;
        options.scenarios = {"lights"};
        options.sweeps = {BenchmarkSweep{.scenario = "lights", .loads = {1, 2, 3}}};
        auto const swept = plan_benchmark_cases(scenarios, options);
        REQUIRE(swept.has_value());
        CHECK(swept->size() == 3);
        CHECK(swept->back().id.load == 3);
    }

    SUBCASE("unknown scenarios and sweeps without an axis are errors") {
        options.scenarios = {"nope"};
        CHECK_FALSE(plan_benchmark_cases(scenarios, options).has_value());

        options.scenarios.clear();
        options.sweeps = {BenchmarkSweep{.scenario = "game", .loads = {1}}};
        CHECK_FALSE(plan_benchmark_cases(scenarios, options).has_value());
    }
}

TEST_CASE("GPU timings land on the frame they were recorded for, however late they arrive") {
    BenchmarkOptions options;
    options.frame_count = 4;
    options.warmup_frame_count = 1;
    options.drain_frame_limit = 8;

    BenchmarkRun run{options, line_path()};

    // Warmup: frame serial 10. Its timings (arriving with serial 12's frame) must not be taken for a measured one's.
    auto const warmup_gpu = FrameTimings{};
    run.on_frame_drawn(BenchmarkFrameInput{.gpu = &warmup_gpu, .frame_serial = 10});
    REQUIRE(run.measuring());

    // Measured frames 11..14, with timings lagging two frames behind.
    for (std::uint64_t serial = 11; serial <= 14; ++serial) {
        auto const gpu = gpu_of(serial - 2);
        CAPTURE(serial);
        run.on_frame_drawn(BenchmarkFrameInput{.gpu = &gpu, .frame_serial = serial, .cpu = {.frame_ms = 5.0F}});
    }
    CHECK_FALSE(run.finished());

    // A stale repeat of the last timings, then the two drain frames that bring in serials 13 and 14.
    auto const stale = gpu_of(12);
    run.on_frame_drawn(BenchmarkFrameInput{.gpu = &stale, .frame_serial = 15});
    CHECK_FALSE(run.finished());
    for (std::uint64_t serial = 13; serial <= 14; ++serial) {
        auto const gpu = gpu_of(serial);
        run.on_frame_drawn(BenchmarkFrameInput{.gpu = &gpu, .frame_serial = serial + 3});
    }
    REQUIRE(run.finished());

    auto const samples = run.samples();
    REQUIRE(samples.size() == 4);
    for (auto const &sample: samples) {
        CAPTURE(sample.serial);
        CHECK(sample.gpu_valid);
        CHECK(sample.gpu_frame_ms == static_cast<float>(sample.serial));
    }

    auto const stages = run.stages();
    REQUIRE(stages.size() == 2);
    CHECK(stages[0].summary.count == 4);
    CHECK(stages[1].id == "forward_pass");
}

TEST_CASE("a run stops waiting for GPU timings after the drain limit") {
    BenchmarkOptions options;
    options.frame_count = 2;
    options.warmup_frame_count = 0;
    options.drain_frame_limit = 3;

    BenchmarkRun run{options, line_path()};
    run.on_frame_drawn(BenchmarkFrameInput{.frame_serial = 1}); // leaves warmup
    run.on_frame_drawn(BenchmarkFrameInput{.frame_serial = 2});
    run.on_frame_drawn(BenchmarkFrameInput{.frame_serial = 3});

    for (int i = 0; i < 2; ++i) {
        CHECK_FALSE(run.finished());
        run.on_frame_drawn(BenchmarkFrameInput{.frame_serial = 0});
    }
    run.on_frame_drawn(BenchmarkFrameInput{.frame_serial = 0});
    CHECK(run.finished());
    CHECK(run.analyse().gpu_frame.count == 0);

    // Frames without GPU timings leave the GPU columns empty in the CSV.
    auto const csv = run.to_csv();
    CHECK(csv.starts_with("frame,serial,path_t,displayed_ms"));
    CHECK(csv.find("event_texture_upload") != std::string::npos);
    CHECK(std::ranges::count(csv, '\n') == 3);
}

TEST_CASE("the single-run JSON keeps its earlier keys and adds environment and analysis") {
    BenchmarkOptions options;
    options.frame_count = 2;
    options.warmup_frame_count = 0;

    BenchmarkRun run{options, line_path()};
    auto const timings = FrameTimings{.full_frame_ms = 3.0F, .valid = true};
    run.on_frame_drawn(timings, true);
    run.on_frame_drawn(timings, true);
    run.on_frame_drawn(timings, true);
    REQUIRE(run.finished());

    auto const parsed = parse_json(run.to_json(BenchmarkEnvironment{.device_name = "gpu", .present_mode = "mailbox"}));
    REQUIRE(parsed.has_value());
    auto const &root = *parsed;
    CHECK(root["schema"].as_number() == 2.0);
    CHECK(root["device"].as_string() == "gpu");
    CHECK(root["frames"].as_number() == 2.0);
    CHECK(root["stages"].items()[0]["median_ms"].as_number() == 3.0);
    CHECK(root["full_frame_ms"].items().size() == 2);
    CHECK(root["environment"]["present_mode"].as_string() == "mailbox");
    CHECK(root["analysis"]["gpu_frame"]["count"].as_number() == 2.0);
    CHECK(root["analysis"]["events"]["texture_upload"]["total"].as_number() == 0.0);
}

TEST_CASE("aggregates take the median over repeats; scaling fits each scenario's loads") {
    std::vector<BenchmarkCaseResult> const results{
            case_result("lights", "point_lights", 100, 0, 5.0F, 2.0F, 1.0F),
            case_result("lights", "point_lights", 200, 0, 6.0F, 3.0F, 1.0F),
            case_result("lights", "point_lights", 100, 1, 5.2F, 2.2F, 1.0F),
            case_result("lights", "point_lights", 200, 1, 6.2F, 3.2F, 1.0F),
            case_result("lights", "point_lights", 100, 2, 5.4F, 2.4F, 1.0F),
            case_result("lights", "point_lights", 200, 2, 6.4F, 3.4F, 1.0F),
            case_result("game", "", 0, 0, 4.0F, 3.0F, 2.0F),
    };

    auto const aggregates = aggregate_cases(results);
    REQUIRE(aggregates.size() == 3);
    CHECK(aggregates[0].key == "lights@point_lights=100");
    CHECK(aggregates[0].repeats == 3);

    auto const *gpu = aggregates[0].metric("gpu_median_ms");
    REQUIRE(gpu != nullptr);
    CHECK(gpu->median == doctest::Approx(2.2));
    CHECK(gpu->min == doctest::Approx(2.0));
    CHECK(gpu->max == doctest::Approx(2.4));
    CHECK(aggregates[0].metric("stage_forward_pass_median_ms") != nullptr);

    auto const scaling = scaling_of(aggregates);
    REQUIRE(scaling.size() == 1); // the game has no load axis
    CHECK(scaling[0].loads == std::vector<std::uint32_t>{100, 200});
    CHECK(scaling[0].gpu.slope_ms_per_unit == doctest::Approx(0.01));

    auto const options = BenchmarkOptions{.mode = BenchmarkMode::suite, .repeats = 3};
    auto const suite = parse_json(suite_to_json(options, BenchmarkEnvironment{.device_name = "gpu"}, results));
    REQUIRE(suite.has_value());
    CHECK((*suite)["kind"].as_string() == "suite");
    CHECK((*suite)["cases"].items().size() == results.size());
    CHECK((*suite)["aggregates"].items().size() == 3);
    CHECK((*suite)["scaling"].items()[0]["fits"]["gpu"]["slope_ms_per_unit"].as_number() == doctest::Approx(0.01));

    auto const report = suite_report_markdown(options, BenchmarkEnvironment{.device_name = "gpu"}, results);
    CHECK(report.find("| lights@point_lights=100 |") != std::string::npos);
    CHECK(report.find("### lights (by point_lights)") != std::string::npos);
}

TEST_CASE("compare flags regressions only outside the noise between repeats") {
    auto const options = BenchmarkOptions{.mode = BenchmarkMode::suite, .repeats = 3};
    auto const environment = BenchmarkEnvironment{.device_name = "gpu"};

    auto const suite_of = [&](float shift, float noise) {
        std::vector<BenchmarkCaseResult> results;
        for (std::uint32_t repeat = 0; repeat < 3; ++repeat) {
            auto const wobble = noise * static_cast<float>(repeat);
            results.push_back(case_result("game", "", 0, repeat, 5.0F + shift + wobble, 4.0F + shift + wobble, 1.0F));
        }
        auto parsed = parse_json(suite_to_json(options, environment, results));
        REQUIRE(parsed.has_value());
        return std::move(*parsed);
    };

    auto const compare_options = BenchmarkCompareOptions{};

    SUBCASE("a clear regression fails") {
        auto const comparison = compare_benchmark_results(suite_of(0.0F, 0.1F), suite_of(2.0F, 0.1F), compare_options);
        CHECK(comparison.failed);
        REQUIRE(comparison.cases.size() == 1);
        auto const gpu = std::ranges::find(comparison.cases[0].metrics, "gpu_median_ms", &MetricComparison::name);
        REQUIRE(gpu != comparison.cases[0].metrics.end());
        CHECK(gpu->verdict == CompareVerdict::regressed);
        CHECK(comparison.markdown.find("slower") != std::string::npos);
    }

    SUBCASE("the same shift inside overlapping ranges is noise") {
        auto const comparison = compare_benchmark_results(suite_of(0.0F, 1.5F), suite_of(1.0F, 1.5F), compare_options);
        CHECK_FALSE(comparison.failed);
        auto const gpu = std::ranges::find(comparison.cases[0].metrics, "gpu_median_ms", &MetricComparison::name);
        REQUIRE(gpu != comparison.cases[0].metrics.end());
        CHECK(gpu->verdict == CompareVerdict::within_noise);
    }

    SUBCASE("an improvement passes") {
        auto const comparison = compare_benchmark_results(suite_of(2.0F, 0.1F), suite_of(0.0F, 0.1F), compare_options);
        CHECK_FALSE(comparison.failed);
        CHECK(comparison.markdown.find("faster") != std::string::npos);
    }
}

TEST_CASE("compare does not judge displayed times of presentation-paced cases") {
    auto const options = BenchmarkOptions{.mode = BenchmarkMode::suite, .repeats = 3};
    auto const environment = BenchmarkEnvironment{.device_name = "gpu"};

    auto const suite_of = [&](float displayed_ms) {
        std::vector<BenchmarkCaseResult> results;
        for (std::uint32_t repeat = 0; repeat < 3; ++repeat) {
            auto result =
                    case_result("game", "", 0, repeat, displayed_ms + 0.01F * static_cast<float>(repeat), 3.0F, 0.8F);
            result.analysis.bound.frames_with_gpu = 100;
            result.analysis.bound.presentation_bound = 95;
            results.push_back(std::move(result));
        }
        auto parsed = parse_json(suite_to_json(options, environment, results));
        REQUIRE(parsed.has_value());
        return std::move(*parsed);
    };

    // The displayed time doubles (say, the refresh rate halved) but GPU and CPU are unchanged: not a regression.
    auto const comparison = compare_benchmark_results(suite_of(6.94F), suite_of(13.88F), BenchmarkCompareOptions{});
    CHECK_FALSE(comparison.failed);
    REQUIRE(comparison.cases.size() == 1);
    auto const displayed =
            std::ranges::find(comparison.cases[0].metrics, "displayed_median_ms", &MetricComparison::name);
    REQUIRE(displayed != comparison.cases[0].metrics.end());
    CHECK(displayed->verdict == CompareVerdict::not_judged);
    CHECK(comparison.markdown.find("presentation-paced, not judged") != std::string::npos);
}

TEST_CASE("compare reads schema 1 single runs") {
    auto const base = parse_json(R"({"schema": 1, "device": "gpu", "render_extent": [1280, 720], "seed": 1337,
        "frames": 600, "stages": [{"id": "full_frame", "name": "Full Frame", "median_ms": 4.0, "p95_ms": 5.0},
        {"id": "forward_pass", "name": "Forward", "median_ms": 2.0, "p95_ms": 2.5}]})");
    auto const head = parse_json(R"({"schema": 1, "device": "other gpu", "render_extent": [1280, 720], "seed": 1337,
        "frames": 600, "stages": [{"id": "full_frame", "name": "Full Frame", "median_ms": 5.5, "p95_ms": 6.0},
        {"id": "forward_pass", "name": "Forward", "median_ms": 3.5, "p95_ms": 4.0},
        {"id": "bloom", "name": "Bloom", "median_ms": 0.2, "p95_ms": 0.3}]})");
    REQUIRE(base.has_value());
    REQUIRE(head.has_value());

    auto const comparison = compare_benchmark_results(*base, *head, BenchmarkCompareOptions{});
    CHECK(comparison.failed); // full frame +37.5%
    CHECK(comparison.notes.size() == 1); // the device differs
    CHECK(comparison.markdown.find("pass `bloom`") != std::string::npos);
    CHECK(comparison.markdown.find("| new |") != std::string::npos);

    std::array<char const *, 3> args{"--benchmark-compare=a.json,b.json", "--benchmark-threshold=5",
                                     "--benchmark-report=out.md"};
    auto const options = parse_benchmark_compare_options(args);
    REQUIRE(options.has_value());
    REQUIRE(options->has_value());
    CHECK((*options)->base == "a.json");
    CHECK((*options)->head == "b.json");
    CHECK((*options)->threshold_percent == 5.0);
    CHECK((*options)->report == std::filesystem::path{"out.md"});
}
