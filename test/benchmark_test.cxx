#include <doctest/doctest.h>

#include <glm/glm.hpp>

#include <array>
#include <string>
#include <vector>

#include "app/benchmark.hxx"
#include "core/random.hxx"
#include "rendering/renderer.hxx"
#include "scene/camera_path.hxx"

// Benchmarks are only comparable if every run sees the same frames: a continuous looping camera path, the
// stats the comparison script expects, and a reproducible seed.

namespace {

    [[nodiscard]] auto square_path() -> std::vector<CameraKeyframe> {
        return {
                {.position = {0.0F, 2.0F, 0.0F}, .target = {0.0F, 0.0F, -1.0F}},
                {.position = {10.0F, 4.0F, 0.0F}, .target = {1.0F, 0.0F, 0.0F}},
                {.position = {10.0F, 2.0F, 10.0F}, .target = {0.0F, 0.0F, 1.0F}},
                {.position = {0.0F, 6.0F, 10.0F}, .target = {-1.0F, 0.0F, 0.0F}},
        };
    }

    [[nodiscard]] auto near(glm::vec3 const &a, glm::vec3 const &b) -> bool { return glm::distance(a, b) < 1e-4F; }

} // namespace

TEST_CASE("camera path passes through every keyframe and closes the loop") {
    auto const keyframes = square_path();
    auto const count = static_cast<float>(keyframes.size());

    for (std::size_t i = 0; i < keyframes.size(); ++i) {
        auto const sample = sample_camera_path(keyframes, static_cast<float>(i) / count);
        CAPTURE(i);
        CHECK(near(sample.position, keyframes[i].position));
        CHECK(near(sample.target, keyframes[i].target));
    }

    // Approaching t = 1 lands back on the first keyframe, no jump.
    CHECK(glm::distance(sample_camera_path(keyframes, 0.9999F).position, keyframes[0].position) < 0.01F);
    CHECK(near(sample_camera_path(keyframes, 1.25F).position, sample_camera_path(keyframes, 0.25F).position));
}

TEST_CASE("camera path moves smoothly between keyframes") {
    auto const keyframes = square_path();
    constexpr int steps = 400;

    auto previous = sample_camera_path(keyframes, 0.0F).position;
    auto largest_step = 0.0F;

    for (int i = 1; i <= steps; ++i) {
        auto const position = sample_camera_path(keyframes, static_cast<float>(i) / steps).position;
        largest_step = std::max(largest_step, glm::distance(previous, position));
        previous = position;
    }

    // The loop is ~40 m long, so each 1/400th step is well under a metre.
    CHECK(largest_step < 0.5F);
}

TEST_CASE("timing summaries use nearest-rank percentiles") {
    std::vector<float> samples;
    for (int i = 100; i >= 1; --i) {
        samples.push_back(static_cast<float>(i));
    }

    auto const summary = summarise_timings(samples);
    CHECK(summary.mean_ms == doctest::Approx(50.5F));
    CHECK(summary.median_ms == 50.0F);
    CHECK(summary.p95_ms == 95.0F);
    CHECK(summary.min_ms == 1.0F);
    CHECK(summary.max_ms == 100.0F);

    auto const empty = summarise_timings({});
    CHECK(empty.median_ms == 0.0F);
}

TEST_CASE("benchmark options") {
    SUBCASE("absent without --benchmark=") {
        std::array<char const *, 1> args{"--seed=7"};
        auto const options = parse_benchmark_options(args);
        REQUIRE(options.has_value());
        CHECK_FALSE(options->has_value());
    }

    SUBCASE("parses every flag") {
        std::array<char const *, 5> args{"--benchmark=out/perf.json", "--benchmark-frames=300", "--benchmark-warmup=10",
                                         "--benchmark-max-warmup=50", "--seed=42"};
        auto const options = parse_benchmark_options(args);
        REQUIRE(options.has_value());
        REQUIRE(options->has_value());
        CHECK((*options)->output_path == "out/perf.json");
        CHECK((*options)->frame_count == 300);
        CHECK((*options)->warmup_frame_count == 10);
        CHECK((*options)->max_warmup_frame_count == 50);
        CHECK((*options)->seed == 42);
    }

    SUBCASE("screenshots are opt-in") {
        std::array<char const *, 2> args{"--benchmark=a.json", "--benchmark-screenshots"};
        auto const options = parse_benchmark_options(args);
        REQUIRE(options.has_value());
        REQUIRE(options->has_value());
        CHECK((*options)->keyframe_screenshots);
    }

    SUBCASE("rejects garbage and zero frames") {
        std::array<char const *, 2> bad_number{"--benchmark=a.json", "--benchmark-frames=12x"};
        CHECK_FALSE(parse_benchmark_options(bad_number).has_value());

        std::array<char const *, 2> zero_frames{"--benchmark=a.json", "--benchmark-frames=0"};
        CHECK_FALSE(parse_benchmark_options(zero_frames).has_value());
    }
}

TEST_CASE("benchmark run warms up until streaming settles, then records one lap") {
    BenchmarkOptions options;
    options.frame_count = 8;
    options.warmup_frame_count = 2;
    options.max_warmup_frame_count = 100;

    auto const keyframes = square_path();
    BenchmarkRun run{options, keyframes};

    StageTimings timings{};
    timings.valid = true;
    timings.milliseconds.fill(1.0F);

    // Still streaming: stays parked at the first keyframe past the minimum.
    for (int i = 0; i < 5; ++i) {
        CHECK(near(run.camera().position, keyframes[0].position));
        run.on_frame_drawn(timings, false);
    }

    run.on_frame_drawn(timings, true);

    std::uint32_t keyframe_frames = 0;

    for (std::uint32_t i = 0; i < options.frame_count; ++i) {
        CHECK_FALSE(run.finished());
        keyframe_frames += run.at_keyframe() ? 1U : 0U;
        run.on_frame_drawn(timings, true);
    }

    // 8 frames over 4 keyframes: frames 0, 2, 4, 6.
    CHECK(keyframe_frames == keyframes.size());

    CHECK(run.finished());

    auto const json = run.to_json(BenchmarkEnvironment{.device_name = "a \"quoted\" gpu"});
    CHECK(json.find("\"frames\": 8") != std::string::npos);
    CHECK(json.find("\"warmup_frames\": 6") != std::string::npos);
    CHECK(json.find("\"streaming_settled\": true") != std::string::npos);
    CHECK(json.find("\"id\": \"forward_pass\"") != std::string::npos);
    CHECK(json.find("a \\\"quoted\\\" gpu") != std::string::npos);
    CHECK(json.find("\"cluster_grid\": [16, 9, 24, 256]") != std::string::npos);
    CHECK(json.find("\"occlusion_culling\": false") != std::string::npos);

    // The occlusion stages are always present, so on/off runs compare stage by stage.
    CHECK(json.find("\"id\": \"hiz_build\"") != std::string::npos);
    CHECK(json.find("\"id\": \"occlusion_culling\"") != std::string::npos);
    CHECK(json.find("\"id\": \"depth_prepass_late\"") != std::string::npos);

    auto const occlusion_json = run.to_json(BenchmarkEnvironment{.device_name = "gpu", .occlusion_culling = true});
    CHECK(occlusion_json.find("\"occlusion_culling\": true") != std::string::npos);

    // Meshlet-level occlusion is recorded separately, so runs with and without it aren't mistaken for one another.
    CHECK(json.find("\"meshlet_occlusion\": false") != std::string::npos);
    CHECK(occlusion_json.find("\"meshlet_occlusion\": false") != std::string::npos);

    auto const meshlet_json = run.to_json(
            BenchmarkEnvironment{.device_name = "gpu", .occlusion_culling = true, .meshlet_occlusion = true});
    CHECK(meshlet_json.find("\"meshlet_occlusion\": true") != std::string::npos);
}

TEST_CASE("benchmark counters average over valid frames and report the last value") {
    BenchmarkOptions options;
    options.frame_count = 3;
    options.warmup_frame_count = 1;

    BenchmarkRun run{options, square_path()};

    StageTimings timings{};
    timings.valid = true;

    run.on_frame_drawn(timings, true); // warmup

    // Occlusion readbacks: 10, an invalid frame that must not count as zero, then 20.
    run.on_frame_drawn(timings, true, BenchmarkCounters{.occlusion_valid = true, .frustum_visible_instances = 10});
    run.on_frame_drawn(timings, true, BenchmarkCounters{.frustum_visible_instances = 99});
    run.on_frame_drawn(timings, true,
                       BenchmarkCounters{.occlusion_valid = true,
                                         .frustum_visible_instances = 20,
                                         .cluster_valid = true,
                                         .stored_lights = 7});
    REQUIRE(run.finished());

    auto const json = run.to_json(BenchmarkEnvironment{.device_name = "gpu"});
    CHECK(json.find("\"frustum_visible_instances\": {\"mean\": 15.0000, \"final\": 20}") != std::string::npos);
    CHECK(json.find("\"stored_lights\": {\"mean\": 7.0000, \"final\": 7}") != std::string::npos);

    // Never read back: null, not zero.
    CHECK(json.find("\"occluded_meshlets\": {\"mean\": null, \"final\": null}") != std::string::npos);
}

TEST_CASE("simulated time restarts with the measured lap, independent of warmup length") {
    BenchmarkOptions options;
    options.frame_count = 4;
    options.warmup_frame_count = 3;

    BenchmarkRun run{options, square_path()};

    StageTimings timings{};
    timings.valid = true;

    for (int i = 0; i < 3; ++i) {
        CHECK(run.simulated_time() == 0.0F);
        run.on_frame_drawn(timings, true);
    }

    CHECK(run.simulated_time() == 0.0F);
    run.on_frame_drawn(timings, true);
    CHECK(run.simulated_time() == doctest::Approx(benchmark_timestep));
}

TEST_CASE("a fixed seed reproduces random sequences, per stream") {
    set_fixed_random_seed(1337U);

    auto first = make_random_engine(2);
    auto second = make_random_engine(2);
    auto other_stream = make_random_engine(3);

    auto const a = first();
    CHECK(a == second());
    CHECK(a != other_stream());

    set_fixed_random_seed(std::nullopt);
}
