#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rendering/render_stage.hxx"
#include "scene/camera_path.hxx"

struct StageTimings;

// --benchmark mode: fly the editor camera around the game's
// benchmark_camera_path() loop at a fixed timestep with a fixed scene seed,
// record every frame's per-stage GPU timings, write a JSON summary and
// exit. Two builds benchmarked on the same machine are then directly
// comparable (tools/perf/compare_benchmarks.py, .github/workflows/perf.yml).
//
//   --benchmark=<out.json>        enables it; where the results go
//   --benchmark-frames=<n>        measured frames, one lap of the loop (600)
//   --benchmark-warmup=<n>        minimum frames parked at the first
//                                 keyframe before measuring (60)
//   --benchmark-max-warmup=<n>    measure anyway after this many, even if
//                                 streaming hasn't settled (1200)
//   --seed=<n>                    scene seed (1337), see core/random.hxx
//   --benchmark-screenshots       also screenshot (F12-style, into
//                                 screenshots/) the first frame at or past
//                                 each keyframe -- what the run looked at
struct BenchmarkOptions {
    std::filesystem::path output_path;
    std::uint32_t frame_count = 600;
    std::uint32_t warmup_frame_count = 60;
    std::uint32_t max_warmup_frame_count = 1200;
    std::uint32_t seed = 1337;
    bool keyframe_screenshots = false;
};

// std::nullopt when --benchmark= isn't given (the other flags are then
// ignored); an error message for a malformed or zero value.
[[nodiscard]]
auto parse_benchmark_options(std::span<char const *const> args)
        -> std::expected<std::optional<BenchmarkOptions>, std::string>;

// Simulated seconds per benchmark frame -- wind, enemy motion etc. advance
// by exactly this, so frame N shows the same scene in every run.
inline constexpr float benchmark_timestep = 1.0F / 60.0F;

struct TimingSummary {
    float mean_ms = 0.0F;
    float median_ms = 0.0F;
    float p95_ms = 0.0F;
    float min_ms = 0.0F;
    float max_ms = 0.0F;
};

// Nearest-rank percentiles over `samples_ms`; all zero when empty.
[[nodiscard]]
auto summarise_timings(std::span<float const> samples_ms) -> TimingSummary;

// Stable snake_case key for a stage in the JSON output.
[[nodiscard]]
auto benchmark_stage_id(RenderStage stage) noexcept -> std::string_view;

struct BenchmarkEnvironment {
    std::string device_name;
    std::uint32_t render_width = 0;
    std::uint32_t render_height = 0;
};

class BenchmarkRun {
public:
    BenchmarkRun(BenchmarkOptions options, std::vector<CameraKeyframe> keyframes);

    // Where the camera goes for the frame about to be drawn: the first
    // keyframe while warming up, then one lap of the loop.
    [[nodiscard]]
    auto camera() const noexcept -> CameraKeyframe;

    // After each drawn frame. `timings` lag the drawn frame by the frames
    // in flight -- harmless, the camera moves continuously. `streaming_idle`
    // gates the end of warmup: nothing may still be loading when
    // measurement starts.
    auto on_frame_drawn(StageTimings const &timings, bool streaming_idle) -> void;

    // True for the frame about to be drawn if it's the first measured
    // frame at or past a keyframe (one per keyframe per lap).
    [[nodiscard]]
    auto at_keyframe() const noexcept -> bool;

    [[nodiscard]]
    auto finished() const noexcept -> bool {
        return measured_frames_ >= options_.frame_count;
    }

    [[nodiscard]]
    auto options() const noexcept -> BenchmarkOptions const & {
        return options_;
    }

    [[nodiscard]]
    auto to_json(BenchmarkEnvironment const &environment) const -> std::string;

    auto write(BenchmarkEnvironment const &environment) const -> std::expected<void, std::string>;

private:
    BenchmarkOptions options_;
    std::vector<CameraKeyframe> keyframes_;

    bool measuring_ = false;
    bool streaming_settled_ = true;
    std::uint32_t warmup_frames_ = 0;
    std::uint32_t measured_frames_ = 0;

    // Per stage, one sample per measured frame with valid timings.
    std::array<std::vector<float>, stage_count> samples_ms_{};
};
