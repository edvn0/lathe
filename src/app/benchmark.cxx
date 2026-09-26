#include "app/benchmark.hxx"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <numeric>
#include <utility>

#include "rendering/renderer.hxx"

namespace {

    [[nodiscard]] auto parse_count(std::string_view flag,
                                   std::string_view value) -> std::expected<std::uint32_t, std::string> {
        std::uint32_t result = 0;
        auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);

        if (error != std::errc{} || end != value.data() + value.size()) {
            return std::unexpected(std::format("{}: '{}' is not a non-negative integer", flag, value));
        }

        return result;
    }

    // Escapes the two characters a JSON string can't hold literally; device
    // names are the only free text written.
    [[nodiscard]] auto json_escape(std::string_view text) -> std::string {
        std::string escaped;
        escaped.reserve(text.size());

        for (auto const character: text) {
            if (character == '"' || character == '\\') {
                escaped.push_back('\\');
            }
            escaped.push_back(character);
        }

        return escaped;
    }

} // namespace

auto parse_benchmark_options(std::span<char const *const> args)
        -> std::expected<std::optional<BenchmarkOptions>, std::string> {
    BenchmarkOptions options;
    bool enabled = false;

    struct CountFlag {
        std::string_view prefix;
        std::uint32_t *value;
        bool allow_zero;
    };

    std::array const count_flags{
            CountFlag{"--benchmark-frames=", &options.frame_count, false},
            CountFlag{"--benchmark-warmup=", &options.warmup_frame_count, true},
            CountFlag{"--benchmark-max-warmup=", &options.max_warmup_frame_count, true},
            CountFlag{"--seed=", &options.seed, true},
    };

    for (auto const *raw: args) {
        std::string_view const arg = raw;

        if (arg == "--benchmark-screenshots") {
            options.keyframe_screenshots = true;
            continue;
        }

        if (constexpr std::string_view prefix = "--benchmark="; arg.starts_with(prefix)) {
            auto const path = arg.substr(prefix.size());

            if (path.empty()) {
                return std::unexpected(std::string{"--benchmark= needs an output path"});
            }

            options.output_path = std::filesystem::path{path};
            enabled = true;
            continue;
        }

        for (auto const &flag: count_flags) {
            if (!arg.starts_with(flag.prefix)) {
                continue;
            }

            auto const name = flag.prefix.substr(0, flag.prefix.size() - 1);
            auto const value = parse_count(name, arg.substr(flag.prefix.size()));

            if (!value) {
                return std::unexpected(value.error());
            }
            if (*value == 0 && !flag.allow_zero) {
                return std::unexpected(std::format("{} must be at least 1", name));
            }

            *flag.value = *value;
        }
    }

    if (!enabled) {
        return std::nullopt;
    }

    options.max_warmup_frame_count = std::max(options.max_warmup_frame_count, options.warmup_frame_count);
    return options;
}

auto summarise_timings(std::span<float const> samples_ms) -> TimingSummary {
    if (samples_ms.empty()) {
        return {};
    }

    std::vector<float> sorted{samples_ms.begin(), samples_ms.end()};
    std::ranges::sort(sorted);

    // Nearest rank: the smallest sample with at least `fraction` of the
    // samples at or below it.
    auto const percentile = [&](float fraction) {
        auto const rank = static_cast<std::size_t>(std::ceil(fraction * static_cast<float>(sorted.size())));
        return sorted[std::clamp(rank, std::size_t{1}, sorted.size()) - 1];
    };

    auto const sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);

    return TimingSummary{
            .mean_ms = static_cast<float>(sum / static_cast<double>(sorted.size())),
            .median_ms = percentile(0.5F),
            .p95_ms = percentile(0.95F),
            .min_ms = sorted.front(),
            .max_ms = sorted.back(),
    };
}

auto benchmark_stage_id(RenderStage stage) noexcept -> std::string_view {
    switch (stage) {
        using enum RenderStage;
        case FullFrame:
            return "full_frame";
        case Culling:
            return "gpu_culling";
        case ShadowPass:
            return "shadow_pass";
        case DepthPrepass:
            return "depth_prepass";
        case AmbientOcclusion:
            return "ambient_occlusion";
        case ForwardPass:
            return "forward_pass";
        case Composition:
            return "composition";
        case BloomPass:
            return "bloom";
        default:
            return "unknown";
    }
}

BenchmarkRun::BenchmarkRun(BenchmarkOptions options, std::vector<CameraKeyframe> keyframes) :
    options_(std::move(options)), keyframes_(std::move(keyframes)) {
    for (auto &samples: samples_ms_) {
        samples.reserve(options_.frame_count);
    }
}

auto BenchmarkRun::camera() const noexcept -> CameraKeyframe {
    if (!measuring_) {
        return sample_camera_path(keyframes_, 0.0F);
    }

    auto const t = static_cast<float>(measured_frames_) / static_cast<float>(options_.frame_count);
    return sample_camera_path(keyframes_, t);
}

auto BenchmarkRun::at_keyframe() const noexcept -> bool {
    if (!measuring_ || finished() || keyframes_.empty()) {
        return false;
    }

    // Which keyframe-to-keyframe segment measured frame `i` sits in; a
    // keyframe is crossed where that changes.
    auto const segment = [&](std::uint32_t frame) {
        return static_cast<std::uint64_t>(frame) * keyframes_.size() / options_.frame_count;
    };

    return measured_frames_ == 0 || segment(measured_frames_) != segment(measured_frames_ - 1);
}

auto BenchmarkRun::on_frame_drawn(StageTimings const &timings, bool streaming_idle) -> void {
    if (finished()) {
        return;
    }

    if (!measuring_) {
        ++warmup_frames_;

        auto const warmed_up = warmup_frames_ >= options_.warmup_frame_count && streaming_idle;
        auto const gave_up = warmup_frames_ >= options_.max_warmup_frame_count;

        if (warmed_up || gave_up) {
            measuring_ = true;
            streaming_settled_ = streaming_idle;
        }

        return;
    }

    if (timings.valid) {
        for (std::uint32_t stage = 0; stage < stage_count; ++stage) {
            samples_ms_[stage].push_back(timings.milliseconds[stage]);
        }
    }

    ++measured_frames_;
}

auto BenchmarkRun::to_json(BenchmarkEnvironment const &environment) const -> std::string {
    std::string json;

    json += "{\n";
    json += "  \"schema\": 1,\n";
    json += std::format("  \"device\": \"{}\",\n", json_escape(environment.device_name));
    json += std::format("  \"render_extent\": [{}, {}],\n", environment.render_width, environment.render_height);
    json += std::format("  \"seed\": {},\n", options_.seed);
    json += std::format("  \"keyframes\": {},\n", keyframes_.size());
    json += std::format("  \"frames\": {},\n", measured_frames_);
    json += std::format("  \"frames_with_timings\": {},\n", samples_ms_[0].size());
    json += std::format("  \"warmup_frames\": {},\n", warmup_frames_);
    json += std::format("  \"streaming_settled\": {},\n", streaming_settled_);
    json += "  \"stages\": [\n";

    for (std::uint32_t stage = 0; stage < stage_count; ++stage) {
        auto const render_stage = static_cast<RenderStage>(stage);
        auto const summary = summarise_timings(samples_ms_[stage]);

        json += std::format("    {{\"id\": \"{}\", \"name\": \"{}\", \"mean_ms\": {:.4f}, \"median_ms\": {:.4f}, "
                            "\"p95_ms\": {:.4f}, \"min_ms\": {:.4f}, \"max_ms\": {:.4f}}}{}\n",
                            benchmark_stage_id(render_stage), to_string(render_stage), summary.mean_ms,
                            summary.median_ms, summary.p95_ms, summary.min_ms, summary.max_ms,
                            stage + 1 < stage_count ? "," : "");
    }

    json += "  ],\n";

    // Full-frame time of every measured frame, in path order -- enough to
    // plot where along the loop a regression sits.
    json += "  \"full_frame_ms\": [";

    auto const &full_frame = samples_ms_[static_cast<std::uint32_t>(RenderStage::FullFrame)];
    for (std::size_t i = 0; i < full_frame.size(); ++i) {
        json += std::format("{}{:.4f}", i == 0 ? "" : ", ", full_frame[i]);
    }

    json += "]\n}\n";
    return json;
}

auto BenchmarkRun::write(BenchmarkEnvironment const &environment) const -> std::expected<void, std::string> {
    if (auto const parent = options_.output_path.parent_path(); !parent.empty()) {
        std::error_code error;
        std::filesystem::create_directories(parent, error);

        if (error) {
            return std::unexpected(std::format("could not create {}: {}", parent.string(), error.message()));
        }
    }

    std::ofstream file{options_.output_path, std::ios::binary | std::ios::trunc};

    if (!file) {
        return std::unexpected(std::format("could not open {} for writing", options_.output_path.string()));
    }

    file << to_json(environment);

    if (!file) {
        return std::unexpected(std::format("could not write {}", options_.output_path.string()));
    }

    return {};
}
