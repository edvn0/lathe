// The editor: a Runtime with panels, gizmos and scene files on top (app/application.hxx), plus the benchmark runner,
// which drives the same Application headless. Everything else -- startup, the window, the frame loop -- is
// run_engine (runtime/engine_main.hxx), shared with the player.

#include <imgui.h>

#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "app/application.hxx"
#include "benchmark/benchmark.hxx"
#include "benchmark/benchmark_compare.hxx"
#include "benchmark/benchmark_driver.hxx"
#include "core/logger.hxx"
#include "core/thread_pool.hxx"
#include "core/random.hxx"
#include "gpu/context.hxx"
#include "gpu/swapchain.hxx"
#include "runtime/engine_cvars.hxx"
#include "runtime/engine_main.hxx"
#include "runtime/frame_clock.hxx"

namespace {

    class EditorProfile final : public EngineProfile {
    public:
        [[nodiscard]] auto name() const -> std::string_view override { return "lathe"; }
        [[nodiscard]] auto description() const -> std::string_view override { return "Lathe engine"; }

        [[nodiscard]] auto custom_title_bar() const -> bool override { return true; }

        [[nodiscard]] auto create_runtime(VulkanContext &context) -> std::unique_ptr<Runtime> override {
            auto application = std::make_unique<Application>(context);
            application->show_panels_from(cvars::show_panels.get());
            return application;
        }

        [[nodiscard]] auto early_exit() -> std::optional<int> override {
            if (auto const compare_options = benchmark_compare_options()) {
                return run_benchmark_compare(*compare_options);
            }

            return std::nullopt;
        }

        [[nodiscard]] auto prepare() -> std::expected<void, std::string> override {
            auto options = benchmark_options();

            if (!options) {
                return std::unexpected(std::format("Invalid benchmark arguments: {}", options.error()));
            }

            benchmark_options_ = std::move(*options);

            if (benchmark_options_) {
                set_fixed_random_seed(benchmark_options_->seed);
            }

            return {};
        }

        [[nodiscard]] auto deterministic() const -> bool override { return benchmark_options_.has_value(); }

        [[nodiscard]] auto attach(Runtime &runtime, VulkanContext &context)
                -> std::expected<void, std::string> override {
            if (!benchmark_options_) {
                return {};
            }

            auto driver = BenchmarkDriver::create(std::move(*benchmark_options_), static_cast<Application &>(runtime));

            if (!driver) {
                return std::unexpected(std::move(driver.error()));
            }

            ImGui::GetIO().IniFilename = nullptr;

            info("Benchmark: {} measured frames per run, seed {}, writing {}{}", driver->options().frame_count,
                 driver->options().seed, driver->options().output_path.string(),
                 context.vsync ? " (vsync on: displayed intervals are capped at the refresh rate)" : "");
            info("Benchmark: presenting with {} swapchain images", context.swapchain.image_count());
            benchmark_.emplace(std::move(*driver));

            return {};
        }

        auto save_scene(Runtime &runtime, std::filesystem::path const &path) -> void override {
            static_cast<Application &>(runtime).start_save_scene(path);
        }

        [[nodiscard]] auto fixed_timestep() const -> std::optional<float> override {
            return benchmark_ ? std::optional{benchmark_timestep} : std::nullopt;
        }

        auto begin_frame(Runtime &runtime) -> void override {
            if (benchmark_) {
                benchmark_->begin_frame(static_cast<Application &>(runtime));
            }
        }

        [[nodiscard]] auto end_frame(Runtime &runtime, VulkanContext const &context, CpuFrameTimes const &cpu,
                                     RenderExtent extent) -> LoopStatus override {
            if (!benchmark_) {
                return LoopStatus::running;
            }

            switch (benchmark_->end_frame(static_cast<Application &>(runtime), context, cpu,
                                          BenchmarkRenderSize{.width = extent.width, .height = extent.height})) {
                case BenchmarkDriver::Status::failed:
                    return LoopStatus::failed;
                case BenchmarkDriver::Status::finished:
                    return LoopStatus::finished;
                case BenchmarkDriver::Status::running:
                    break;
            }

            return LoopStatus::running;
        }

        [[nodiscard]] auto render_extent(RenderExtent desired) -> RenderExtent override {
            if (!benchmark_) {
                return desired;
            }

            auto const size = benchmark_->render_size(BenchmarkRenderSize{.width = desired.width, .height = desired.height});

            return RenderExtent{.width = size.width, .height = size.height};
        }

        [[nodiscard]] auto interrupted() const -> bool override {
            if (benchmark_ && !benchmark_->finished()) {
                error("Benchmark interrupted before it finished; results of unfinished runs were not written");
                return true;
            }

            return false;
        }

    private:
        std::optional<BenchmarkOptions> benchmark_options_;
        std::optional<BenchmarkDriver> benchmark_;
    };

}

auto main(int argc, char **argv) -> int {
    EditorProfile profile;

    return exit_promptly(run_engine(argc, argv, profile));
}
