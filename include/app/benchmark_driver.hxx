#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include "app/benchmark.hxx"
#include "app/benchmark_scenarios.hxx"
#include "app/frame_clock.hxx"
#include "core/perf_events.hxx"

struct Application;
struct VulkanContext;

// Runs --benchmark / --benchmark-suite from the main loop: sets up each case's scene, flies its camera, collects each
// frame's samples and writes the results. main.cxx calls begin_frame() before updating and drawing, and end_frame()
// after presenting.
class BenchmarkDriver {
public:
    enum class Status : std::uint8_t {
        running,
        finished,
        failed,
    };

    // Fails for an unknown scenario or a game without a benchmark path when one is needed.
    [[nodiscard]]
    static auto create(BenchmarkOptions options,
                       Application const &application) -> std::expected<BenchmarkDriver, std::string>;

    // Before Application::update(): starts the next case if none is running (repopulating the scene), then poses the
    // camera and sets the shader clock for the frame about to be drawn.
    auto begin_frame(Application &application) -> void;

    // The render size for this frame, given the Viewport panel's.
    [[nodiscard]] auto render_size(BenchmarkRenderSize panel) const noexcept -> BenchmarkRenderSize;

    // After the frame was presented. `cpu` is FrameClock::finish_frame(); `render_extent` the size it rendered at.
    [[nodiscard]]
    auto end_frame(Application &application, VulkanContext const &context, CpuFrameTimes const &cpu,
                   BenchmarkRenderSize render_extent) -> Status;

    [[nodiscard]] auto finished() const noexcept -> bool { return case_index_ >= cases_.size(); }

    [[nodiscard]] auto options() const noexcept -> BenchmarkOptions const & { return options_; }

private:
    BenchmarkDriver() = default;

    auto start_case(Application &application) -> void;
    [[nodiscard]] auto finish_case(Application &application, VulkanContext const &context,
                                   BenchmarkRenderSize render_extent) -> Status;
    [[nodiscard]] auto environment(Application const &application, VulkanContext const &context,
                                   BenchmarkRenderSize render_extent) const -> BenchmarkEnvironment;

    BenchmarkOptions options_;
    std::vector<BenchmarkScenario> scenarios_;
    std::vector<BenchmarkCase> cases_;
    std::size_t case_index_ = 0;

    // Materials synthetic scenes keep alive (BenchmarkScenarioContext::owned_materials), released when the scene is
    // next replaced. Ones still held at exit go with the renderer.
    std::vector<MaterialHandle> scenario_materials_;
    auto release_scenario_materials(Application &application) -> void;

    std::optional<BenchmarkRun> run_;
    std::uint32_t render_scale_percent_ = 100;
    ThermalSample thermals_start_{};
    std::vector<BenchmarkCaseResult> results_;

    std::uint64_t last_frame_serial_ = 0;
    PerfEventCounts last_events_{};
    std::uint64_t last_allocations_ = 0;
    std::uint64_t last_allocated_bytes_ = 0;
};
