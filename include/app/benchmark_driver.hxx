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

class BenchmarkDriver {
public:
    enum class Status : std::uint8_t {
        running,
        finished,
        failed,
    };

    [[nodiscard]]
    static auto create(BenchmarkOptions options,
                       Application const &application) -> std::expected<BenchmarkDriver, std::string>;

    auto begin_frame(Application &application) -> void;

    [[nodiscard]] auto render_size(BenchmarkRenderSize panel) const noexcept -> BenchmarkRenderSize;

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
