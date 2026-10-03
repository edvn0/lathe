#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string_view>

// Process-wide counters of the things that tend to cause frame-time spikes: uploads, shader builds, frame graph
// recompiles, resizes and device-wide waits. Engine code calls perf_events::record() where one happens; the benchmark
// snapshots the counters once per frame and stores the difference, so a slow frame can be matched to what happened in
// it (docs/perf-benchmark.md, "Hitches"). Recording is one relaxed atomic add, cheap enough to leave on everywhere.
enum class PerfEvent : std::uint8_t {
    texture_upload,
    model_install,
    terrain_chunk_upload,
    shader_compile,
    shader_object_build,
    frame_graph_compile,
    transient_allocation,
    swapchain_recreate,
    render_resize,
    device_wait_idle,
    count,
};

inline constexpr std::size_t perf_event_count = static_cast<std::size_t>(PerfEvent::count);

// Stable ids, used as CSV column and JSON key names.
[[nodiscard]] constexpr auto perf_event_name(PerfEvent event) noexcept -> std::string_view {
    switch (event) {
        case PerfEvent::texture_upload:
            return "texture_upload";
        case PerfEvent::model_install:
            return "model_install";
        case PerfEvent::terrain_chunk_upload:
            return "terrain_chunk_upload";
        case PerfEvent::shader_compile:
            return "shader_compile";
        case PerfEvent::shader_object_build:
            return "shader_object_build";
        case PerfEvent::frame_graph_compile:
            return "frame_graph_compile";
        case PerfEvent::transient_allocation:
            return "transient_allocation";
        case PerfEvent::swapchain_recreate:
            return "swapchain_recreate";
        case PerfEvent::render_resize:
            return "render_resize";
        case PerfEvent::device_wait_idle:
            return "device_wait_idle";
        case PerfEvent::count:
            break;
    }
    return "unknown";
}

struct PerfEventCounts {
    std::array<std::uint64_t, perf_event_count> values{};

    [[nodiscard]] constexpr auto operator[](PerfEvent event) const noexcept -> std::uint64_t {
        return values[static_cast<std::size_t>(event)];
    }

    [[nodiscard]] constexpr auto any() const noexcept -> bool {
        for (auto const value: values) {
            if (value != 0) {
                return true;
            }
        }
        return false;
    }

    // Events since `earlier`. Counters only grow, so this never wraps for snapshots taken in order.
    [[nodiscard]] constexpr auto since(PerfEventCounts const &earlier) const noexcept -> PerfEventCounts {
        PerfEventCounts delta;
        for (std::size_t i = 0; i < perf_event_count; ++i) {
            delta.values[i] = values[i] - earlier.values[i];
        }
        return delta;
    }
};

namespace perf_events {

    namespace detail {
        // NOLINTNEXTLINE: process-wide by design, like MemoryTracker's counters.
        inline std::array<std::atomic<std::uint64_t>, perf_event_count> counters{};
    } // namespace detail

    inline auto record(PerfEvent event, std::uint64_t amount = 1) noexcept -> void {
        detail::counters[static_cast<std::size_t>(event)].fetch_add(amount, std::memory_order_relaxed);
    }

    [[nodiscard]] inline auto snapshot() noexcept -> PerfEventCounts {
        PerfEventCounts counts;
        for (std::size_t i = 0; i < perf_event_count; ++i) {
            counts.values[i] = detail::counters[i].load(std::memory_order_relaxed);
        }
        return counts;
    }

} // namespace perf_events
