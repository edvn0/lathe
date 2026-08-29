#include <imgui.h>

#include <cstdint>

#include "memory_tracker.hxx"
auto on_memory_ui() -> void {
    static auto previous = MemoryTracker::stats();

    auto const stats = MemoryTracker::stats();

    auto const allocated_this_frame = stats.total_allocated_bytes - previous.total_allocated_bytes;

    auto const freed_this_frame = stats.total_freed_bytes - previous.total_freed_bytes;

    auto const allocations_this_frame = stats.total_allocations - previous.total_allocations;

    auto const frees_this_frame = stats.total_frees - previous.total_frees;

    auto const net_bytes =
            static_cast<std::int64_t>(allocated_this_frame) - static_cast<std::int64_t>(freed_this_frame);

    auto const average_allocation_size =
            allocations_this_frame != 0
                    ? static_cast<double>(allocated_this_frame) / static_cast<double>(allocations_this_frame)
                    : 0.0;

    previous = stats;

    constexpr auto kib = 1024.0;
    constexpr auto mib = 1024.0 * 1024.0;

    auto const to_kib = [](std::uint64_t const bytes) { return static_cast<double>(bytes) / kib; };

    auto const to_mib = [](std::uint64_t const bytes) { return static_cast<double>(bytes) / mib; };

    if (!ImGui::Begin("Memory")) {
        ImGui::End();
        return;
    }

    ImGui::SeparatorText("Heap");

    ImGui::Text("Live heap:            %10.2f MiB", to_mib(stats.live_bytes));

    ImGui::Text("Peak heap:            %10.2f MiB", to_mib(stats.peak_bytes));

    ImGui::Text("Live allocations:     %10llu", static_cast<unsigned long long>(stats.live_allocations));

    ImGui::Spacing();
    ImGui::SeparatorText("This frame");

    ImGui::Text("Allocations:          %10llu", static_cast<unsigned long long>(allocations_this_frame));

    ImGui::Text("Frees:                %10llu", static_cast<unsigned long long>(frees_this_frame));

    ImGui::Text("Allocated:            %10.2f KiB", to_kib(allocated_this_frame));

    ImGui::Text("Freed:                %10.2f KiB", to_kib(freed_this_frame));

    ImGui::Text("Net:                  %+10.2f KiB", static_cast<double>(net_bytes) / kib);

    ImGui::Text("Average allocation:   %10.2f bytes", average_allocation_size);

    ImGui::End();
}
