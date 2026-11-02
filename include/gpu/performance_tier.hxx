#pragma once

#include <volk.h>

#include <cstdint>
#include <optional>
#include <string_view>

// A coarse performance class of the GPU, detected from its type and memory, that picks the defaults of the settings that
// trade quality for speed (a project setting left on "auto" follows it). Overridable with `[render] performance_tier`.
enum class PerformanceTier : std::uint8_t {
    low,
    medium,
    high,
};

// What each tier chooses. Mirrors the "auto" values of the settings it feeds.
struct TierProfile {
    // The ray traced shadow mask is traced at the screen size >> this (0 full, 1 half, 2 quarter).
    std::uint32_t rt_shadow_resolution_shift = 0;
    // The far probe cascade updates every this many frames.
    std::uint32_t far_cascade_update_interval = 2;
};

namespace performance_tier {

    // Discrete and virtual GPUs (an external GPU is discrete too) are medium below 6 GiB of device local memory and high
    // from it; integrated, CPU and unknown devices are low.
    constexpr auto high_memory_bytes = std::uint64_t{6} * 1024 * 1024 * 1024;

    [[nodiscard]] auto detect(VkPhysicalDeviceType type, std::uint64_t device_local_bytes) noexcept -> PerformanceTier;
    [[nodiscard]] auto profile(PerformanceTier tier) noexcept -> TierProfile;
    [[nodiscard]] auto name(PerformanceTier tier) noexcept -> std::string_view;
    // "low", "medium" or "high"; nothing for anything else (including "auto").
    [[nodiscard]] auto parse(std::string_view text) noexcept -> std::optional<PerformanceTier>;

}
