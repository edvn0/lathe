#include "gpu/performance_tier.hxx"

namespace performance_tier {

    auto detect(VkPhysicalDeviceType type, std::uint64_t device_local_bytes) noexcept -> PerformanceTier {
        if (type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || type == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
            return device_local_bytes >= high_memory_bytes ? PerformanceTier::high : PerformanceTier::medium;
        }
        return PerformanceTier::low;
    }

    auto profile(PerformanceTier tier) noexcept -> TierProfile {
        switch (tier) {
            case PerformanceTier::low:
                return {.rt_shadow_resolution_shift = 2, .far_cascade_update_interval = 4};
            case PerformanceTier::medium:
                return {.rt_shadow_resolution_shift = 1, .far_cascade_update_interval = 2};
            case PerformanceTier::high:
                break;
        }
        return {.rt_shadow_resolution_shift = 0, .far_cascade_update_interval = 2};
    }

    auto name(PerformanceTier tier) noexcept -> std::string_view {
        switch (tier) {
            case PerformanceTier::low:
                return "low";
            case PerformanceTier::medium:
                return "medium";
            case PerformanceTier::high:
                break;
        }
        return "high";
    }

    auto parse(std::string_view text) noexcept -> std::optional<PerformanceTier> {
        if (text == "low") {
            return PerformanceTier::low;
        }
        if (text == "medium") {
            return PerformanceTier::medium;
        }
        if (text == "high") {
            return PerformanceTier::high;
        }
        return std::nullopt;
    }

}
