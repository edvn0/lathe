#pragma once

#include <volk.h>

#include <cstdint>
#include <string_view>

#include <tracy/TracyVulkan.hpp>

struct VulkanContext;

// Tracy's GPU zones for one queue. Uses a host-calibrated context (queries reset and collected from the host, so it
// needs no queue), falling back to a calibrated and then an uncalibrated context on `queue`. Null only without
// TRACY_ENABLE. `name` labels the track in Tracy.
struct HostQueryContext {
    tracy::VkCtx *context = nullptr;

    auto initialize(VulkanContext &vulkan_context, VkQueue queue, std::uint32_t family, std::string_view name) -> void;
    auto destroy() -> void;
};
