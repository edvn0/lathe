#pragma once

#include <volk.h>

#include <tracy/TracyVulkan.hpp>

struct VulkanContext;

// Tracy's GPU zones. Uses a host-calibrated context (queries reset and collected from the host), falling back
// to a calibrated and then an uncalibrated context. Null only without TRACY_ENABLE.
struct HostQueryContext {
    tracy::VkCtx *context = nullptr;

    auto initialize(VulkanContext &vulkan_context) -> void;
    auto destroy() -> void;
};
