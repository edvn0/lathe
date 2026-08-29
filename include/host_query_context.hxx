#pragma once

#include <volk.h>

#include <tracy/TracyVulkan.hpp>

struct VulkanContext;

//
// The GPU-side counterpart to Tracy's CPU zones: command-buffer-recorded
// zones timestamped on the device and correlated back to the CPU timeline.
//
// Tracy calls this a "host query" context because it uses Vulkan 1.2's
// VK_EXT_host_query_reset mechanics (TracyVkContextHostCalibrated /
// TracyVkCollectHost) to reset and collect its internal query pool from the
// host, without needing a command buffer for either -- unlike a plain or
// calibrated Tracy Vulkan context, which needs one for both.
//
// Not every GPU/driver exposes calibrated device/host time domains, so
// initialize() falls back to a calibrated (queue + command buffer based)
// context, and finally to an uncalibrated one, in that order -- context is
// only ever null if TRACY_ENABLE isn't defined.
//
struct HostQueryContext {
    tracy::VkCtx *context = nullptr;

    auto initialize(VulkanContext &vulkan_context) -> void;
    auto destroy() -> void;
};
