#pragma once

#include <volk.h>

#include <string_view>

struct VulkanContext;

// VK_NV_device_diagnostic_checkpoints: markers in the command buffers that say, after the device is lost, which pass
// each queue had reached. A no-op where the extension is missing.
namespace device_checkpoints {

    // Called once the device exists; 'supported' is whether the extension was enabled on it.
    auto enable(bool supported) noexcept -> void;

    [[nodiscard]] auto enabled() noexcept -> bool;

    // Records a marker named 'name' into 'commands'. The name is interned, so the marker outlives the caller's string.
    auto set(VkCommandBuffer commands, std::string_view name) -> void;

    // Logs the last marker each queue of 'context' reached.
    auto report(VulkanContext const &context) -> void;

}
