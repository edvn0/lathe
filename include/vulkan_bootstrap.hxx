#pragma once

#include <volk.h>

#include <string_view>

#include "core/forward.hxx"

// From --screen-type=.
enum class ScreenType {
    windowed,
    fullscreen,
    borderless,
};

[[nodiscard]] auto parse_screen_type(int argc, char **argv) noexcept -> ScreenType;

// Creates the window, Vulkan instance, surface, device, allocator and initial swapchain. Call context.destroy()
// afterwards whether or not this succeeds.
[[nodiscard]] auto initialize_vulkan(VulkanContext &context, ScreenType screen_type) noexcept -> bool;

// Logs "<operation> failed: <VkResult name> (<value>)".
auto report_vk_error(std::string_view operation, VkResult result) noexcept -> void;
