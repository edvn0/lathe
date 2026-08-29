#pragma once

#include <volk.h>

#include <string_view>

#include "forward.hxx"

// Selects the window mode main() passes to initialize_vulkan() -- parsed
// from a --screen-type= command line argument.
enum class ScreenType {
    windowed,
    fullscreen,
    borderless,
};

[[nodiscard]] auto parse_screen_type(int argc, char **argv) noexcept -> ScreenType;

// Brings up everything main() needs before the render loop can start: the
// GLFW window, the Vulkan instance/surface/device, the allocator, and the
// initial swapchain. On failure, context is left in whatever partial state
// the failing step produced -- the caller is expected to call
// context.destroy() either way.
[[nodiscard]] auto initialize_vulkan(VulkanContext &context, ScreenType screen_type) noexcept -> bool;

// Logs "<operation> failed: <VkResult name> (<value>)" -- shared by the
// bootstrap steps above and by callers elsewhere (e.g. main.cxx's shutdown
// path) that need to report a raw VkResult failure the same way.
auto report_vk_error(std::string_view operation, VkResult result) noexcept -> void;
