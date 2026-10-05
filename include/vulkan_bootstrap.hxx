#pragma once

#include <volk.h>

#include <cstdint>
#include <string_view>

#include "core/forward.hxx"

// From --screen-type=.
enum class ScreenType : std::uint8_t {
    windowed,
    fullscreen,
    borderless,
    // A windowed-size window on GLFW's null platform, presented through VK_EXT_headless_surface: no display server,
    // for benchmarks on headless machines and in containers.
    headless,
};

// Creates the window, Vulkan instance, surface, device, allocator and initial swapchain. Call context.destroy()
// afterwards whether or not this succeeds.
[[nodiscard]] auto initialize_vulkan(VulkanContext &context, ScreenType screen_type) noexcept -> bool;

// Logs "<operation> failed: <VkResult name> (<value>)".
auto report_vk_error(std::string_view operation, VkResult result) noexcept -> void;
