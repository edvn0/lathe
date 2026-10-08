#pragma once

#include <volk.h>

#include <cstdint>
#include <string_view>

#include "core/forward.hxx"

enum class ScreenType : std::uint8_t {
    windowed,
    fullscreen,
    borderless,
    headless,
};

[[nodiscard]] auto initialize_vulkan(VulkanContext &context, ScreenType screen_type) noexcept -> bool;

auto report_vk_error(std::string_view operation, VkResult result) noexcept -> void;
