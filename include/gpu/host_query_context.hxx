#pragma once

#include <volk.h>

#include <cstdint>
#include <string_view>

#include <tracy/TracyVulkan.hpp>

struct VulkanContext;

struct HostQueryContext {
    tracy::VkCtx *context = nullptr;

    auto initialize(VulkanContext &vulkan_context, VkQueue queue, std::uint32_t family, std::string_view name) -> void;
    auto destroy() -> void;
};
