#pragma once

#include <atomic>
#include <condition_variable>
#include <volk.h>

#include <GLFW/glfw3.h>

#include "allocator.hxx"
#include "swapchain.hxx"

#include "forward.hxx"

struct QueueFamilies {
    std::uint32_t graphics = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t present = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]]
    auto complete() const noexcept -> bool {
        constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();

        return graphics != invalid && present != invalid;
    }
};

struct VulkanContext {
    GLFWwindow *window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VmaAllocator allocator{VK_NULL_HANDLE};

    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;

    QueueFamilies queue_families{};

    Swapchain swapchain{};

    std::atomic_bool running{true};
    std::atomic_bool framebuffer_dirty{false};
    std::atomic_int framebuffer_width{0};
    std::atomic_int framebuffer_height{0};

    std::mutex render_wake_mutex{};
    std::condition_variable render_wake_condition{};
};
