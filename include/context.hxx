#pragma once

#include <volk.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <array>

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

    VkCommandPool one_time_pool;
    std::array<VkCommandBuffer, 4> one_time_command_buffers;
    std::uint32_t one_time_buffer_index{0};
    auto one_time_submit(std::function<void(VkCommandBuffer)> &&) -> void;

    std::mutex render_wake_mutex{};
    std::condition_variable render_wake_condition{};
};
