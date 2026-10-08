#pragma once

#include <volk.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include <GLFW/glfw3.h>

#include "core/allocator.hxx"
#include "gpu/host_query_context.hxx"
#include "gpu/queue_selection.hxx"
#include "gpu/queue_set.hxx"
#include "gpu/swapchain.hxx"

#include "core/forward.hxx"

struct VulkanContext {
    GLFWwindow *window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VmaAllocator allocator{VK_NULL_HANDLE};

    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    bool shader_objects_supported = false;

    bool calibrated_timestamps_supported = false;
    bool host_calibrated_timestamps_supported = false;

    bool mesh_shader_queries_supported = false;

    bool depth_resolve_min_supported = false;

    HostQueryContext host_query_context{};
    HostQueryContext compute_host_query_context{};

    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;

    VkQueue compute_queue = VK_NULL_HANDLE;

    AsyncComputeMode async_compute_mode = AsyncComputeMode::automatic;
    bool sync_validation = false;

    bool vsync = true;

    std::optional<VkPresentModeKHR> present_mode;
    std::uint32_t swapchain_image_count = 0;

    QueueFamilies queue_families{};

    QueueSet queue_set{};

    Swapchain swapchain{};

    std::atomic_bool running{true};

    std::atomic_bool device_lost{false};
    std::atomic_bool framebuffer_dirty{false};
    std::atomic_int framebuffer_width{0};
    std::atomic_int framebuffer_height{0};

    VkCommandPool one_time_pool;
    std::array<VkCommandBuffer, 4> one_time_command_buffers;
    std::uint32_t one_time_buffer_index{0};
    auto one_time_submit(std::function<void(VkCommandBuffer)> &&) -> void;

    auto destroy() -> void;
};
