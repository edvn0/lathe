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
    bool undecorated_window = false;

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

    // VK_KHR_ray_query and VK_KHR_acceleration_structure are enabled (ray tracing from compute shaders, no pipeline).
    bool ray_query_supported = false;

    // VK_EXT_debug_utils is enabled on the instance, so command-buffer labels are legal (Nsight pass names).
    bool debug_utils_enabled = false;

    bool depth_resolve_min_supported = false;

    // Sample counts usable for both colour and depth/stencil framebuffer attachments.
    VkSampleCountFlags supported_sample_counts = VK_SAMPLE_COUNT_1_BIT;

    HostQueryContext host_query_context{};
    HostQueryContext compute_host_query_context{};

    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;

    VkQueue compute_queue = VK_NULL_HANDLE;

    AsyncComputeMode async_compute_mode = AsyncComputeMode::automatic;
    // Load the Vulkan validation layer (Debug builds by default; --validation turns it on in any build).
    bool validation = false;
    bool sync_validation = false;

    bool vsync = true;

    // Ask for an HDR (scRGB) swapchain; swapchain.hdr_active() says whether the surface gave one.
    bool hdr = false;
    bool hdr_metadata_supported = false;
    float hdr_paper_white_nits = 203.0F;
    float hdr_peak_nits = 1000.0F;

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
