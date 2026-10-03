#pragma once

#include <volk.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>

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

    // VK_EXT_shader_object and the dynamic state it needs. Decides the enabled features and VkPipeline vs
    // ShaderObjectSet.
    bool shader_objects_supported = false;

    // VK_EXT_calibrated_timestamps, with time domains usable by Tracy's host-calibrated context.
    bool calibrated_timestamps_supported = false;
    bool host_calibrated_timestamps_supported = false;

    // Lets the pipeline-statistics query count task/mesh invocations.
    bool mesh_shader_queries_supported = false;

    // VK_RESOLVE_MODE_MIN_BIT in VkPhysicalDeviceDepthStencilResolveProperties::supportedDepthResolveModes. Under
    // MSAA, Hi-Z occlusion culling needs it to resolve the farthest sample (reverse-Z) of each pixel.
    bool depth_resolve_min_supported = false;

    // Tracy's GPU contexts: graphics, and compute when it is a separate queue (null otherwise).
    HostQueryContext host_query_context{};
    HostQueryContext compute_host_query_context{};

    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;

    // The async compute queue. Equals graphics_queue when queue_families.topology is single.
    VkQueue compute_queue = VK_NULL_HANDLE;

    // Set before initialize_vulkan: --async-compute=auto|off|same-family and --sync-validation.
    AsyncComputeMode async_compute_mode = AsyncComputeMode::automatic;
    bool sync_validation = false;

    // --async-compute-smoke: submit empty compute and graphics batches each frame to exercise the timelines.
    bool async_compute_smoke = false;

    // --frame-graph-serialize: the frame graph compiler puts ALL_COMMANDS barriers between all passes, to tell a missing
    // dependency from a real bug (CompileOptions::serialize).
    bool frame_graph_serialize = false;

    QueueFamilies queue_families{};

    QueueSet queue_set{};

    Swapchain swapchain{};

    std::atomic_bool running{true};

    // Set once the device is lost or stops responding. Nothing recovers from it in-process: main shows the user a
    // restart notice and exits, instead of crashing or hanging.
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
