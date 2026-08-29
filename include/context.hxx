#pragma once

#include <volk.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>

#include <GLFW/glfw3.h>

#include "allocator.hxx"
#include "host_query_context.hxx"
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

    // Whether VK_EXT_shader_object + the extended-dynamic-state3 bits it
    // needs are all present on physical_device. Decided once at device
    // selection (see select_physical_device in main.cxx) and used both to
    // gate which extensions/features create_device() enables and to pick
    // VkPipeline vs ShaderObjectSet at pipeline-registration time -- not
    // every GPU (e.g. some Intel iGPUs) implements this extension yet.
    bool shader_objects_supported = false;

    // Whether VK_EXT_calibrated_timestamps is present on physical_device,
    // and whether its calibrateable time domains additionally allow Tracy's
    // host-calibrated Vulkan context (see host_query_context.hxx). Decided
    // once at device selection in main.cxx.
    bool calibrated_timestamps_supported = false;
    bool host_calibrated_timestamps_supported = false;

    HostQueryContext host_query_context{};

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

    auto destroy() -> void;
};
