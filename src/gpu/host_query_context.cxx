#include "gpu/host_query_context.hxx"

#include "core/logger.hxx"
#include "gpu/context.hxx"

auto HostQueryContext::initialize([[maybe_unused]] VulkanContext &vulkan_context, [[maybe_unused]] VkQueue queue,
                                  [[maybe_unused]] std::uint32_t family,
                                  [[maybe_unused]] std::string_view name) -> void {
    if (vulkan_context.host_calibrated_timestamps_supported) {
        context = TracyVkContextHostCalibrated(vulkan_context.physical_device, vulkan_context.device, vkResetQueryPool,
                                               vkGetPhysicalDeviceCalibrateableTimeDomainsEXT,
                                               vkGetCalibratedTimestampsEXT);

        if (context != nullptr) {
            TracyVkContextName(context, name.data(), static_cast<std::uint16_t>(name.size()));

            return;
        }

        warn("Host-calibrated Tracy Vulkan context creation failed for the {} queue; falling back", name);
    }

    // The calibrated fallbacks record and submit one command buffer on `queue` while the context is created. The
    // graphics queue has one ready; any other family gets a short-lived pool.
    auto one_time_buffer = vulkan_context.one_time_command_buffers[0];
    VkCommandPool temporary_pool = VK_NULL_HANDLE;

    if (family != vulkan_context.queue_families.graphics) {
        VkCommandPoolCreateInfo const pool_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = family,
        };
        VkCommandBufferAllocateInfo allocate_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = VK_NULL_HANDLE,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
        };

        if (vkCreateCommandPool(vulkan_context.device, &pool_info, nullptr, &temporary_pool) != VK_SUCCESS) {
            warn("Could not create a command pool for the {} Tracy context", name);

            return;
        }

        allocate_info.commandPool = temporary_pool;

        if (vkAllocateCommandBuffers(vulkan_context.device, &allocate_info, &one_time_buffer) != VK_SUCCESS) {
            warn("Could not allocate a command buffer for the {} Tracy context", name);
            vkDestroyCommandPool(vulkan_context.device, temporary_pool, nullptr);

            return;
        }
    }

    if (vulkan_context.calibrated_timestamps_supported) {
        context =
                TracyVkContextCalibrated(vulkan_context.physical_device, vulkan_context.device, queue, one_time_buffer,
                                         vkGetPhysicalDeviceCalibrateableTimeDomainsEXT, vkGetCalibratedTimestampsEXT);
    } else {
        context = TracyVkContext(vulkan_context.physical_device, vulkan_context.device, queue, one_time_buffer);
    }

    if (temporary_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(vulkan_context.device, temporary_pool, nullptr);
    }

    if (context != nullptr) {
        TracyVkContextName(context, name.data(), static_cast<std::uint16_t>(name.size()));
    }
}

auto HostQueryContext::destroy() -> void {
    if (context != nullptr) {
        TracyVkDestroy(context);
        context = nullptr;
    }
}
