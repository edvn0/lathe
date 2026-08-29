#include "host_query_context.hxx"

#include "context.hxx"
#include "logger.hxx"

auto HostQueryContext::initialize(VulkanContext &vulkan_context) -> void {
    if (vulkan_context.host_calibrated_timestamps_supported) {
        context = TracyVkContextHostCalibrated(vulkan_context.physical_device, vulkan_context.device, vkResetQueryPool,
                                               vkGetPhysicalDeviceCalibrateableTimeDomainsEXT,
                                               vkGetCalibratedTimestampsEXT);

        if (context != nullptr) {
            TracyVkContextName(context, "host-calibrated", sizeof("host-calibrated") - 1);

            return;
        }

        warn("Host-calibrated Tracy Vulkan context creation failed; falling back");
    }

    auto &one_time_buffer = vulkan_context.one_time_command_buffers[0];

    if (vulkan_context.calibrated_timestamps_supported) {
        context = TracyVkContextCalibrated(
                vulkan_context.physical_device, vulkan_context.device, vulkan_context.graphics_queue, one_time_buffer,
                vkGetPhysicalDeviceCalibrateableTimeDomainsEXT, vkGetCalibratedTimestampsEXT);
    } else {
        context = TracyVkContext(vulkan_context.physical_device, vulkan_context.device, vulkan_context.graphics_queue,
                                 one_time_buffer);
    }

    if (context != nullptr) {
        TracyVkContextName(context, "uncalibrated", sizeof("uncalibrated") - 1);
    }
}

auto HostQueryContext::destroy() -> void {
    if (context != nullptr) {
        TracyVkDestroy(context);
        context = nullptr;
    }
}
