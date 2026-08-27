#include "context.hxx"

auto VulkanContext::destroy() -> void {
    auto &context = *this;

    context.swapchain.destroy();

    if (context.one_time_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(context.device, context.one_time_pool, nullptr);

        context.one_time_pool = VK_NULL_HANDLE;
        context.one_time_command_buffers.fill(VK_NULL_HANDLE);
    }

    if (context.allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(context.allocator);

        context.allocator = VK_NULL_HANDLE;
    }

    if (context.device != VK_NULL_HANDLE) {
        vkDestroyDevice(context.device, nullptr);

        context.device = VK_NULL_HANDLE;
    }

    if (context.surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(context.instance, context.surface, nullptr);

        context.surface = VK_NULL_HANDLE;
    }

    if (context.debug_messenger != VK_NULL_HANDLE) {
        vkDestroyDebugUtilsMessengerEXT(context.instance, context.debug_messenger, nullptr);

        context.debug_messenger = VK_NULL_HANDLE;
    }

    if (context.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(context.instance, nullptr);

        context.instance = VK_NULL_HANDLE;
    }

    if (context.window != nullptr) {
        glfwDestroyWindow(context.window);

        context.window = nullptr;
    }

    glfwTerminate();
}

auto VulkanContext::one_time_submit(std::function<void(VkCommandBuffer)> &&func) -> void {
    auto &buf = one_time_command_buffers.at(one_time_buffer_index);

    vkResetCommandBuffer(buf, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(buf, &begin_info);

    func(buf);

    vkEndCommandBuffer(buf);

    VkFence fence;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fence_info, nullptr, &fence);

    VkCommandBufferSubmitInfo cmd_info{};
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmd_info.commandBuffer = buf;

    VkSubmitInfo2 submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit_info.commandBufferInfoCount = 1;
    submit_info.pCommandBufferInfos = &cmd_info;

    vkQueueSubmit2(graphics_queue, 1, &submit_info, fence);

    vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());

    vkDestroyFence(device, fence, nullptr);

    const auto size = static_cast<std::uint32_t>(one_time_command_buffers.size());
    one_time_buffer_index = ((one_time_buffer_index + 1) % size);
}
