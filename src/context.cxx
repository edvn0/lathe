#include "context.hxx"


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

    one_time_buffer_index = (one_time_buffer_index++ % one_time_command_buffers.size());
}
