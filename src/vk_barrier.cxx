#include "vk_barrier.hxx"

auto transition_image_layout(VkCommandBuffer command_buffer, VkImage image, VkImageLayout old_layout,
                             VkImageLayout new_layout, VkPipelineStageFlags2 src_stage_mask,
                             VkPipelineStageFlags2 dst_stage_mask, VkAccessFlags2 src_access_mask,
                             VkAccessFlags2 dst_access_mask, VkImageAspectFlags aspect_mask,
                             std::uint32_t base_mip_level, std::uint32_t level_count) noexcept -> void {
    VkImageMemoryBarrier2 const barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = src_stage_mask,
            .srcAccessMask = src_access_mask,
            .dstStageMask = dst_stage_mask,
            .dstAccessMask = dst_access_mask,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange =
                    VkImageSubresourceRange{
                            .aspectMask = aspect_mask,
                            .baseMipLevel = base_mip_level,
                            .levelCount = level_count,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                    },
    };

    VkDependencyInfo const dependency_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier,
    };

    vkCmdPipelineBarrier2(command_buffer, &dependency_info);
}