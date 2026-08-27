#pragma once

#include <volk.h>

#include <cstdint>

auto transition_image_layout(VkCommandBuffer command_buffer, VkImage image, VkImageLayout old_layout,
                             VkImageLayout new_layout, VkPipelineStageFlags2 src_stage_mask,
                             VkPipelineStageFlags2 dst_stage_mask, VkAccessFlags2 src_access_mask,
                             VkAccessFlags2 dst_access_mask, VkImageAspectFlags aspect_mask,
                             std::uint32_t base_mip_level, std::uint32_t level_count) noexcept -> void;
