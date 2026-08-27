#include "renderer.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <expected>
#include <fstream>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "buffer.hxx"
#include "context.hxx"
#include "device_error.hxx"
#include "gpu_resource_table.hxx"
#include "logger.hxx"
#include "material_storage.hxx"
#include "renderer_application_policy.hxx"
#include "sampler_storage.hxx"
#include "slang_compiler.hxx"
#include "vk_barrier.hxx"

namespace {
    template<typename Action>
    struct FinalAction {
        Action action;

        ~FinalAction() { action(); }
    };

    struct ForwardPushConstants {
        VkDeviceAddress draw_buffer_address;
        VkDeviceAddress transform_buffer_address;
        VkDeviceAddress material_buffer_address;
        VkDeviceAddress ubo_buffer_address;
        VkDeviceAddress lights_buffer_address;
        std::uint32_t light_count;
        std::uint32_t padding;
    };

    static_assert(sizeof(ForwardPushConstants) == 48);

    struct ShadowPushConstants {
        VkDeviceAddress draw_buffer_address;
        VkDeviceAddress transform_buffer_address;
        VkDeviceAddress material_buffer_address;
        VkDeviceAddress ubo_buffer_address;
        VkDeviceAddress lights_buffer_address;
        std::uint32_t light_count;
        std::uint32_t padding0;
        std::uint32_t cascade_index;
        std::uint32_t padding1;
    };

    static_assert(sizeof(ShadowPushConstants) == 56);
    static_assert(offsetof(ShadowPushConstants, cascade_index) == 48);

    struct CompositePushConstants {
        std::uint32_t hdr_texture_index;
        std::uint32_t bloom_texture_index; // <-- Added
        std::uint32_t sampler_index;
        float exposure;
        float bloom_intensity; // <-- Added
    };

    static_assert(sizeof(CompositePushConstants) == 20);

    struct LightIconPushConstants {
        VkDeviceAddress lights_buffer_address;
        VkDeviceAddress ubo_buffer_address;
        std::uint32_t light_count;
        std::uint32_t icon_texture_index;
        std::uint32_t sampler_index;
        float icon_world_size;
    };

    static_assert(sizeof(LightIconPushConstants) == 32);

    // Mirrors CullPC in assets/shaders/frustum_cull.slang. One compute
    // workgroup handles one batch: it reads that batch's un-culled
    // indirect command to learn [firstInstance, firstInstance +
    // instanceCount), culls each instance, compacts survivors in place via
    // a shared-memory scan, and writes the recomputed command to
    // dst_indirect[batch]. No per-instance batch lookup or atomic counter
    // buffer is needed -- the batch index *is* the workgroup index.
    struct CullPushConstants {
        VkDeviceAddress src_draws_address;
        VkDeviceAddress src_transforms_address;
        VkDeviceAddress batch_bounds_address;
        VkDeviceAddress src_indirect_address;
        VkDeviceAddress dst_indirect_address;
        VkDeviceAddress dst_draws_address;
        VkDeviceAddress dst_transforms_address;
        VkDeviceAddress frustum_planes_address;
        std::uint32_t batch_count;
        std::uint32_t padding;
    };

    static_assert(sizeof(CullPushConstants) == 72);

    struct DownsamplePushConstants {
        std::uint32_t src_texture_index;
        std::uint32_t linear_sampler_index;
        std::uint32_t dst_mip0_storage_index;
        std::uint32_t dst_mip1_storage_index;
        std::uint32_t dst_mip2_storage_index;
        std::uint32_t dst_mip3_storage_index;
        float src_texel_size_x;
        float src_texel_size_y;
        std::int32_t mip0_size_x;
        std::int32_t mip0_size_y;
        float threshold;
        float knee;
    };
    static_assert(sizeof(DownsamplePushConstants) == 48);

    struct UpsamplePushConstants {
        std::uint32_t lower_mip_texture_index;
        std::uint32_t target_mip_storage_index;
        std::uint32_t linear_sampler_index;
        float lower_texel_size_x;
        float lower_texel_size_y;
        int target_size_x;
        int target_size_y;
        float filter_radius;
    };

    static_assert(sizeof(UpsamplePushConstants) == 32);


    // Pins VkDrawIndexedIndirectCommand's ABI layout against IndirectCommand
    // in assets/shaders/frustum_cull.slang, which mirrors it field-for-field
    // (both sides: indexCount, instanceCount, firstIndex, vertexOffset,
    // firstInstance -- 4 x uint32 + 1 x int32, tightly packed, no padding).
    // The Vulkan spec guarantees this layout for the host-side struct; nothing
    // enforced the Slang side matching it before this, so pin it explicitly.
    static_assert(sizeof(VkDrawIndexedIndirectCommand) == 20);
    static_assert(offsetof(VkDrawIndexedIndirectCommand, indexCount) == 0);
    static_assert(offsetof(VkDrawIndexedIndirectCommand, instanceCount) == 4);
    static_assert(offsetof(VkDrawIndexedIndirectCommand, firstIndex) == 8);
    static_assert(offsetof(VkDrawIndexedIndirectCommand, vertexOffset) == 12);
    static_assert(offsetof(VkDrawIndexedIndirectCommand, firstInstance) == 16);
} // namespace

namespace {
    auto transition_forward_target_to_attachments(VkCommandBuffer command_buffer, Image const &hdr, Image const &depth,
                                                  Image const *resolved_hdr, Image const *resolved_depth) noexcept
            -> void {
        std::array<VkImageMemoryBarrier2, 4> barriers{};
        std::uint32_t barrier_count = 0;

        barriers[barrier_count++] = VkImageMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = hdr.image(),
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = hdr.mip_levels(),
                                .baseArrayLayer = 0,
                                .layerCount = hdr.array_layers(),
                        },
        };

        barriers[barrier_count++] = VkImageMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask =
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                .dstAccessMask =
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = depth.image(),
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                .baseMipLevel = 0,
                                .levelCount = depth.mip_levels(),
                                .baseArrayLayer = 0,
                                .layerCount = depth.array_layers(),
                        },
        };

        if (resolved_hdr != nullptr) {
            barriers[barrier_count++] = VkImageMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                    .srcAccessMask = VK_ACCESS_2_NONE,
                    .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = resolved_hdr->image(),
                    .subresourceRange =
                            VkImageSubresourceRange{
                                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                    .baseMipLevel = 0,
                                    .levelCount = resolved_hdr->mip_levels(),
                                    .baseArrayLayer = 0,
                                    .layerCount = resolved_hdr->array_layers(),
                            },
            };
        }

        if (resolved_depth != nullptr) {
            barriers[barrier_count++] = VkImageMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                    .srcAccessMask = VK_ACCESS_2_NONE,
                    .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = resolved_depth->image(),
                    .subresourceRange =
                            VkImageSubresourceRange{
                                    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                    .baseMipLevel = 0,
                                    .levelCount = resolved_depth->mip_levels(),
                                    .baseArrayLayer = 0,
                                    .layerCount = resolved_depth->array_layers(),
                            },
            };
        }

        VkDependencyInfo const dependency_info{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = nullptr,
                .dependencyFlags = 0,
                .memoryBarrierCount = 0,
                .pMemoryBarriers = nullptr,
                .bufferMemoryBarrierCount = 0,
                .pBufferMemoryBarriers = nullptr,
                .imageMemoryBarrierCount = barrier_count,
                .pImageMemoryBarriers = barriers.data(),
        };

        vkCmdPipelineBarrier2(command_buffer, &dependency_info);
    }

    auto transition_hdr_to_shader_read(VkCommandBuffer command_buffer, Image const &hdr) noexcept -> void {
        VkImageMemoryBarrier2 const barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = hdr.image(),
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = hdr.mip_levels(),
                                .baseArrayLayer = 0,
                                .layerCount = hdr.array_layers(),
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

    auto transition_shadow_atlas_to_attachment(VkCommandBuffer command_buffer, Image const &atlas) noexcept -> void {
        VkImageMemoryBarrier2 const barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                .dstStageMask =
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                .dstAccessMask =
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = atlas.image(),
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                .baseMipLevel = 0,
                                .levelCount = atlas.mip_levels(),
                                .baseArrayLayer = 0,
                                .layerCount = atlas.array_layers(),
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

    auto transition_shadow_atlas_to_shader_read(VkCommandBuffer command_buffer, Image const &atlas) noexcept -> void {
        VkImageMemoryBarrier2 const barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = atlas.image(),
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                .baseMipLevel = 0,
                                .levelCount = atlas.mip_levels(),
                                .baseArrayLayer = 0,
                                .layerCount = atlas.array_layers(),
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

    auto transition_swapchain_to_attachment(VkCommandBuffer command_buffer, VkImage image) noexcept -> void {
        VkImageMemoryBarrier2 const barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
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

    auto transition_swapchain_to_present(VkCommandBuffer command_buffer, VkImage image) noexcept -> void {
        VkImageMemoryBarrier2 const barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
                .dstAccessMask = VK_ACCESS_2_NONE,
                .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
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

    // main: opaque/mask forward draws (depth write on, EQUAL against the
    // prepass-written depth). blend has no prepass contribution, so it
    // tests GREATER_OR_EQUAL (reverse-Z) against existing depth without
    // expecting an exact match, and must not write depth itself.
    enum class ForwardDynamicStateMode : std::uint8_t {
        prepass,
        main,
        blend,
    };

    auto set_forward_dynamic_state(VkCommandBuffer command_buffer, VkExtent2D extent,
                                   ForwardDynamicStateMode mode) noexcept -> void {
        VkViewport const viewport{
                .x = 0.0F,
                .y = static_cast<float>(extent.height),
                .width = static_cast<float>(extent.width),
                .height = -static_cast<float>(extent.height),
                .minDepth = 1.0F,
                .maxDepth = 0.0F,
        };

        VkRect2D const scissor{
                .offset = {0, 0},
                .extent = extent,
        };

        vkCmdSetViewportWithCount(command_buffer, 1, &viewport);
        vkCmdSetScissorWithCount(command_buffer, 1, &scissor);

        vkCmdSetPrimitiveTopology(command_buffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        vkCmdSetPrimitiveRestartEnable(command_buffer, VK_FALSE);

        vkCmdSetRasterizerDiscardEnable(command_buffer, VK_FALSE);

        // Default; callers issuing a mask or blend draw override this with
        // an explicit vkCmdSetCullMode(VK_CULL_MODE_NONE) afterwards.
        vkCmdSetCullMode(command_buffer, VK_CULL_MODE_BACK_BIT);

        vkCmdSetFrontFace(command_buffer, VK_FRONT_FACE_CLOCKWISE);

        vkCmdSetDepthTestEnable(command_buffer, VK_TRUE);

        vkCmdSetDepthWriteEnable(command_buffer, mode == ForwardDynamicStateMode::blend ? VK_FALSE : VK_TRUE);

        vkCmdSetDepthCompareOp(command_buffer, mode == ForwardDynamicStateMode::main ? VK_COMPARE_OP_EQUAL
                                                                                     : VK_COMPARE_OP_GREATER_OR_EQUAL);

        vkCmdSetDepthBiasEnable(command_buffer, VK_FALSE);

        vkCmdSetStencilTestEnable(command_buffer, VK_FALSE);
    }

    // Plain (non-inverted) viewport: reverse-Z here is baked into
    // cascade_view_projection via a swapped near/far in the orthographic
    // projection, not via the viewport depth range the main pass uses.
    // Doing both would cancel out. See shadow_cascades.cxx.
    auto set_shadow_dynamic_state(VkCommandBuffer command_buffer, std::uint32_t cascade, float depth_bias_constant,
                                  float depth_bias_slope) noexcept -> void {
        auto const resolution = shadow_cascade_resolutions[cascade];
        auto const offset_x = shadow_cascade_offset_x[cascade];

        VkViewport const viewport{
                .x = static_cast<float>(offset_x),
                .y = 0.0F,
                .width = static_cast<float>(resolution),
                .height = static_cast<float>(resolution),
                .minDepth = 0.0F,
                .maxDepth = 1.0F,
        };

        VkRect2D const scissor{
                .offset = {static_cast<std::int32_t>(offset_x), 0},
                .extent = {resolution, resolution},
        };

        vkCmdSetViewportWithCount(command_buffer, 1, &viewport);
        vkCmdSetScissorWithCount(command_buffer, 1, &scissor);

        vkCmdSetPrimitiveTopology(command_buffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        vkCmdSetPrimitiveRestartEnable(command_buffer, VK_FALSE);

        vkCmdSetRasterizerDiscardEnable(command_buffer, VK_FALSE);

        // No Y flip in this pass inverts framebuffer-space winding relative
        // to the main pass, so front-face culling would cull the wrong side.
        // CULL_MODE_NONE sidesteps the question -- the standard robust
        // choice for shadow casters.
        vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);

        vkCmdSetFrontFace(command_buffer, VK_FRONT_FACE_CLOCKWISE);

        vkCmdSetDepthTestEnable(command_buffer, VK_TRUE);

        vkCmdSetDepthWriteEnable(command_buffer, VK_TRUE);

        vkCmdSetDepthCompareOp(command_buffer, VK_COMPARE_OP_GREATER_OR_EQUAL);

        vkCmdSetDepthBiasEnable(command_buffer, VK_TRUE);

        // Negative: bias adds to the fragment's depth, and in reverse-Z
        // "farther from the light" is smaller.
        vkCmdSetDepthBias(command_buffer, depth_bias_constant, 0.0F, depth_bias_slope);

        vkCmdSetStencilTestEnable(command_buffer, VK_FALSE);
    }

    auto set_composite_dynamic_state(VkCommandBuffer command_buffer, VkExtent2D extent) noexcept -> void {
        VkViewport const viewport{
                .x = 0.0F,
                .y = 0.0F,
                .width = static_cast<float>(extent.width),
                .height = static_cast<float>(extent.height),
                .minDepth = 1.0F,
                .maxDepth = 0.0F,
        };

        VkRect2D const scissor{
                .offset = {0, 0},
                .extent = extent,
        };

        vkCmdSetViewportWithCount(command_buffer, 1, &viewport);
        vkCmdSetScissorWithCount(command_buffer, 1, &scissor);

        vkCmdSetPrimitiveTopology(command_buffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        vkCmdSetPrimitiveRestartEnable(command_buffer, VK_FALSE);

        vkCmdSetRasterizerDiscardEnable(command_buffer, VK_FALSE);

        vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);

        vkCmdSetFrontFace(command_buffer, VK_FRONT_FACE_COUNTER_CLOCKWISE);

        vkCmdSetDepthTestEnable(command_buffer, VK_FALSE);

        vkCmdSetDepthWriteEnable(command_buffer, VK_FALSE);

        vkCmdSetDepthCompareOp(command_buffer, VK_COMPARE_OP_ALWAYS);

        vkCmdSetDepthBiasEnable(command_buffer, VK_FALSE);

        vkCmdSetStencilTestEnable(command_buffer, VK_FALSE);
    }

    // --- VK_EXT_shader_object dynamic-state additions --------------------
    //
    // Phase 3 of docs/pipeline_to_shader_objects.md: state that used to be
    // baked into a VkGraphicsPipelineCreateInfo (vertex input, blend,
    // rasterization samples/mask, polygon mode, depth clamp, logic op)
    // becomes per-draw dynamic state once a pass binds a ShaderObjectSet
    // instead of a Pipeline. These helpers compile against the
    // extended-dynamic-state3 / vertex-input-dynamic-state function
    // pointers ahead of time; none are called yet -- Phase 5 wires them
    // into the per-pass call sites once each pass migrates off VkPipeline.

    // Mirrors default_vertex_description() (load_model.hxx) in the
    // VK_EXT_vertex_input_dynamic_state shape.
    auto default_shader_object_vertex_input() noexcept
            -> std::pair<std::array<VkVertexInputBindingDescription2EXT, 1>,
                         std::array<VkVertexInputAttributeDescription2EXT, 4>> {
        std::array<VkVertexInputBindingDescription2EXT, 1> bindings{};
        bindings[0] = VkVertexInputBindingDescription2EXT{
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .binding = 0,
                .stride = sizeof(ModelVertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
                .divisor = 1,
        };

        std::array<VkVertexInputAttributeDescription2EXT, 4> attributes{};

        attributes[0] = VkVertexInputAttributeDescription2EXT{
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(ModelVertex, position),
        };
        attributes[1] = VkVertexInputAttributeDescription2EXT{
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(ModelVertex, normal),
        };
        attributes[2] = VkVertexInputAttributeDescription2EXT{
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = offsetof(ModelVertex, tangent),
        };
        attributes[3] = VkVertexInputAttributeDescription2EXT{
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .location = 3,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(ModelVertex, texcoord),
        };

        return {bindings, attributes};
    }

    auto set_shader_object_vertex_input(VkCommandBuffer command_buffer,
                                        std::span<VkVertexInputBindingDescription2EXT const> bindings,
                                        std::span<VkVertexInputAttributeDescription2EXT const> attributes) noexcept
            -> void {
        vkCmdSetVertexInputEXT(command_buffer, static_cast<std::uint32_t>(bindings.size()), bindings.data(),
                               static_cast<std::uint32_t>(attributes.size()), attributes.data());
    }

    auto set_shader_object_raster_state(VkCommandBuffer command_buffer, VkPolygonMode polygon_mode,
                                        VkSampleCountFlagBits samples, bool depth_clamp_enable) noexcept -> void {
        vkCmdSetPolygonModeEXT(command_buffer, polygon_mode);

        vkCmdSetRasterizationSamplesEXT(command_buffer, samples);

        VkSampleMask const sample_mask = 0xFFFFFFFFU;
        vkCmdSetSampleMaskEXT(command_buffer, samples, &sample_mask);

        vkCmdSetAlphaToCoverageEnableEXT(command_buffer, VK_FALSE);

        vkCmdSetDepthClampEnableEXT(command_buffer, depth_clamp_enable ? VK_TRUE : VK_FALSE);

        vkCmdSetLogicOpEnableEXT(command_buffer, VK_FALSE);
    }

    // Replaces GraphicsPipelineCreateInfo::blending. attachment_count must
    // match VkPipelineRenderingCreateInfo::colorAttachmentCount for the
    // current render pass -- pass 0 for the depth prepass and shadow pass
    // (no color attachments), 1 for forward/composite (today's only
    // multi-attachment case would need one entry per attachment here).
    auto set_shader_object_color_blend_state(VkCommandBuffer command_buffer, std::uint32_t attachment_count,
                                             bool blending) noexcept -> void {
        if (attachment_count == 0) {
            return;
        }

        std::vector<VkBool32> const blend_enable(attachment_count, blending ? VK_TRUE : VK_FALSE);

        std::vector<VkColorBlendEquationEXT> const blend_equation(
                attachment_count, VkColorBlendEquationEXT{
                                          .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                                          .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                          .colorBlendOp = VK_BLEND_OP_ADD,
                                          .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                          .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                          .alphaBlendOp = VK_BLEND_OP_ADD,
                                  });

        std::vector<VkColorComponentFlags> const write_mask(
                attachment_count, VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                          VK_COLOR_COMPONENT_A_BIT);

        vkCmdSetColorBlendEnableEXT(command_buffer, 0, attachment_count, blend_enable.data());

        vkCmdSetColorBlendEquationEXT(command_buffer, 0, attachment_count, blend_equation.data());

        vkCmdSetColorWriteMaskEXT(command_buffer, 0, attachment_count, write_mask.data());
    }

    // Task/mesh ShaderObjectSets have no vertex input stage, so this skips
    // set_shader_object_vertex_input entirely rather than calling it with
    // empty spans -- confirm against validation layer output during Phase 5
    // whether omitting the call is actually required by spec or just
    // equivalent to an empty VkVertexInputBindingDescription2EXT list.
    auto set_mesh_shader_object_dynamic_state(VkCommandBuffer command_buffer, VkExtent2D extent,
                                              VkSampleCountFlagBits samples) noexcept -> void {
        VkViewport const viewport{
                .x = 0.0F,
                .y = static_cast<float>(extent.height),
                .width = static_cast<float>(extent.width),
                .height = -static_cast<float>(extent.height),
                .minDepth = 1.0F,
                .maxDepth = 0.0F,
        };

        VkRect2D const scissor{
                .offset = {0, 0},
                .extent = extent,
        };

        vkCmdSetViewportWithCount(command_buffer, 1, &viewport);
        vkCmdSetScissorWithCount(command_buffer, 1, &scissor);

        vkCmdSetRasterizerDiscardEnable(command_buffer, VK_FALSE);

        vkCmdSetCullMode(command_buffer, VK_CULL_MODE_BACK_BIT);

        vkCmdSetFrontFace(command_buffer, VK_FRONT_FACE_CLOCKWISE);

        vkCmdSetDepthTestEnable(command_buffer, VK_TRUE);

        vkCmdSetDepthWriteEnable(command_buffer, VK_TRUE);

        vkCmdSetDepthCompareOp(command_buffer, VK_COMPARE_OP_GREATER_OR_EQUAL);

        vkCmdSetDepthBiasEnable(command_buffer, VK_FALSE);

        vkCmdSetStencilTestEnable(command_buffer, VK_FALSE);

        set_shader_object_raster_state(command_buffer, VK_POLYGON_MODE_FILL, samples, false);
    }

    // --- VkPipeline / ShaderObjectSet runtime dispatch --------------------
    //
    // VK_EXT_shader_object is optional (VulkanContext::shader_objects_supported):
    // some target GPUs (e.g. certain Intel iGPUs) don't implement it, so
    // PipelineRegisterInfo::use_shader_objects is set per-node from that
    // capability flag rather than unconditionally. Every draw call site
    // resolves through these two helpers instead of assuming one backend.

    // Returns the pipeline layout regardless of which backend a node
    // resolved to, or VK_NULL_HANDLE for an invalid/unbuilt handle. Used for
    // both the upfront per-frame validity check and vkCmdPushConstants.
    [[nodiscard]]
    auto resolve_layout(PipelineGraphRepository const &graph, PipelineNodeHandle handle) noexcept -> VkPipelineLayout {
        if (auto const *shader_objects = graph.resolve_shader_objects(handle); shader_objects != nullptr) {
            return shader_objects->layout();
        }

        if (auto const *pipeline = graph.resolve(handle); pipeline != nullptr) {
            return pipeline->layout();
        }

        return VK_NULL_HANDLE;
    }

    // Binds a graphics node and, only for the ShaderObjectSet backend, sets
    // the dynamic state a VkPipeline would already have baked in (vertex
    // input, rasterization samples/polygon-mode, color blend). Pass
    // has_vertex_input_stage = false for task/mesh shader nodes, which have
    // no vertex input stage at all.
    auto bind_graphics_node(PipelineGraphRepository const &graph, PipelineNodeHandle handle,
                            VkCommandBuffer command_buffer, VkSampleCountFlagBits samples,
                            std::uint32_t colour_attachment_count, bool blending,
                            bool has_vertex_input_stage = true) noexcept -> void {
        if (auto const *shader_objects = graph.resolve_shader_objects(handle); shader_objects != nullptr) {
            shader_objects->bind(command_buffer);

            if (has_vertex_input_stage) {
                set_shader_object_vertex_input(command_buffer, {}, {});
            }

            set_shader_object_raster_state(command_buffer, VK_POLYGON_MODE_FILL, samples, false);
            set_shader_object_color_blend_state(command_buffer, colour_attachment_count, blending);

            return;
        }

        if (auto const *pipeline = graph.resolve(handle); pipeline != nullptr) {
            vkCmdBindPipeline(command_buffer, pipeline->bind_point(), pipeline->pipeline());
        }
    }

    // Compute nodes have no rasterization state to set either way.
    auto bind_compute_node(PipelineGraphRepository const &graph, PipelineNodeHandle handle,
                           VkCommandBuffer command_buffer) noexcept -> void {
        if (auto const *shader_objects = graph.resolve_shader_objects(handle); shader_objects != nullptr) {
            shader_objects->bind(command_buffer);

            return;
        }

        if (auto const *pipeline = graph.resolve(handle); pipeline != nullptr) {
            vkCmdBindPipeline(command_buffer, pipeline->bind_point(), pipeline->pipeline());
        }
    }
} // namespace

namespace {
    auto make_error(RendererErrorType type) -> RendererError {
        return RendererError{
                .type = type,
        };
    }

    auto make_resource_table_error(GpuResourceTableError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::gpu_resource_table_error,
                .cause = ErrorCause{Boxed<GpuResourceTableError>{std::move(error)}},
        };
    }

    auto make_pipeline_graph_error(PipelineGraphError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::pipeline_graph_error,
                .cause = ErrorCause{Boxed<PipelineGraphError>{std::move(error)}},
        };
    }

    auto make_device_error(DeviceError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::device_error,
                .cause = ErrorCause{Boxed<DeviceError>{std::move(error)}},
        };
    }

    auto make_geometry_error(GeometryArenaError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::geometry_error,
                .cause = ErrorCause{Boxed<GeometryArenaError>{std::move(error)}},
        };
    }

    auto make_compiler_error(renderer::ShaderCompileError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::compiler_error,
                .cause = ErrorCause{Boxed<renderer::ShaderCompileError>{std::move(error)}},
        };
    }

    auto make_material_error(MaterialStorageError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::material_error,
                .cause = ErrorCause{Boxed<MaterialStorageError>{std::move(error)}},
        };
    }

    auto make_model_load_error(ModelLoadError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::model_load_error,
                .cause = ErrorCause{Boxed<ModelLoadError>{std::move(error)}},
        };
    }

    auto make_image_error(ImageStorageError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::image_error,
                .cause = ErrorCause{Boxed<ImageStorageError>{std::move(error)}},
        };
    }

    auto align_up(VkDeviceSize value, VkDeviceSize alignment) noexcept -> VkDeviceSize {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    auto checked_multiply(VkDeviceSize lhs, VkDeviceSize rhs) -> std::expected<VkDeviceSize, RendererError> {
        if (lhs != 0 && rhs > std::numeric_limits<VkDeviceSize>::max() / lhs) {
            return std::unexpected(make_error(RendererErrorType::size_overflow));
        }

        return lhs * rhs;
    }

    auto index_stride(VkIndexType index_type) -> std::expected<VkDeviceSize, RendererError> {
        switch (index_type) {
            case VK_INDEX_TYPE_UINT16:
                return sizeof(std::uint16_t);

            case VK_INDEX_TYPE_UINT32:
                return sizeof(std::uint32_t);

            default:
                return std::unexpected(make_error(RendererErrorType::unsupported_index_type));
        }
    }
} // namespace

Renderer::Renderer(VulkanContext &context) noexcept : context_(context) {}

auto Renderer::thread_pool() noexcept -> BS::priority_thread_pool & {
    static std::unique_ptr<BS::priority_thread_pool> thread_pool_ =
            std::make_unique<BS::priority_thread_pool>(std::thread::hardware_concurrency());
    return *thread_pool_;
}

auto Renderer::initialize(RendererCreateInfo const &create_info) -> std::expected<void, RendererError> {
    debug("[Renderer::initialize] enter");

    if (initialized_ || create_info.extent.width == 0 || create_info.extent.height == 0 || frames_in_flight == 0 ||
        create_info.material_capacity < 2 || create_info.mesh_capacity < 2 || create_info.model_capacity < 2 ||
        create_info.maximum_draw_count == 0 || create_info.maximum_submission_count == 0) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    hdr_format_ = create_info.hdr_format;
    depth_format_ = create_info.depth_format;
    samples_ = create_info.samples;
    extent_ = create_info.extent;

    auto rollback_on_failure = true;
    auto const rollback_guard = FinalAction{[this, &rollback_on_failure] {
        if (rollback_on_failure) {
            destroy();
        }
    }};

    auto maybe_compiler = renderer::SlangCompiler::create();
    if (!maybe_compiler) {
        return std::unexpected(make_compiler_error(maybe_compiler.error()));
    }

    this->compiler = std::move(*maybe_compiler);

    auto geometry_arena = GeometryArena::create(context_, GeometryArenaCreateInfo{
                                                                  .capacity = create_info.geometry_capacity,
                                                                  .debug_name = "renderer.geometry",
                                                          });

    if (!geometry_arena) {
        return std::unexpected(make_geometry_error(geometry_arena.error()));
    }

    auto material_storage = MaterialStorage::create(context_, MaterialStorageCreateInfo{
                                                                      .capacity = create_info.material_capacity,
                                                                      .debug_name = "renderer.materials",
                                                              });

    if (!material_storage) {
        return std::unexpected(make_material_error(material_storage.error()));
    }

    auto image_storage = ImageStorage::create(context_, ImageStorageCreateInfo{
                                                                .capacity = create_info.image_capacity,
                                                                .debug_name = "renderer.images",
                                                        });

    if (!image_storage) {
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    auto sampler_storage = SamplerStorage::create(context_, create_info.sampler_capacity);

    if (!sampler_storage) {
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    auto gpu_resource_table =
            GpuResourceTable::create(context_, GpuResourceTableCreateInfo{
                                                       .frames_in_flight = frames_in_flight,
                                                       .image_capacity = create_info.image_capacity,
                                                       .sampler_capacity = create_info.sampler_capacity,
                                                       .debug_name = "renderer.resources",
                                               });

    if (!gpu_resource_table) {
        return std::unexpected(make_resource_table_error(gpu_resource_table.error()));
    }

    gpu_resource_table_ = std::move(*gpu_resource_table);

    auto pipeline_graph = PipelineGraphRepository::create(
            context_, PipelineGraphCreateInfo{
                              .pipeline_capacity = create_info.pipeline_capacity,
                              .frames_in_flight = frames_in_flight,
                              .global_descriptor_set_layout = gpu_resource_table_.layout(),
                              .cache_file_path = "cache/pipeline_cache.bin",
                              .debug_name = "renderer.pipelines",
                      });

    if (!pipeline_graph) {
        return std::unexpected(make_pipeline_graph_error(pipeline_graph.error()));
    }


    pipeline_graph_ = std::move(*pipeline_graph);
    image_storage_ = std::move(*image_storage);
    sampler_storage_ = std::move(*sampler_storage);
    geometry_arena_ = std::move(*geometry_arena);
    material_storage_ = std::move(*material_storage);

    // Registers every startup pipeline in one batched, parallel call rather
    // than 9 sequential register_pipeline() calls -- see
    // docs/parallel-pipeline.md. Indices below must stay in sync with the
    // pipeline_infos.push_back() order.
    std::vector<PipelineRegisterInfo> pipeline_infos;
    pipeline_infos.reserve(9);

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/forward_geom.slang",
                                    .entry_point = "mainVs",
                                    .stage = renderer::ShaderStage::vertex,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/forward_geom.slang",
                                    .entry_point = "mainFs",
                                    .stage = renderer::ShaderStage::fragment,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {create_info.hdr_format},
            .depth_format = create_info.depth_format,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = create_info.samples,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.forward_pipeline",
    }); // index 0: forward

    // Same shaders/layout as forward, blending enabled -- see
    // PipelineRegisterInfo::blending, which pipeline.cxx turns into
    // standard alpha-over (SRC_ALPHA / ONE_MINUS_SRC_ALPHA) factors.
    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/forward_geom.slang",
                                    .entry_point = "mainVs",
                                    .stage = renderer::ShaderStage::vertex,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/forward_geom.slang",
                                    .entry_point = "mainFs",
                                    .stage = renderer::ShaderStage::fragment,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {create_info.hdr_format},
            .depth_format = create_info.depth_format,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = create_info.samples,
            .blending = true,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.forward_blend_pipeline",
    }); // index 1: forward_blend

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/light_icons.slang",
                                    .entry_point = "main_task",
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/light_icons.slang",
                                    .entry_point = "main_mesh",
                                    .stage = renderer::ShaderStage::mesh,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/light_icons.slang",
                                    .entry_point = "main_fs",
                                    .stage = renderer::ShaderStage::fragment,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {create_info.hdr_format},
            .depth_format = create_info.depth_format,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = create_info.samples,
            .blending = true,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.light_icon_pipeline",
    }); // index 2: light_icon

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/shadow_depth.slang",
                                    .entry_point = "mainVs",
                                    .stage = renderer::ShaderStage::vertex,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {},
            .depth_format = VK_FORMAT_D32_SFLOAT,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.shadow_pipeline",
    }); // index 3: shadow

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/shadow_depth.slang",
                                    .entry_point = "mainVs",
                                    .stage = renderer::ShaderStage::vertex,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/shadow_depth.slang",
                                    .entry_point = "mainFs",
                                    .stage = renderer::ShaderStage::fragment,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {},
            .depth_format = VK_FORMAT_D32_SFLOAT,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.shadow_mask_pipeline",
    }); // index 4: shadow_mask

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/depth_prepass.slang",
                                    .entry_point = "mainVs",
                                    .stage = renderer::ShaderStage::vertex,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {},
            .depth_format = create_info.depth_format,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = create_info.samples,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.depth_prepass_pipeline",
    }); // index 5: depth_prepass

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/depth_prepass.slang",
                                    .entry_point = "mainVs",
                                    .stage = renderer::ShaderStage::vertex,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/depth_prepass.slang",
                                    .entry_point = "mainFs",
                                    .stage = renderer::ShaderStage::fragment,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {},
            .depth_format = create_info.depth_format,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = create_info.samples,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.depth_prepass_mask_pipeline",
    }); // index 6: depth_prepass_mask

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/composite.slang",
                                    .entry_point = "mainVs",
                                    .stage = renderer::ShaderStage::vertex,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/composite.slang",
                                    .entry_point = "mainFs",
                                    .stage = renderer::ShaderStage::fragment,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {create_info.swapchain_format},
            .depth_format = VK_FORMAT_UNDEFINED,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.composite_pipeline",
    }); // index 7: composite

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/frustum_cull.slang",
                                    .entry_point = "mainCs",
                                    .stage = renderer::ShaderStage::compute,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {},
            .depth_format = VK_FORMAT_UNDEFINED,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.frustum_cull_pipeline",
    }); // index 8: frustum_cull


    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/bloom_downsample.slang",
                                    .entry_point = "mainCs",
                                    .stage = renderer::ShaderStage::compute,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {},
            .depth_format = VK_FORMAT_UNDEFINED,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.bloom_downsample_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/bloom_upsample.slang",
                                    .entry_point = "mainCs",
                                    .stage = renderer::ShaderStage::compute,
                                    .include_directories = {},
                                    .defines = {},
                            },
                    },
            .additional_descriptor_set_layouts = {},
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {},
            .depth_format = VK_FORMAT_UNDEFINED,
            .stencil_format = VK_FORMAT_UNDEFINED,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .use_shader_objects = context_.shader_objects_supported,
            .debug_name = "renderer.bloom_downsample_pipeline",
    });

    debug("[Renderer::initialize] calling register_pipelines_parallel with {} entries", pipeline_infos.size());
    auto registered_pipelines = pipeline_graph_.register_pipelines_parallel(compiler, pipeline_infos);
    debug("[Renderer::initialize] register_pipelines_parallel returned {} results", registered_pipelines.size());

    for (std::size_t i = 0; i < registered_pipelines.size(); ++i) {
        auto const &registered = registered_pipelines[i];

        if (!registered) {
            error("[Renderer::initialize] pipeline index {} failed to register", i);
            return std::unexpected(make_pipeline_graph_error(registered.error()));
        }
    }

    debug("[Renderer::initialize] all pipelines registered, assigning handles");

    forward_pipeline_ = *registered_pipelines[0];
    forward_blend_pipeline_ = *registered_pipelines[1];
    light_icon_pipeline_ = *registered_pipelines[2];
    shadow_pipeline_ = *registered_pipelines[3];
    shadow_mask_pipeline_ = *registered_pipelines[4];
    depth_prepass_pipeline_ = *registered_pipelines[5];
    depth_prepass_mask_pipeline_ = *registered_pipelines[6];
    composite_pipeline_ = *registered_pipelines[7];
    frustum_cull_pipeline_ = *registered_pipelines[8];
    bloom_downsample_pipeline_ = *registered_pipelines[9];
    bloom_upsample_pipeline_ = *registered_pipelines[10];

    {
        auto light_icon_image = DecodedImage::load_from_file("assets/textures/light_bulb.png");
        if (!light_icon_image) {
            return std::unexpected(make_error(RendererErrorType::device_error));
        }
        auto light_icon_texture = image_storage_.create_image(
                ImageCreateInfo{
                        .extent = VkExtent3D{.width = light_icon_image->width(),
                                             .height = light_icon_image->height(),
                                             .depth = 1},
                        .format = VK_FORMAT_R8G8B8A8_UNORM,
                        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                        .image_type = VK_IMAGE_TYPE_2D,
                        .view_type = VK_IMAGE_VIEW_TYPE_2D,
                        .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                        .samples = VK_SAMPLE_COUNT_1_BIT,
                        .tiling = VK_IMAGE_TILING_OPTIMAL,
                        .mip_levels = 1,
                        .array_layers = 1,
                        .debug_name = "renderer.light_icon_texture",
                },
                light_icon_image->span());

        if (!light_icon_texture) {
            return std::unexpected(make_image_error(light_icon_texture.error()));
        }

        light_icon_texture_ = *light_icon_texture;
    }

    auto const white = image_storage_.white();
    auto const flat_normal = image_storage_.flat_normal();
    auto const metallic_roughness = image_storage_.metallic_roughness();
    auto const occlusion = image_storage_.occlusion();
    auto const emissive = image_storage_.emissive();

    constexpr auto def_mat = MaterialHandle{0, 1};
    const GpuMaterial mat{
            .base_colour_factor =
                    {
                            1.0F,
                            1.0F,
                            1.0F,
                            1.0F,
                    },
            .emissive_factor =
                    {
                            0.0F,
                            0.0F,
                            0.0F,
                    },
            .emissive_strength = 1.0F,
            .metallic_factor = 0.0F,
            .roughness_factor = 1.0F,
            .normal_scale = 1.0F,
            .occlusion_strength = 1.0F,
            .base_colour_texture = white.index,
            .normal_texture = flat_normal.index,
            .metallic_roughness_texture = metallic_roughness.index,
            .occlusion_texture = occlusion.index,
            .emissive_texture = emissive.index,
            .sampler_index = 0,
            .alpha_mode = AlphaMode::opaque,
            .alpha_cutoff = 0.5F,
    };

    if (!material_storage_.update_material(def_mat, mat)) {
        error("Could not update default material");
        return std::unexpected(RendererError{
                .type = RendererErrorType::material_error,
        });
    }

    default_material_handle_ = def_mat;

    meshes_.resize(create_info.mesh_capacity);

    for (std::uint32_t index = 1; index < create_info.mesh_capacity; ++index) {
        meshes_[index].next_free = index + 1 < create_info.mesh_capacity ? index + 1 : 0;
    }

    mesh_free_head_ = 1;

    models_.resize(create_info.model_capacity);

    for (std::uint32_t index = 1; index < create_info.model_capacity; ++index) {
        models_[index].next_free = index + 1 < create_info.model_capacity ? index + 1 : 0;
    }

    model_free_head_ = 1;
    maximum_draw_count_ = create_info.maximum_draw_count;
    maximum_submission_count_ = create_info.maximum_submission_count;
    submissions_.reserve(maximum_submission_count_);
    model_submissions_.reserve(maximum_submission_count_);
    auto draw_size_result = checked_multiply(sizeof(GpuDraw), maximum_draw_count_);
    auto transform_size_result = checked_multiply(sizeof(glm::mat4), maximum_submission_count_);
    auto indirect_size_result = checked_multiply(sizeof(VkDrawIndexedIndirectCommand), maximum_draw_count_);
    auto batch_bounds_size_result = checked_multiply(sizeof(GpuCullBounds), maximum_draw_count_);

    if (!draw_size_result || !transform_size_result || !indirect_size_result || !batch_bounds_size_result) {
        return std::unexpected(make_error(RendererErrorType::size_overflow));
    }

    auto const draw_size = *draw_size_result;
    auto const transform_size = *transform_size_result;
    auto const indirect_size = *indirect_size_result;
    auto const culled_indirect_size = indirect_size;
    auto const batch_bounds_size = *batch_bounds_size_result;
    auto const transform_offset = align_up(draw_size, 16);
    auto const indirect_offset = align_up(transform_offset + transform_size, 16);

    if (indirect_offset > std::numeric_limits<VkDeviceSize>::max() - indirect_size) {
        return std::unexpected(make_error(RendererErrorType::size_overflow));
    }

    auto const batch_bounds_offset = align_up(indirect_offset + indirect_size, 16);

    if (batch_bounds_offset > std::numeric_limits<VkDeviceSize>::max() - batch_bounds_size) {
        return std::unexpected(make_error(RendererErrorType::size_overflow));
    }

    auto const upload_size = batch_bounds_offset + batch_bounds_size;

    frames_.resize(frames_in_flight);

    for (std::uint32_t frame_index = 0; frame_index < static_cast<std::uint32_t>(frames_.size()); ++frame_index) {
        auto &frame = frames_[frame_index];

        auto upload = Buffer::create(context_, BufferCreateInfo{
                                                       .size = upload_size,
                                                       .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                       .memory = BufferMemory::upload,
                                                       .debug_name = "renderer.frame_upload",
                                               });

        if (!upload) {
            return std::unexpected(make_device_error(upload.error()));
        }

        frame.upload_buffer = std::move(*upload);

        auto draws = Buffer::create(context_, BufferCreateInfo{
                                                      .size = draw_size,
                                                      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                      .memory = BufferMemory::device,
                                                      .debug_name = "renderer.frame_draws",
                                              });

        if (!draws) {
            return std::unexpected(make_device_error(draws.error()));
        }

        frame.draw_buffer = std::move(*draws);

        auto transforms = Buffer::create(context_, BufferCreateInfo{
                                                           .size = transform_size,
                                                           .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                           .memory = BufferMemory::device,
                                                           .debug_name = "renderer.frame_transforms",
                                                   });

        if (!transforms) {
            return std::unexpected(make_device_error(transforms.error()));
        }

        frame.transform_buffer = std::move(*transforms);

        auto indirect = Buffer::create(
                context_,
                BufferCreateInfo{
                        .size = indirect_size,
                        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        .memory = BufferMemory::device,
                        .debug_name = "renderer.frame_indirect",
                });

        if (!indirect) {
            return std::unexpected(make_device_error(indirect.error()));
        }

        frame.indirect_buffer = std::move(*indirect);

        auto batch_bounds = Buffer::create(context_, BufferCreateInfo{
                                                             .size = batch_bounds_size,
                                                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                             .memory = BufferMemory::device,
                                                             .debug_name = "renderer.frame_batch_bounds",
                                                     });

        if (!batch_bounds) {
            return std::unexpected(make_device_error(batch_bounds.error()));
        }

        frame.batch_bounds_buffer = std::move(*batch_bounds);

        // No TRANSFER_DST: unlike indirect_buffer above, this is never
        // host-seeded -- mainCs is its sole writer, one thread (lane 0)
        // per batch, no atomics needed.
        auto culled_indirect = Buffer::create(context_, BufferCreateInfo{
                                                                .size = culled_indirect_size,
                                                                .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                                .memory = BufferMemory::device,
                                                                .debug_name = "renderer.frame_culled_indirect",
                                                        });

        if (!culled_indirect) {
            return std::unexpected(make_device_error(culled_indirect.error()));
        }

        frame.culled_indirect_buffer = std::move(*culled_indirect);

        auto visible_draws = Buffer::create(context_, BufferCreateInfo{
                                                              .size = draw_size,
                                                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                              .memory = BufferMemory::device,
                                                              .debug_name = "renderer.frame_visible_draws",
                                                      });

        if (!visible_draws) {
            return std::unexpected(make_device_error(visible_draws.error()));
        }

        frame.visible_draw_buffer = std::move(*visible_draws);

        auto visible_transforms = Buffer::create(context_, BufferCreateInfo{
                                                                   .size = transform_size,
                                                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                                   .memory = BufferMemory::device,
                                                                   .debug_name = "renderer.frame_visible_transforms",
                                                           });

        if (!visible_transforms) {
            return std::unexpected(make_device_error(visible_transforms.error()));
        }

        frame.visible_transform_buffer = std::move(*visible_transforms);

        // BufferMemory::upload, not ::device: this is written every frame
        // by a host memcpy (Buffer::write in prepare_frame below), never by
        // the GPU, so its memory class should say so honestly. It happened
        // to work under BufferMemory::device too, because every memory
        // class here requests VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT (see
        // make_allocation_create_info in buffer.cxx) -- but that's an
        // accident of this allocator's current policy, not a guarantee.
        auto frustum_planes_buffer =
                Buffer::create(context_, BufferCreateInfo{
                                                 .size = sizeof(glm::vec4) * 6,
                                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                 .memory = BufferMemory::upload,
                                                 .debug_name = "renderer.frame_frustum_planes",
                                         });

        if (!frustum_planes_buffer) {
            return std::unexpected(make_device_error(frustum_planes_buffer.error()));
        }

        frame.frustum_planes_buffer = std::move(*frustum_planes_buffer);

        auto lights_buffer = Buffer::create(context_, BufferCreateInfo{
                                                              .size = sizeof(GpuLight) * maximum_light_count,
                                                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                              .memory = BufferMemory::upload,
                                                              .debug_name = "renderer.frame_lights",
                                                      });

        if (!lights_buffer) {
            return std::unexpected(make_device_error(lights_buffer.error()));
        }

        frame.lights_buffer = std::move(*lights_buffer);

        const auto target_name = std::format("renderer.forward_target_{}", frame_index);
        auto forward_target = ForwardTarget::create(image_storage_, ForwardTargetCreateInfo{
                                                                            .extent = create_info.extent,
                                                                            .hdr_format = create_info.hdr_format,
                                                                            .depth_format = create_info.depth_format,
                                                                            .samples = create_info.samples,
                                                                            .debug_name = target_name,
                                                                    });

        if (!forward_target) {
            return std::unexpected(RendererError{
                    .type = RendererErrorType::forward_target_error,
                    .cause = ErrorCause{Boxed<ForwardTargetError>{forward_target.error()}},
            });
        }

        frame.forward_target = std::move(*forward_target);

        const auto shadow_atlas_name = std::format("renderer.shadow_atlas_{}", frame_index);
        auto shadow_atlas = image_storage_.create_image(ImageCreateInfo{
                .extent =
                        VkExtent3D{
                                .width = shadow_atlas_width,
                                .height = shadow_atlas_height,
                                .depth = 1,
                        },
                .format = VK_FORMAT_D32_SFLOAT,
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                .image_type = VK_IMAGE_TYPE_2D,
                .view_type = VK_IMAGE_VIEW_TYPE_2D,
                .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                .flags = 0,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .mip_levels = 1,
                .array_layers = 1,
                .debug_name = shadow_atlas_name,
        });

        if (!shadow_atlas) {
            return std::unexpected(make_image_error(shadow_atlas.error()));
        }

        auto const bloom_target_name = std::format("renderer.bloom_target_{}", frame_index);

        auto bloom_image = image_storage_.create_image(ImageCreateInfo{
                .extent =
                        VkExtent3D{
                                .width = create_info.extent.width / 2,
                                .height = create_info.extent.height / 2,
                                .depth = 1,
                        },
                .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                .image_type = VK_IMAGE_TYPE_2D,
                .view_type = VK_IMAGE_VIEW_TYPE_2D,
                .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d) |
                                    image_descriptor_view_bit(ImageDescriptorView::storage_2d),
                .flags = 0,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .mip_levels = 4,
                .array_layers = 1,
                .create_mip_layer_views = true,
                .debug_name = bloom_target_name,
        });

        if (!bloom_image) {
            return std::unexpected(make_image_error(bloom_image.error()));
        }

        RendererFrame::BloomTarget bloom_target{.image = *bloom_image};

        auto const *bloom_image_ptr = image_storage_.get(*bloom_image);

        for (std::uint32_t mip = 0; mip < 4; ++mip) {
            auto const view = bloom_image_ptr->mip_layer_view(mip, 0);

            auto mip_slot = image_storage_.register_view(ImageViewRegistration{
                    .sampled_2d = view,
                    .storage_2d = view,
            });

            if (!mip_slot) {
                return std::unexpected(make_image_error(mip_slot.error()));
            }

            bloom_target.mip_slots[mip] = *mip_slot;
        }

        frame.shadow_atlas = *shadow_atlas;
        frame.bloom_target = bloom_target;

        frame.shadow_atlas = *shadow_atlas;
        frame.draw_upload_offset = 0;
        frame.transform_upload_offset = transform_offset;
        frame.indirect_upload_offset = indirect_offset;
        frame.batch_bounds_upload_offset = batch_bounds_offset;
        frame.draws.reserve(maximum_draw_count_);
        frame.transforms.reserve(maximum_submission_count_);
        frame.indirect_commands.reserve(maximum_draw_count_);
        frame.batch_bounds.reserve(maximum_draw_count_);
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context_.physical_device, &properties);

    if (!properties.limits.timestampComputeAndGraphics) {
        error("Physical device does not support timestamps on graphics/compute queues!");
    }

    this->timestamp_period_ = properties.limits.timestampPeriod;

    timestamp_queries_.resize(frames_in_flight);

    VkQueryPoolCreateInfo query_pool_info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = static_cast<std::uint32_t>(RenderStage::Count) * 2,
            .pipelineStatistics = 0,
    };

    for (std::uint32_t frame_index = 0; frame_index < frames_in_flight; ++frame_index) {
        VkQueryPool query_pool = VK_NULL_HANDLE;
        VkResult result = vkCreateQueryPool(context_.device, &query_pool_info, nullptr, &query_pool);

        if (result != VK_SUCCESS) {
            error("Failed to create timestamp query pool for frame index {}", frame_index);
            return std::unexpected(make_error(RendererErrorType::device_error));
        }

        timestamp_queries_[frame_index] = FrameTimestamps{.query_pool = query_pool, .has_results = false};
        vkResetQueryPool(context_.device, query_pool, 0, query_count);
    }

    pipeline_stat_queries_.resize(frames_in_flight);
    VkQueryPoolCreateInfo const pipeline_stat_pool_info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS,
            .queryCount = 1,
            .pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
                                  VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
                                  VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
                                  VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT,
    };

    for (std::uint32_t frame_index = 0; frame_index < frames_in_flight; ++frame_index) {
        VkQueryPool query_pool = VK_NULL_HANDLE;
        VkResult const result = vkCreateQueryPool(context_.device, &pipeline_stat_pool_info, nullptr, &query_pool);

        if (result != VK_SUCCESS) {
            error("Failed to create pipeline statistics query pool for frame index {}", frame_index);
            return std::unexpected(make_error(RendererErrorType::device_error));
        }

        pipeline_stat_queries_[frame_index] = FramePipelineQuery{.query_pool = query_pool, .has_results = false};
        vkResetQueryPool(context_.device, query_pool, 0, 1);
    }

    ubos_.resize(frames_in_flight);
    for (auto &ubo: ubos_) {
        auto maybe_ubo = Buffer::create(context_, BufferCreateInfo{
                                                          .size = sizeof(UBO),
                                                          .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                          .memory = BufferMemory::upload,
                                                          .debug_name = "renderer.ubo",
                                                  });
        if (!maybe_ubo) {
            error("Failed to create ubo");
            return std::unexpected(make_error(RendererErrorType::device_error));
        }

        ubo = std::move(*maybe_ubo);
    }

    mark_lights_dirty();

    rollback_on_failure = false;
    initialized_ = true;

    debug("[Renderer::initialize] exit: success");

    return {};
}

auto Renderer::destroy() noexcept -> void {
    debug("[Renderer::destroy] enter");

    pipeline_graph_.save_pipeline_cache();
    pipeline_graph_.destroy();
    gpu_resource_table_.destroy();
    sampler_storage_.destroy();

    for (auto &query: timestamp_queries_) {
        if (query.query_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(context_.device, query.query_pool, nullptr);
            query.query_pool = VK_NULL_HANDLE;
        }
    }

    for (auto &query: pipeline_stat_queries_) {
        if (query.query_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(context_.device, query.query_pool, nullptr);
            query.query_pool = VK_NULL_HANDLE;
        }
    }

    pipeline_stat_queries_.clear();
    last_frame_pipeline_stats_ = {};

    for (auto &ubo: ubos_) {
        ubo.destroy();
    }

    for (auto &frame: frames_) {
        frame.forward_target.destroy(image_storage_);

        if (frame.shadow_atlas.valid()) {
            static_cast<void>(image_storage_.destroy_image(frame.shadow_atlas));
            frame.shadow_atlas = ImageHandle{};
        }

        frame.lights_buffer.destroy();
        frame.frustum_planes_buffer.destroy();
        frame.visible_transform_buffer.destroy();
        frame.visible_draw_buffer.destroy();
        frame.culled_indirect_buffer.destroy();
        frame.batch_bounds_buffer.destroy();
        frame.indirect_buffer.destroy();
        frame.transform_buffer.destroy();
        frame.draw_buffer.destroy();
        frame.upload_buffer.destroy();

        frame.draws.clear();
        frame.transforms.clear();
        frame.indirect_commands.clear();
        frame.batch_bounds.clear();

        frame.indirect_command_count = 0;
    }

    frames_.clear();
    material_storage_.destroy();
    image_storage_.destroy();
    geometry_arena_.destroy(context_);
    compiler.destroy();

    clear_submissions();

    for (auto &model: models_) {
        model.draws.clear();
        model.occupied = false;
    }

    models_.clear();

    for (auto &mesh: meshes_) {
        mesh.submeshes.clear();
        mesh.occupied = false;
    }

    meshes_.clear();

    default_material_handle_ = {};

    forward_pipeline_ = {};
    composite_pipeline_ = {};

    mesh_free_head_ = 0;
    model_free_head_ = 0;

    maximum_draw_count_ = 0;
    maximum_submission_count_ = 0;

    hdr_format_ = VK_FORMAT_UNDEFINED;
    depth_format_ = VK_FORMAT_UNDEFINED;

    samples_ = VK_SAMPLE_COUNT_1_BIT;
    extent_ = {};

    initialized_ = false;
}

auto Renderer::load_model(std::filesystem::path const &path) -> std::expected<ModelHandle, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    std::array<char, 64> header_buffer{};
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    file.read(header_buffer.data(), header_buffer.size());
    std::streamsize bytes_read = file.gcount();
    file.close();

    std::string_view header_view(header_buffer.data(), static_cast<std::size_t>(bytes_read));
    std::size_t file_hash = std::hash<std::string_view>{}(header_view);

    if (auto it = model_cache_.find(file_hash); it != model_cache_.end()) {
        return it->second; // Return cached handle
    }

    auto cpu_data = load_model_cpu(path, sampler_storage_);
    if (!cpu_data) {
        return std::unexpected(make_model_load_error(cpu_data.error()));
    }

    auto model_result = create_model_from_cpu_data(*cpu_data);

    if (!model_result) {
        return std::unexpected{model_result.error()};
    }

    model_cache_[file_hash] = *model_result;

    return model_result;
}

auto Renderer::create_model_from_cpu_data(ModelCpuData const &cpu_data) -> std::expected<ModelHandle, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    std::expected<Model, ModelLoadError> imported_model{
            std::unexpected(ModelLoadError{.type = ModelLoadErrorType::invalid_argument}),
    };

    context_.one_time_submit([&](VkCommandBuffer command_buffer) {
        imported_model =
                record_model_gpu_upload(cpu_data, command_buffer, geometry_arena_, image_storage_, material_storage_);
    });

    if (!imported_model) {
        return std::unexpected(make_model_load_error(imported_model.error()));
    }

    image_storage_.release_completed_uploads();

    return create_model(*imported_model, default_material_handle_);
}


auto Renderer::create_model(Model const &model) -> std::expected<ModelHandle, RendererError> {
    return create_model(model, default_material_handle_);
}

auto Renderer::create_model(Model const &model, MaterialHandle fallback_material)
        -> std::expected<ModelHandle, RendererError> {
    if (!initialized_ || model_free_head_ == 0) {
        return std::unexpected(make_error(model_free_head_ == 0 ? RendererErrorType::capacity_exceeded
                                                                : RendererErrorType::invalid_argument));
    }

    if (material_storage_.get(fallback_material) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_material));
    }

    std::vector<MeshHandle> imported_meshes;
    imported_meshes.resize(model.meshes.size());

    std::vector<MeshHandle> created_meshes;
    created_meshes.reserve(model.meshes.size());

    auto rollback_meshes = [this, &created_meshes] {
        for (auto const handle: created_meshes) {
            static_cast<void>(destroy_mesh(handle));
        }
    };

    for (std::size_t mesh_index = 0; mesh_index < model.meshes.size(); ++mesh_index) {
        auto const &source_mesh = model.meshes[mesh_index];

        if (source_mesh.primitives.empty()) {
            rollback_meshes();

            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }

        std::vector<SubmeshCreateInfo> submesh_infos;

        submesh_infos.reserve(source_mesh.primitives.size());

        for (auto const &source_submesh: source_mesh.primitives) {
            const auto index = source_submesh.material_index;

            if (source_submesh.material_index >= model.materials.size()) {
                rollback_meshes();

                return std::unexpected(make_error(RendererErrorType::invalid_material));
            }

            auto material = index.has_value() ? model.materials[index.value()] : fallback_material;

            submesh_infos.push_back(SubmeshCreateInfo{
                    .geometry = source_submesh.geometry,
                    .material = material,
                    .bounds_min = source_submesh.bounds_min,
                    .bounds_max = source_submesh.bounds_max,
            });
        }

        auto mesh = create_mesh(MeshCreateInfo{
                .submeshes = submesh_infos,
        });

        if (!mesh) {
            rollback_meshes();
            return std::unexpected(mesh.error());
        }

        imported_meshes[mesh_index] = *mesh;
        created_meshes.push_back(*mesh);
    }

    std::vector<ModelDraw> flattened_draws;

    auto add_node = [&](auto &&self, std::uint32_t node_index,
                        glm::mat4 const &parent_transform) -> std::expected<void, RendererError> {
        if (node_index >= model.nodes.size()) {
            return std::unexpected(make_error(RendererErrorType::invalid_model));
        }

        auto const &node = model.nodes[node_index];

        auto const local_to_model = parent_transform * node.local_transform;

        constexpr auto invalid_mesh = std::numeric_limits<std::uint32_t>::max();

        if (node.mesh_index != invalid_mesh) {
            if (node.mesh_index >= imported_meshes.size()) {
                return std::unexpected(make_error(RendererErrorType::invalid_mesh));
            }

            flattened_draws.push_back(ModelDraw{
                    .mesh = imported_meshes[node.mesh_index],
                    .local_transform = local_to_model,
            });
        }

        for (auto const child: node.children) {
            auto result = self(self, child, local_to_model);

            if (!result) {
                return result;
            }
        }

        return {};
    };

    for (auto const root: model.scene_roots) {
        auto result = add_node(add_node, root, glm::mat4{1.0F});

        if (!result) {
            rollback_meshes();
            return std::unexpected(result.error());
        }
    }

    if (flattened_draws.empty()) {
        rollback_meshes();

        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    auto const model_index = model_free_head_;

    auto &destination = models_[model_index];

    model_free_head_ = destination.next_free;

    destination.draws = std::move(flattened_draws);
    destination.bounds_min = model.bounds_min;
    destination.bounds_max = model.bounds_max;

    destination.next_free = 0;
    destination.occupied = true;

    destination.lights = model.lights;

    return ModelHandle{
            .index = model_index,
            .generation = destination.generation,
    };
}

auto Renderer::model_bounds(ModelHandle model) const -> std::optional<std::pair<glm::vec3, glm::vec3>> {
    auto const *slot = model_slot(model);

    if (slot == nullptr) {
        return std::nullopt;
    }

    return std::make_pair(slot->bounds_min, slot->bounds_max);
}

auto Renderer::model_lights(ModelHandle model) const -> std::span<ModelCpuLight const> {
    auto const *slot = model_slot(model);

    if (slot == nullptr) {
        return {};
    }

    return slot->lights;
}

auto Renderer::submit_model(ModelHandle model, glm::mat4 const &transform, MaterialHandle material_override)
        -> std::expected<void, RendererError> {
    if (model_slot(model) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (model_submissions_.size() >= maximum_submission_count_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    model_submissions_.push_back(ModelSubmission{
            .model = model,
            .transform = transform,
            .material_override = material_override,
    });

    return {};
}

auto Renderer::submit_model(ModelHandle model, glm::mat4 &&transform, MaterialHandle material_override)
        -> std::expected<void, RendererError> {
    if (model_slot(model) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (model_submissions_.size() >= maximum_submission_count_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    model_submissions_.push_back(ModelSubmission{
            .model = model,
            .transform = std::move(transform),
            .material_override = material_override,
    });

    return {};
}

auto Renderer::create_material(MaterialCreateInfo const &create_info) -> std::expected<MaterialHandle, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto material = material_storage_.create_material(to_gpu_material(create_info));

    if (!material) {
        return std::unexpected(make_material_error(material.error()));
    }

    return *material;
}

auto Renderer::update_material(MaterialHandle handle, MaterialCreateInfo const &create_info)
        -> std::expected<void, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto result = material_storage_.update_material(handle, to_gpu_material(create_info));

    if (!result) {
        return std::unexpected(make_material_error(result.error()));
    }

    return {};
}

auto Renderer::destroy_material(MaterialHandle handle) -> std::expected<void, RendererError> {
    if (handle == default_material_handle_) {
        return std::unexpected(make_error(RendererErrorType::invalid_material));
    }

    auto result = material_storage_.destroy_material(handle);

    if (!result) {
        return std::unexpected(make_material_error(result.error()));
    }

    return {};
}

auto Renderer::create_mesh(MeshCreateInfo const &create_info) -> std::expected<MeshHandle, RendererError> {
    if (!initialized_ || create_info.submeshes.empty()) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    if (mesh_free_head_ == 0) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    for (auto const &submesh_info: create_info.submeshes) {
        if (!submesh_info.geometry.vertices.bytes.valid() || !submesh_info.geometry.indices.bytes.valid() ||
            submesh_info.geometry.vertices.vertex_count == 0 || submesh_info.geometry.indices.index_count == 0) {
            return std::unexpected(make_error(RendererErrorType::invalid_argument));
        }

        if (material_storage_.get(submesh_info.material) == nullptr) {
            return std::unexpected(make_error(RendererErrorType::invalid_material));
        }

        auto stride = index_stride(submesh_info.geometry.indices.index_type);

        if (!stride) {
            return std::unexpected(stride.error());
        }

        if (submesh_info.geometry.indices.bytes.offset % *stride != 0) {
            return std::unexpected(make_error(RendererErrorType::invalid_argument));
        }
    }

    auto const index = mesh_free_head_;
    auto &slot = meshes_[index];

    mesh_free_head_ = slot.next_free;

    slot.submeshes.clear();

    slot.submeshes.reserve(create_info.submeshes.size());

    for (auto const &submesh_info: create_info.submeshes) {
        slot.submeshes.push_back(Submesh{
                .geometry = submesh_info.geometry,
                .material = submesh_info.material,
                .bounds_min = submesh_info.bounds_min,
                .bounds_max = submesh_info.bounds_max,
        });
    }

    slot.next_free = 0;
    slot.occupied = true;

    return MeshHandle{
            .index = index,
            .generation = slot.generation,
    };
}

auto Renderer::destroy_mesh(MeshHandle handle) -> std::expected<void, RendererError> {
    auto *slot = mesh_slot(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_mesh));
    }

    slot->submeshes.clear();
    slot->occupied = false;

    ++slot->generation;

    if (slot->generation == 0) {
        slot->generation = 1;
    }

    slot->next_free = mesh_free_head_;
    mesh_free_head_ = handle.index;

    return {};
}

auto Renderer::submit_point_light(PointLight const &light) -> std::expected<void, RendererError> {
    if (point_light_submissions_.size() + spot_light_submissions_.size() >= maximum_light_count) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    point_light_submissions_.push_back(light);

    return {};
}

auto Renderer::submit_spot_light(SpotLight const &light) -> std::expected<void, RendererError> {
    if (point_light_submissions_.size() + spot_light_submissions_.size() >= maximum_light_count) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    spot_light_submissions_.push_back(light);

    return {};
}

auto Renderer::submit_mesh(MeshHandle mesh, glm::mat4 const &transform, MaterialHandle material_override)
        -> std::expected<void, RendererError> {
    if (mesh_slot(mesh) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_mesh));
    }

    if (submissions_.size() >= maximum_submission_count_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    submissions_.push_back(Submission{
            .mesh = mesh,
            .transform = transform,
            .material_override = material_override,
    });

    return {};
}

auto Renderer::prepare_frame(VkCommandBuffer command_buffer, const CameraMatrices &matrices, std::uint32_t frame_index)
        -> std::expected<void, RendererError> {
    if (!initialized_ || command_buffer == VK_NULL_HANDLE || frame_index >= frames_.size()) {
        clear_submissions();

        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    std::uint32_t submitted_triangle_count = 0;

    // The frustum-culling compute dispatch below is recorded into this same
    // command buffer before record_frame() runs, so the query pool must be
    // reset and the FullFrame timer started here rather than at the top of
    // record_frame -- otherwise record_frame's reset would wipe out the
    // Culling timestamps written below.
    auto &frame_query = timestamp_queries_[frame_index];
    vkCmdResetQueryPool(command_buffer, frame_query.query_pool, 0, query_count);
    auto &frame_pipeline_query = pipeline_stat_queries_[frame_index];
    vkCmdResetQueryPool(command_buffer, frame_pipeline_query.query_pool, 0, 1);

    vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame_query.query_pool,
                         static_cast<std::uint32_t>(RenderStage::FullFrame) * 2);

    pipeline_graph_.tick_retirement();

    if (auto changed = shader_change_queue_.drain(); !changed.empty()) {
        pipeline_graph_.on_files_changed(changed);
    }

    pipeline_graph_.process_dirty(compiler);


    auto image_result = image_storage_.prepare_frame(command_buffer);

    if (!image_result) {
        clear_submissions();

        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    auto resource_result = gpu_resource_table_.prepare_frame(frame_index, image_storage_, sampler_storage_);

    if (!resource_result) {
        return std::unexpected(make_resource_table_error(resource_result.error()));
    }

    auto &frame = frames_[frame_index];

    frame.draws.clear();
    frame.transforms.clear();
    frame.indirect_commands.clear();
    frame.batch_bounds.clear();
    frame.indirect_command_count = 0;
    frame.opaque_indirect_count = 0;
    frame.mask_indirect_count = 0;
    frame.blend_indirect_count = 0;

    // Used below to sort blend batches back-to-front; computed here (rather
    // than where the UBO is built further down) so it's available before
    // the batch partition.
    auto const camera_position = glm::vec3(glm::inverse(matrices.view)[3]);

    struct batch_key {
        std::uint32_t mesh_index;
        std::uint32_t submesh_index;
        std::uint32_t material_index;

        auto operator==(batch_key const &) const noexcept -> bool = default;
    };

    struct batch_key_hash {
        auto operator()(batch_key const &key) const noexcept -> std::size_t {
            auto const mesh_hash =
                    std::hash<std::uint64_t>{}((static_cast<std::uint64_t>(key.mesh_index) << 32) | key.submesh_index);

            return mesh_hash ^ (std::hash<std::uint32_t>{}(key.material_index) << 1);
        }
    };

    struct batch_entry {
        MeshHandle mesh{};
        std::uint32_t submesh_index = 0;
        MaterialHandle material{};
        std::vector<glm::mat4> transforms;
    };

    std::unordered_map<batch_key, batch_entry, batch_key_hash> batches;
    batches.reserve(model_submissions_.size() + submissions_.size());

    for (auto const &model_submission: model_submissions_) {
        auto const *model = model_slot(model_submission.model);

        if (model == nullptr) {
            clear_submissions();

            return std::unexpected(make_error(RendererErrorType::invalid_model));
        }

        for (auto const &model_draw: model->draws) {
            auto const *mesh = mesh_slot(model_draw.mesh);

            if (mesh == nullptr) {
                clear_submissions();

                return std::unexpected(make_error(RendererErrorType::invalid_mesh));
            }

            auto const instance_transform = model_submission.transform * model_draw.local_transform;

            for (std::uint32_t submesh_index = 0; submesh_index < mesh->submeshes.size(); ++submesh_index) {
                auto const &submesh = mesh->submeshes[submesh_index];

                auto const material = model_submission.material_override.valid() ? model_submission.material_override
                                                                                 : submesh.material;

                auto const key = batch_key{
                        .mesh_index = model_draw.mesh.index,
                        .submesh_index = submesh_index,
                        .material_index = material_storage_.gpu_index(material),
                };

                auto [entry, inserted] = batches.try_emplace(key);

                if (inserted) {
                    entry->second.mesh = model_draw.mesh;
                    entry->second.submesh_index = submesh_index;
                    entry->second.material = material;
                }

                entry->second.transforms.push_back(instance_transform);
            }
        }
    }

    for (auto const &submission: submissions_) {
        auto const *mesh = mesh_slot(submission.mesh);

        if (mesh == nullptr) {
            clear_submissions();

            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }

        for (std::uint32_t submesh_index = 0; submesh_index < mesh->submeshes.size(); ++submesh_index) {
            auto const &submesh = mesh->submeshes[submesh_index];

            auto const material =
                    submission.material_override.valid() ? submission.material_override : submesh.material;

            auto const key = batch_key{
                    .mesh_index = submission.mesh.index,
                    .submesh_index = submesh_index,
                    .material_index = material_storage_.gpu_index(material),
            };

            auto [entry, inserted] = batches.try_emplace(key);

            if (inserted) {
                entry->second.mesh = submission.mesh;
                entry->second.submesh_index = submesh_index;
                entry->second.material = material;
            }

            entry->second.transforms.push_back(submission.transform);
        }
    }

    // Emits one batch into frame.transforms/draws/indirect_commands/
    // batch_bounds -- the same per-batch logic the single loop used to run
    // inline, now shared by the three-way partition below so opaque, mask
    // and blend batches land in three contiguous ranges (opaque, then mask,
    // then blend) within frame.indirect_commands.
    auto const emit_batch =
            [this, &frame, &submitted_triangle_count](batch_entry const &batch) -> std::expected<void, RendererError> {
        auto const *mesh = mesh_slot(batch.mesh);

        if (mesh == nullptr) {
            clear_submissions();

            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }

        auto const &submesh = mesh->submeshes[batch.submesh_index];

        auto const instance_count = static_cast<std::uint32_t>(batch.transforms.size());
        submitted_triangle_count += (submesh.geometry.indices.index_count / 3) * instance_count;

        if (frame.transforms.size() + instance_count > maximum_submission_count_ ||
            frame.draws.size() + instance_count > maximum_draw_count_) {
            clear_submissions();

            return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
        }

        auto const base_transform_index = static_cast<std::uint32_t>(frame.transforms.size());

        for (auto const &transform: batch.transforms) {
            frame.transforms.push_back(transform);
        }

        auto const stride = index_stride(submesh.geometry.indices.index_type);

        if (!stride) {
            clear_submissions();

            return std::unexpected(stride.error());
        }

        auto const first_index_u64 = submesh.geometry.indices.bytes.offset / *stride;

        if (first_index_u64 > std::numeric_limits<std::uint32_t>::max()) {
            clear_submissions();

            return std::unexpected(make_error(RendererErrorType::size_overflow));
        }

        auto const first_instance = static_cast<std::uint32_t>(frame.draws.size());

        for (std::uint32_t instance = 0; instance < instance_count; ++instance) {
            frame.draws.push_back(GpuDraw{
                    .vertex_address = geometry_arena_.vertex_address(submesh.geometry.vertices),
                    .material_index = material_storage_.gpu_index(batch.material),
                    .transform_index = base_transform_index + instance,
            });
        }

        auto const indirect_command = VkDrawIndexedIndirectCommand{
                .indexCount = submesh.geometry.indices.index_count,
                .instanceCount = instance_count,
                .firstIndex = static_cast<std::uint32_t>(first_index_u64),
                .vertexOffset = 0,
                .firstInstance = first_instance,
        };

        frame.indirect_commands.push_back(indirect_command);

        auto const *material = material_storage_.get(batch.material);
        auto const wind_padding = material != nullptr ? material->wind_strength : 0.0F;

        frame.batch_bounds.push_back(GpuCullBounds{
                .bounds_min = submesh.bounds_min,
                .wind_padding = wind_padding,
                .bounds_max = submesh.bounds_max,
        });

        return {};
    };

    struct pending_blend_batch {
        batch_entry const *entry;
        float camera_distance_sq;
    };

    std::vector<batch_entry const *> opaque_batches;
    std::vector<batch_entry const *> mask_batches;
    std::vector<pending_blend_batch> blend_batches;

    opaque_batches.reserve(batches.size());
    mask_batches.reserve(batches.size());
    blend_batches.reserve(batches.size());

    for (auto const &[key, batch]: batches) {
        auto const *material = material_storage_.get(batch.material);
        auto const alpha_mode = material != nullptr ? material->alpha_mode : AlphaMode::opaque;

        switch (alpha_mode) {
            case AlphaMode::opaque:
                opaque_batches.push_back(&batch);
                break;

            case AlphaMode::mask:
                mask_batches.push_back(&batch);
                break;

            case AlphaMode::blend: {
                auto const *mesh = mesh_slot(batch.mesh);

                if (mesh == nullptr) {
                    clear_submissions();

                    return std::unexpected(make_error(RendererErrorType::invalid_mesh));
                }

                auto const &submesh = mesh->submeshes[batch.submesh_index];
                auto const local_centre = (submesh.bounds_min + submesh.bounds_max) * 0.5F;

                auto const world_centre = glm::vec3(batch.transforms.front() * glm::vec4(local_centre, 1.0F));
                auto const distance = world_centre - camera_position;

                blend_batches.push_back({&batch, glm::dot(distance, distance)});
                break;
            }
        }
    }

    std::sort(blend_batches.begin(), blend_batches.end(),
              [](auto const &a, auto const &b) { return a.camera_distance_sq > b.camera_distance_sq; });

    // Sort opaque/mask batches by descending max_shadow_cascade so that
    // batches restricted to fewer cascades sort to the back. Because the
    // resulting order is non-increasing, "how many leading indirect commands
    // does cascade C need" is just a prefix length -- computed below and
    // used to shrink drawCount per cascade in the shadow pass, skipping
    // batches (e.g. grass) that opted out of the farther cascades entirely
    // instead of rasterizing them into every cascade regardless.
    //
    // GpuMaterial::no_shadow_cascade (opt out of the shadow pass entirely)
    // is mapped to -1 here so it sorts behind even cascade 0 -- max_shadow_
    // cascade is otherwise a small non-negative cascade index, so this is
    // the one value that must compare lower than all of them.
    auto const batch_max_shadow_cascade = [this](batch_entry const *batch) noexcept -> std::int32_t {
        auto const *material = material_storage_.get(batch->material);
        auto const cascade = material != nullptr ? material->max_shadow_cascade : shadow_cascade_count - 1;

        return cascade == GpuMaterial::no_shadow_cascade ? -1 : static_cast<std::int32_t>(cascade);
    };

    std::sort(opaque_batches.begin(), opaque_batches.end(),
              [&](auto const *a, auto const *b) { return batch_max_shadow_cascade(a) > batch_max_shadow_cascade(b); });

    std::sort(mask_batches.begin(), mask_batches.end(),
              [&](auto const *a, auto const *b) { return batch_max_shadow_cascade(a) > batch_max_shadow_cascade(b); });

    auto const shadow_prefix_counts = [&batch_max_shadow_cascade](
                                              std::vector<batch_entry const *> const &sorted_batches) {
        std::array<std::uint32_t, shadow_cascade_count> counts{};

        for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
            auto const it = std::find_if(sorted_batches.begin(), sorted_batches.end(), [&](batch_entry const *batch) {
                return batch_max_shadow_cascade(batch) < static_cast<std::int32_t>(cascade);
            });

            counts[cascade] = static_cast<std::uint32_t>(std::distance(sorted_batches.begin(), it));
        }

        return counts;
    };

    frame.shadow_opaque_indirect_count = shadow_prefix_counts(opaque_batches);
    frame.shadow_mask_indirect_count = shadow_prefix_counts(mask_batches);

    for (auto const *batch: opaque_batches) {
        if (auto result = emit_batch(*batch); !result) {
            return std::unexpected(result.error());
        }
    }

    frame.opaque_indirect_count = static_cast<std::uint32_t>(frame.indirect_commands.size());

    for (auto const *batch: mask_batches) {
        if (auto result = emit_batch(*batch); !result) {
            return std::unexpected(result.error());
        }
    }

    frame.mask_indirect_count =
            static_cast<std::uint32_t>(frame.indirect_commands.size()) - frame.opaque_indirect_count;

    for (auto const &pending: blend_batches) {
        if (auto result = emit_batch(*pending.entry); !result) {
            return std::unexpected(result.error());
        }
    }

    frame.blend_indirect_count = static_cast<std::uint32_t>(frame.indirect_commands.size()) -
                                 frame.opaque_indirect_count - frame.mask_indirect_count;

    frame.indirect_command_count = static_cast<std::uint32_t>(frame.indirect_commands.size());

    auto material_result = material_storage_.prepare_frame(command_buffer, frame_index);

    if (!material_result) {
        clear_submissions();

        return std::unexpected(make_material_error(material_result.error()));
    }

    // Checked immediately, not deferred to the end of this function: the
    // dispatch and draws recorded below all read buffers this call fills,
    // so recording them against a failed (partial/unwritten) upload would
    // be worse than bailing out with the frame's other work already
    // clear_submissions()-ed.
    if (auto upload_result = upload_frame_data(command_buffer, frame); !upload_result) {
        clear_submissions();

        return std::unexpected(upload_result.error());
    }

    auto const &view = matrices.view;
    auto const &projection = matrices.projection;

    auto const cascades = fit_shadow_cascades(ShadowCascadeFitInput{
            .camera_view = view,
            .camera_near = matrices.near_clip,
            .camera_far = matrices.far_clip,
            .vertical_fov_radians = matrices.vertical_fov_radians,
            .aspect_ratio = matrices.aspect_ratio,
            .light_direction = glm::normalize(light_.direction),
            .settings = shadow_settings_.cascades,
    });

    auto const view_projection = projection * view;
    auto const frustum_planes = extract_frustum_planes(view_projection);

    UBO const ubo{
            .view_projection = view_projection,
            .view = view,
            .projection = projection,
            .camera_position = camera_position,
            .fog_colour = glm::vec3{0.5F},
            .cascade_view_projection = cascades.view_projection,
            .cascade_split_far = glm::make_vec4(cascades.split_far.data()),
            .cascade_texel_world = glm::make_vec4(cascades.texel_world.data()),
            .cascade_depth_scale = glm::make_vec4(cascades.depth_scale.data()),
            .cascade_atlas_offset_u = glm::vec4{static_cast<float>(shadow_cascade_offset_x[0]),
                                                static_cast<float>(shadow_cascade_offset_x[1]),
                                                static_cast<float>(shadow_cascade_offset_x[2]),
                                                static_cast<float>(shadow_cascade_offset_x[3])} /
                                      static_cast<float>(shadow_atlas_width),
            .cascade_atlas_scale_u = glm::vec4{static_cast<float>(shadow_cascade_resolutions[0]),
                                               static_cast<float>(shadow_cascade_resolutions[1]),
                                               static_cast<float>(shadow_cascade_resolutions[2]),
                                               static_cast<float>(shadow_cascade_resolutions[3])} /
                                     static_cast<float>(shadow_atlas_width),
            .cascade_atlas_scale_v = glm::vec4{static_cast<float>(shadow_cascade_resolutions[0]),
                                               static_cast<float>(shadow_cascade_resolutions[1]),
                                               static_cast<float>(shadow_cascade_resolutions[2]),
                                               static_cast<float>(shadow_cascade_resolutions[3])} /
                                     static_cast<float>(shadow_atlas_height),
            .light_direction = glm::normalize(light_.direction),
            .light_intensity = light_.intensity,
            .light_colour = light_.colour,
            .shadow_normal_offset_texels = shadow_settings_.normal_offset_texels,
            .shadow_atlas_texture = frame.shadow_atlas.index,
            .shadow_sampler = sampler_storage_.shadow_compare().index,
            .shadow_depth_bias_world = shadow_settings_.depth_bias_world,
            .shadow_pcf_radius_texels = shadow_settings_.pcf_radius_texels,
            .cascade_count = shadow_cascade_count,
            .shadow_atlas_texel_u = 1.0F / static_cast<float>(shadow_atlas_width),
            .shadow_atlas_texel_v = 1.0F / static_cast<float>(shadow_atlas_height),
            .shadow_debug_cascade_tint = shadow_settings_.debug_cascade_tint ? 1U : 0U,
            .time = matrices.time,
            .ambient_intensity = ambient_intensity_,
    };
    if (!ubos_[frame_index].write(0, std::span{&ubo, 1})) {
        clear_submissions();

        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    if (!frame.frustum_planes_buffer.write(0, std::span{frustum_planes})) {
        clear_submissions();

        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    if ((lights_dirty_mask_ & (1u << frame_index)) != 0) {
        light_staging_.clear();
        light_staging_.reserve(point_light_submissions_.size() + spot_light_submissions_.size());

        for (auto const &point: point_light_submissions_) {
            light_staging_.push_back(GpuLight{
                    .position = point.position,
                    .range = point.range,
                    .colour = point.colour,
                    .intensity = point.intensity,
                    .type = GpuLightType::point,
            });
        }

        for (auto const &spot: spot_light_submissions_) {
            auto const inner = glm::radians(std::min(spot.inner_cone_degrees, spot.outer_cone_degrees));
            auto const outer = glm::radians(std::max(spot.inner_cone_degrees, spot.outer_cone_degrees));
            auto const cos_inner = std::cos(inner);
            auto const cos_outer = std::cos(outer);
            auto const spot_scale = 1.0F / std::max(cos_inner - cos_outer, 1e-4F);
            auto const spot_offset = -cos_outer * spot_scale;

            light_staging_.push_back(GpuLight{
                    .position = spot.position,
                    .range = spot.range,
                    .colour = spot.colour,
                    .intensity = spot.intensity,
                    .direction = glm::normalize(spot.direction),
                    .spot_scale = spot_scale,
                    .spot_offset = spot_offset,
                    .type = GpuLightType::spot,
            });
        }

        light_count_ = static_cast<std::uint32_t>(light_staging_.size());
        frame.light_count = light_count_;

        if (!light_staging_.empty() && !frame.lights_buffer.write(0, std::as_bytes(std::span{light_staging_}))) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::device_error));
        }

        lights_dirty_mask_ &= ~(1u << frame_index);
    } else {
        frame.light_count = light_count_;
    }

    vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame_query.query_pool,
                         static_cast<std::uint32_t>(RenderStage::Culling) * 2);

    if (frame.indirect_command_count != 0) {
        if (frame.indirect_command_count > 65535) {
            clear_submissions();

            return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
        }

        auto const frustum_cull_pipeline = resolve_layout(pipeline_graph_, frustum_cull_pipeline_);

        if (frustum_cull_pipeline == VK_NULL_HANDLE) {
            clear_submissions();

            return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
        }

        bind_compute_node(pipeline_graph_, frustum_cull_pipeline_, command_buffer);
        gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, frustum_cull_pipeline);

        CullPushConstants const cull_pc{
                .src_draws_address = frame.draw_buffer.device_address,
                .src_transforms_address = frame.transform_buffer.device_address,
                .batch_bounds_address = frame.batch_bounds_buffer.device_address,
                .src_indirect_address = frame.indirect_buffer.device_address,
                .dst_indirect_address = frame.culled_indirect_buffer.device_address,
                .dst_draws_address = frame.visible_draw_buffer.device_address,
                .dst_transforms_address = frame.visible_transform_buffer.device_address,
                .frustum_planes_address = frame.frustum_planes_buffer.device_address,
                .batch_count = frame.indirect_command_count,
                .padding = 0,
        };

        vkCmdPushConstants(command_buffer, frustum_cull_pipeline, VK_SHADER_STAGE_ALL, 0, sizeof(CullPushConstants),
                           &cull_pc);

        vkCmdDispatch(command_buffer, frame.indirect_command_count, 1, 1);

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, frame_query.query_pool,
                             (static_cast<std::uint32_t>(RenderStage::Culling) * 2) + 1);

        std::array<VkBufferMemoryBarrier2, 3> const post_cull_barriers{
                VkBufferMemoryBarrier2{
                        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                        .pNext = nullptr,
                        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = frame.visible_draw_buffer.buffer,
                        .offset = 0,
                        .size = VK_WHOLE_SIZE,
                },
                VkBufferMemoryBarrier2{
                        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                        .pNext = nullptr,
                        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = frame.visible_transform_buffer.buffer,
                        .offset = 0,
                        .size = VK_WHOLE_SIZE,
                },
                VkBufferMemoryBarrier2{
                        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                        .pNext = nullptr,
                        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = frame.culled_indirect_buffer.buffer,
                        .offset = 0,
                        .size = VK_WHOLE_SIZE,
                },
        };

        VkDependencyInfo const post_cull_dependency_info{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = nullptr,
                .dependencyFlags = 0,
                .memoryBarrierCount = 0,
                .pMemoryBarriers = nullptr,
                .bufferMemoryBarrierCount = static_cast<std::uint32_t>(post_cull_barriers.size()),
                .pBufferMemoryBarriers = post_cull_barriers.data(),
                .imageMemoryBarrierCount = 0,
                .pImageMemoryBarriers = nullptr,
        };

        vkCmdPipelineBarrier2(command_buffer, &post_cull_dependency_info);
    } else {
        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame_query.query_pool,
                             (static_cast<std::uint32_t>(RenderStage::Culling) * 2) + 1);
    }

    last_frame_stats_ = FrameStats{
            .submitted_triangle_count = submitted_triangle_count,
            .submitted_instance_count = static_cast<std::uint32_t>(frame.transforms.size()),
            .indirect_command_count = frame.indirect_command_count,
            .opaque_indirect_count = frame.opaque_indirect_count,
            .mask_indirect_count = frame.mask_indirect_count,
            .blend_indirect_count = frame.blend_indirect_count,
            .model_submission_count = static_cast<std::uint32_t>(model_submissions_.size()),
            .mesh_submission_count = static_cast<std::uint32_t>(submissions_.size()),
            .point_light_count = static_cast<std::uint32_t>(point_light_submissions_.size()),
            .spot_light_count = static_cast<std::uint32_t>(spot_light_submissions_.size()),
    };

    clear_submissions();

    if (frame_query.has_results) {
        std::vector<std::uint64_t> results(query_count);

        VkResult query_res = vkGetQueryPoolResults(context_.device, frame_query.query_pool, 0, query_count,
                                                   sizeof(std::uint64_t) * query_count, results.data(),
                                                   sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);

        if (query_res == VK_SUCCESS) {
            for (std::uint32_t i = 0; i < stage_count; ++i) {
                std::uint64_t const start = results[i * 2];
                std::uint64_t const end = results[i * 2 + 1];

                last_frame_timings_.milliseconds[i] = static_cast<float>(end - start) * timestamp_period_ / 1000000.0f;
            }

            last_frame_timings_.valid = true;
        }

        frame_query.has_results = false;
    }

    if (frame_pipeline_query.has_results) {
        std::array<std::uint64_t, pipeline_stat_count> results{};

        VkResult const query_res =
                vkGetQueryPoolResults(context_.device, frame_pipeline_query.query_pool, 0, 1, sizeof(results),
                                      results.data(), sizeof(results), VK_QUERY_RESULT_64_BIT);

        if (query_res == VK_SUCCESS) {
            last_frame_pipeline_stats_.assembled_primitive_count = results[0];
            last_frame_pipeline_stats_.clipped_primitive_count = results[1];
            last_frame_pipeline_stats_.assembled_vertex_count = results[2];
            last_frame_pipeline_stats_.fragment_shader_invocation_count = results[3];
            last_frame_pipeline_stats_.valid = true;
        }

        frame_pipeline_query.has_results = false;
    }

    return {};
}


template<typename OverlayPolicy>
[[nodiscard]] auto Renderer::record_frame(VkCommandBuffer command_buffer, SwapchainImage const &swapchain_image,
                                          std::uint32_t frame_index, Application const &app, glm::mat4 const &vp)
        -> std::expected<void, RendererError> {

    if (!initialized_ || command_buffer == VK_NULL_HANDLE || swapchain_image.image == VK_NULL_HANDLE ||
        swapchain_image.view == VK_NULL_HANDLE || swapchain_image.format == VK_FORMAT_UNDEFINED ||
        swapchain_image.extent.width == 0 || swapchain_image.extent.height == 0 || frame_index >= frames_.size()) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }


    screenshot_.try_resolve(frame_index);

    auto &frame_query = timestamp_queries_[frame_index];

    auto &frame = frames_[frame_index];
    auto const *hdr = image_storage_.get(frame.forward_target.hdr());
    auto const *depth = image_storage_.get(frame.forward_target.depth());

    if (hdr == nullptr || depth == nullptr || !hdr->valid() || !depth->valid()) {
        return std::unexpected(make_error(RendererErrorType::image_error));
    }

    bool const is_msaa = frame.forward_target.is_multisampled();

    Image const *resolved_hdr = hdr;
    Image const *resolved_depth = depth;

    if (is_msaa) {
        resolved_hdr = image_storage_.get(frame.forward_target.resolved_hdr());
        resolved_depth = image_storage_.get(frame.forward_target.resolved_depth());

        if (resolved_hdr == nullptr || resolved_depth == nullptr || !resolved_hdr->valid() ||
            !resolved_depth->valid()) {
            return std::unexpected(make_error(RendererErrorType::image_error));
        }
    }

    auto const forward_pipeline = resolve_layout(pipeline_graph_, forward_pipeline_);
    if (forward_pipeline == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const forward_blend_pipeline = resolve_layout(pipeline_graph_, forward_blend_pipeline_);
    if (forward_blend_pipeline == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const depth_prepass_pipeline = resolve_layout(pipeline_graph_, depth_prepass_pipeline_);
    if (depth_prepass_pipeline == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const depth_prepass_mask_pipeline = resolve_layout(pipeline_graph_, depth_prepass_mask_pipeline_);
    if (depth_prepass_mask_pipeline == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const composite_pipeline = resolve_layout(pipeline_graph_, composite_pipeline_);
    if (composite_pipeline == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const shadow_pipeline = resolve_layout(pipeline_graph_, shadow_pipeline_);
    if (shadow_pipeline == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const shadow_mask_pipeline = resolve_layout(pipeline_graph_, shadow_mask_pipeline_);
    if (shadow_mask_pipeline == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const *shadow_atlas = image_storage_.get(frame.shadow_atlas);
    if (shadow_atlas == nullptr || !shadow_atlas->valid()) {
        return std::unexpected(make_error(RendererErrorType::image_error));
    }

    auto const target_extent = frame.forward_target.extent();

    if (target_extent.width == 0 || target_extent.height == 0 || hdr->extent_2d().width != target_extent.width ||
        hdr->extent_2d().height != target_extent.height || depth->extent_2d().width != target_extent.width ||
        depth->extent_2d().height != target_extent.height) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

#pragma region Shadow pass
    {
        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame_query.query_pool,
                             static_cast<std::uint32_t>(RenderStage::ShadowPass) * 2);

        transition_shadow_atlas_to_attachment(command_buffer, *shadow_atlas);

        VkRenderingAttachmentInfo shadow_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = shadow_atlas->view(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                // Reverse-Z far value -- "nothing occludes" until a caster writes into it.
                .clearValue =
                        {
                                .depthStencil =
                                        {
                                                .depth = 0.0F,
                                                .stencil = 0,
                                        },
                        },
        };

        VkRenderingInfo const shadow_rendering_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea =
                        VkRect2D{
                                .offset = {0, 0},
                                .extent = {shadow_atlas_width, shadow_atlas_height},
                        },
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 0,
                .pColorAttachments = nullptr,
                .pDepthAttachment = &shadow_attachment,
                .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(command_buffer, &shadow_rendering_info);
        vkCmdBindIndexBuffer(command_buffer, geometry_arena_.bindable_buffer(), 0, VK_INDEX_TYPE_UINT32);

        if (frame.opaque_indirect_count != 0) {
            bind_graphics_node(pipeline_graph_, shadow_pipeline_, command_buffer, VK_SAMPLE_COUNT_1_BIT, 0, false);
            gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
            ShadowPushConstants pc{
                    .draw_buffer_address = frame.draw_buffer.device_address,
                    .transform_buffer_address = frame.transform_buffer.device_address,
                    .material_buffer_address = material_storage_.device_address(),
                    .ubo_buffer_address = ubos_[frame_index].device_address,
                    .lights_buffer_address = frame.lights_buffer.device_address,
                    .light_count = 0,
                    .padding0 = 0,
                    .cascade_index = 0,
                    .padding1 = 0,
            };

            vkCmdPushConstants(command_buffer, shadow_pipeline, VK_SHADER_STAGE_ALL, 0, sizeof(ShadowPushConstants),
                               &pc);

            for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
                auto const cascade_draw_count = frame.shadow_opaque_indirect_count[cascade];

                if (cascade_draw_count == 0) {
                    continue;
                }

                set_shadow_dynamic_state(command_buffer, cascade, shadow_settings_.depth_bias_constant,
                                         shadow_settings_.depth_bias_slope);

                vkCmdPushConstants(command_buffer, shadow_pipeline, VK_SHADER_STAGE_ALL,
                                   offsetof(ShadowPushConstants, cascade_index), sizeof(std::uint32_t), &cascade);

                vkCmdDrawIndexedIndirect(command_buffer, frame.indirect_buffer.buffer, 0, cascade_draw_count,
                                         sizeof(VkDrawIndexedIndirectCommand));
            }
        }

        if (frame.mask_indirect_count != 0) {
            bind_graphics_node(pipeline_graph_, shadow_mask_pipeline_, command_buffer, VK_SAMPLE_COUNT_1_BIT, 0, false);
            gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     shadow_mask_pipeline);

            ShadowPushConstants pc{
                    .draw_buffer_address = frame.draw_buffer.device_address,
                    .transform_buffer_address = frame.transform_buffer.device_address,
                    .material_buffer_address = material_storage_.device_address(),
                    .ubo_buffer_address = ubos_[frame_index].device_address,
                    .lights_buffer_address = frame.lights_buffer.device_address,
                    .light_count = 0,
                    .padding0 = 0,
                    .cascade_index = 0,
                    .padding1 = 0,
            };

            vkCmdPushConstants(command_buffer, shadow_mask_pipeline, VK_SHADER_STAGE_ALL, 0,
                               sizeof(ShadowPushConstants), &pc);

            auto const mask_offset =
                    static_cast<VkDeviceSize>(frame.opaque_indirect_count) * sizeof(VkDrawIndexedIndirectCommand);

            for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
                auto const cascade_draw_count = frame.shadow_mask_indirect_count[cascade];

                if (cascade_draw_count == 0) {
                    continue;
                }

                set_shadow_dynamic_state(command_buffer, cascade, shadow_settings_.depth_bias_constant,
                                         shadow_settings_.depth_bias_slope);

                vkCmdPushConstants(command_buffer, shadow_mask_pipeline, VK_SHADER_STAGE_ALL,
                                   offsetof(ShadowPushConstants, cascade_index), sizeof(std::uint32_t), &cascade);

                vkCmdDrawIndexedIndirect(command_buffer, frame.indirect_buffer.buffer, mask_offset, cascade_draw_count,
                                         sizeof(VkDrawIndexedIndirectCommand));
            }
        }

        vkCmdEndRendering(command_buffer);
        transition_shadow_atlas_to_shader_read(command_buffer, *shadow_atlas);

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, frame_query.query_pool,
                             (static_cast<std::uint32_t>(RenderStage::ShadowPass) * 2) + 1);
    }
#pragma endregion

    transition_forward_target_to_attachments(command_buffer, *hdr, *depth, is_msaa ? resolved_hdr : nullptr,
                                             is_msaa ? resolved_depth : nullptr);

    auto const &main_view_draw_buffer = frame.visible_draw_buffer;
    auto const &main_view_transform_buffer = frame.visible_transform_buffer;
    auto const &main_view_indirect_buffer = frame.culled_indirect_buffer;

#pragma region Predepth pass
    {
        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame_query.query_pool,
                             static_cast<std::uint32_t>(RenderStage::DepthPrepass) * 2);

        VkRenderingAttachmentInfo depth_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = depth->view(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {},
        };

        if (is_msaa) {
            depth_attachment.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
            depth_attachment.resolveImageView = resolved_depth->view();
            depth_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        }

        VkRenderingInfo const depth_prepass_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea =
                        VkRect2D{
                                .offset = {0, 0},
                                .extent = target_extent,
                        },
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 0,
                .pColorAttachments = nullptr,
                .pDepthAttachment = &depth_attachment,
                .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(command_buffer, &depth_prepass_info);
        vkCmdBindIndexBuffer(command_buffer, geometry_arena_.bindable_buffer(), 0, VK_INDEX_TYPE_UINT32);

        ForwardPushConstants pc{
                .draw_buffer_address = main_view_draw_buffer.device_address,
                .transform_buffer_address = main_view_transform_buffer.device_address,
                .material_buffer_address = material_storage_.device_address(),
                .ubo_buffer_address = ubos_[frame_index].device_address,
                .lights_buffer_address = frame.lights_buffer.device_address,
                .light_count = 0,
                .padding = 0,
        };

        if (frame.opaque_indirect_count != 0) {
            bind_graphics_node(pipeline_graph_, depth_prepass_pipeline_, command_buffer, samples_, 0, false);
            gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     depth_prepass_pipeline);
            set_forward_dynamic_state(command_buffer, target_extent, ForwardDynamicStateMode::prepass);
            vkCmdPushConstants(command_buffer, depth_prepass_pipeline, VK_SHADER_STAGE_ALL, 0,
                               sizeof(ForwardPushConstants), &pc);

            vkCmdDrawIndexedIndirect(command_buffer, main_view_indirect_buffer.buffer, 0, frame.opaque_indirect_count,
                                     sizeof(VkDrawIndexedIndirectCommand));
        }

        if (frame.mask_indirect_count != 0) {
            bind_graphics_node(pipeline_graph_, depth_prepass_mask_pipeline_, command_buffer, samples_, 0, false);
            gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     depth_prepass_mask_pipeline);
            set_forward_dynamic_state(command_buffer, target_extent, ForwardDynamicStateMode::prepass);
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);
            vkCmdPushConstants(command_buffer, depth_prepass_mask_pipeline, VK_SHADER_STAGE_ALL, 0,
                               sizeof(ForwardPushConstants), &pc);

            auto const mask_offset =
                    static_cast<VkDeviceSize>(frame.opaque_indirect_count) * sizeof(VkDrawIndexedIndirectCommand);

            vkCmdDrawIndexedIndirect(command_buffer, main_view_indirect_buffer.buffer, mask_offset,
                                     frame.mask_indirect_count, sizeof(VkDrawIndexedIndirectCommand));
        }

        vkCmdEndRendering(command_buffer);

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, frame_query.query_pool,
                             (static_cast<std::uint32_t>(RenderStage::DepthPrepass) * 2) + 1);
    }
#pragma endregion

#pragma region Forward pass
    {
        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, frame_query.query_pool,
                             static_cast<std::uint32_t>(RenderStage::ForwardPass) * 2);

        VkRenderingAttachmentInfo hdr_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = hdr->view(),
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue =
                        VkClearValue{
                                .color =
                                        VkClearColorValue{
                                                .float32 =
                                                        {
                                                                0.015F,
                                                                0.025F,
                                                                0.050F,
                                                                1.0F,
                                                        },
                                        },
                        },
        };

        if (is_msaa) {
            hdr_attachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            hdr_attachment.resolveImageView = resolved_hdr->view();
            hdr_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            hdr_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        }

        VkRenderingAttachmentInfo const depth_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = depth->view(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {},
        };

        VkRenderingInfo const forward_rendering_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea =
                        VkRect2D{
                                .offset = {0, 0},
                                .extent = target_extent,
                        },
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 1,
                .pColorAttachments = &hdr_attachment,
                .pDepthAttachment = &depth_attachment,
                .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(command_buffer, &forward_rendering_info);
        vkCmdBeginQuery(command_buffer, pipeline_stat_queries_[frame_index].query_pool, 0, 0);
        vkCmdBindIndexBuffer(command_buffer, geometry_arena_.bindable_buffer(), 0, VK_INDEX_TYPE_UINT32);

        ForwardPushConstants const pc{
                .draw_buffer_address = main_view_draw_buffer.device_address,
                .transform_buffer_address = main_view_transform_buffer.device_address,
                .material_buffer_address = material_storage_.device_address(),
                .ubo_buffer_address = ubos_[frame_index].device_address,
                .lights_buffer_address = frame.lights_buffer.device_address,
                .light_count = frame.light_count,
                .padding = 0,
        };

        bind_graphics_node(pipeline_graph_, forward_pipeline_, command_buffer, samples_, 1, false);
        gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS, forward_pipeline);
        vkCmdPushConstants(command_buffer, forward_pipeline, VK_SHADER_STAGE_ALL, 0, sizeof(ForwardPushConstants), &pc);

        if (frame.opaque_indirect_count != 0) {
            set_forward_dynamic_state(command_buffer, target_extent, ForwardDynamicStateMode::main);
            vkCmdDrawIndexedIndirect(command_buffer, main_view_indirect_buffer.buffer, 0, frame.opaque_indirect_count,
                                     sizeof(VkDrawIndexedIndirectCommand));
        }

        if (frame.mask_indirect_count != 0) {
            set_forward_dynamic_state(command_buffer, target_extent, ForwardDynamicStateMode::main);
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);

            auto const mask_offset =
                    static_cast<VkDeviceSize>(frame.opaque_indirect_count) * sizeof(VkDrawIndexedIndirectCommand);

            vkCmdDrawIndexedIndirect(command_buffer, main_view_indirect_buffer.buffer, mask_offset,
                                     frame.mask_indirect_count, sizeof(VkDrawIndexedIndirectCommand));
        }

        if (frame.blend_indirect_count != 0) {
            bind_graphics_node(pipeline_graph_, forward_blend_pipeline_, command_buffer, samples_, 1, true);
            gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     forward_blend_pipeline);
            vkCmdPushConstants(command_buffer, forward_blend_pipeline, VK_SHADER_STAGE_ALL, 0,
                               sizeof(ForwardPushConstants), &pc);
            set_forward_dynamic_state(command_buffer, target_extent, ForwardDynamicStateMode::blend);
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);

            auto const blend_offset =
                    static_cast<VkDeviceSize>(frame.opaque_indirect_count + frame.mask_indirect_count) *
                    sizeof(VkDrawIndexedIndirectCommand);

            vkCmdDrawIndexedIndirect(command_buffer, main_view_indirect_buffer.buffer, blend_offset,
                                     frame.blend_indirect_count, sizeof(VkDrawIndexedIndirectCommand));
        }

        vkCmdEndQuery(command_buffer, pipeline_stat_queries_[frame_index].query_pool, 0);
        if (auto const light_icon_pipeline = resolve_layout(pipeline_graph_, light_icon_pipeline_);
            light_icon_pipeline != nullptr && debug_draw_light_icons_ && frame.light_count != 0) {
            bind_graphics_node(pipeline_graph_, light_icon_pipeline_, command_buffer, samples_, 1, true, false);
            gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS, light_icon_pipeline);

            vkCmdSetDepthTestEnable(command_buffer, VK_TRUE);
            vkCmdSetDepthWriteEnable(command_buffer, VK_FALSE);
            vkCmdSetDepthCompareOp(command_buffer, VK_COMPARE_OP_GREATER_OR_EQUAL);
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);

            LightIconPushConstants const light_pc{
                    .lights_buffer_address = frame.lights_buffer.device_address,
                    .ubo_buffer_address = ubos_[frame_index].device_address,
                    .light_count = frame.light_count,
                    .icon_texture_index = light_icon_texture_.index,
                    .sampler_index = sampler_storage_.linear_clamp().index,
                    .icon_world_size = light_icon_world_size_,
            };

            vkCmdPushConstants(command_buffer, light_icon_pipeline, VK_SHADER_STAGE_ALL, 0,
                               sizeof(LightIconPushConstants), &light_pc);

            auto const group_count = (frame.light_count + 31) / 32;
            vkCmdDrawMeshTasksEXT(command_buffer, group_count, 1, 1);
        }

        OverlayPolicy::render_debug(app, command_buffer, vp, frame_index);

        vkCmdEndRendering(command_buffer);

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, frame_query.query_pool,
                             (static_cast<std::uint32_t>(RenderStage::ForwardPass) * 2) + 1);
    }
#pragma endregion

    transition_hdr_to_shader_read(command_buffer, *resolved_hdr);

#pragma region Bloom pass
    auto bloom_texture_index = bloom_settings_.enabled ? frame.bloom_target.mip_slots[0].index : 0;
    {
        if (bloom_settings_.enabled) {
            vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, frame_query.query_pool,
                                 static_cast<std::uint32_t>(RenderStage::BloomPass) * 2);
            record_bloom_pass(command_buffer, frame_index, frame.forward_target.resolved_hdr(), target_extent);
            vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, frame_query.query_pool,
                                 (static_cast<std::uint32_t>(RenderStage::BloomPass) * 2) + 1);
        }
    }
#pragma endregion

    transition_swapchain_to_attachment(command_buffer, swapchain_image.image);

#pragma region Composition for swapchain
    {
        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, frame_query.query_pool,
                             static_cast<std::uint32_t>(RenderStage::Composition) * 2);

        VkRenderingAttachmentInfo const swapchain_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = swapchain_image.view,
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {},
        };

        VkRenderingInfo const composite_rendering_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea =
                        VkRect2D{
                                .offset = {0, 0},
                                .extent = swapchain_image.extent,
                        },
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 1,
                .pColorAttachments = &swapchain_attachment,
                .pDepthAttachment = nullptr,
                .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(command_buffer, &composite_rendering_info);
        bind_graphics_node(pipeline_graph_, composite_pipeline_, command_buffer, VK_SAMPLE_COUNT_1_BIT, 1, false);
        gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS, composite_pipeline);
        set_composite_dynamic_state(command_buffer, swapchain_image.extent);

        CompositePushConstants const composite_pc{
                .hdr_texture_index = frame.forward_target.resolved_hdr().index,
                .bloom_texture_index = bloom_texture_index,
                .sampler_index = sampler_storage_.linear_clamp().index,
                .exposure = 1.0F, // exposure
                .bloom_intensity = bloom_settings_.enabled ? bloom_settings_.intensity : 0.0f,
        };

        vkCmdPushConstants(command_buffer, composite_pipeline, VK_SHADER_STAGE_ALL, 0, sizeof(composite_pc),
                           &composite_pc);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);

        OverlayPolicy::render_ui(app, command_buffer, frame_index);

        vkCmdEndRendering(command_buffer);

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, frame_query.query_pool,
                             (static_cast<std::uint32_t>(RenderStage::Composition) * 2) + 1);
    }
#pragma endregion

    bool const screenshot_recorded = screenshot_.record(context_, command_buffer, swapchain_image.image,
                                                        swapchain_image.format, swapchain_image.extent, frame_index);

    if (!screenshot_recorded) {
        transition_swapchain_to_present(command_buffer, swapchain_image.image);
    }

    // Full frame end
    vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, frame_query.query_pool,
                         (static_cast<std::uint32_t>(RenderStage::FullFrame) * 2) + 1);

    frame_query.has_results = true;
    pipeline_stat_queries_[frame_index].has_results = true;

    return {};
}

template auto Renderer::record_frame<ApplicationOverlayPolicy>(VkCommandBuffer, SwapchainImage const &, std::uint32_t,
                                                               Application const &, glm::mat4 const &)
        -> std::expected<void, RendererError>;


auto Renderer::resize(VkExtent2D extent) -> std::expected<void, RendererError> {
    if (extent.width == 0 || extent.height == 0) {
        return {};
    }

    if (extent.width == extent_.width && extent.height == extent_.height) {
        return {};
    }

    std::vector<ForwardTarget> replacements;
    replacements.reserve(frames_.size());

    for (std::size_t index = 0; index < frames_.size(); ++index) {
        auto const target_name = std::string{"renderer.forward_target_"} + std::to_string(index);

        auto replacement = ForwardTarget::create(image_storage_, ForwardTargetCreateInfo{
                                                                         .extent = extent,
                                                                         .hdr_format = hdr_format_,
                                                                         .depth_format = depth_format_,
                                                                         .samples = samples_,
                                                                         .debug_name = target_name,
                                                                 });

        if (!replacement) {
            for (auto &created: replacements) {
                created.destroy(image_storage_);
            }

            return std::unexpected(RendererError{
                    .type = RendererErrorType::forward_target_error,
                    .cause = ErrorCause{Boxed<ForwardTargetError>{replacement.error()}},
            });
        }

        replacements.push_back(std::move(*replacement));
    }

    for (std::size_t index = 0; index < frames_.size(); ++index) {
        frames_[index].forward_target.destroy(image_storage_);

        frames_[index].forward_target = std::move(replacements[index]);
    }

    extent_ = extent;

    return {};
}

void Renderer::queue_render_thread_event(std::move_only_function<void()> &&task) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    event_queue_.push(std::move(task));
    queued_events_.fetch_add(1);
}

void Renderer::drain_event_queue() {
    if (queued_events_.load(std::memory_order_relaxed) == 0) [[likely]] {
        return;
    }

    std::queue<std::move_only_function<void()>> local_queue;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::swap(event_queue_, local_queue);
    }

    while (!local_queue.empty()) {
        local_queue.front()();
        local_queue.pop();
    }
}


auto Renderer::wait_idle() -> std::expected<void, RendererError> {
    auto result = vkDeviceWaitIdle(context_.device);
    return result == VK_SUCCESS ? std::expected<void, RendererError>{}
                                : std::unexpected<RendererError>(RendererError{
                                          .type = RendererErrorType::device_error,
                                          .cause = ErrorCause{Boxed<DeviceError>{DeviceError{
                                                  .type = DeviceError::Type::Unknown,
                                                  .message = FlyString{"Could not wait"},
                                                  .vk_result = result,
                                          }}},
                                  });
}

auto Renderer::mesh_slot(MeshHandle handle) noexcept -> MeshSlot * {
    if (handle.index >= meshes_.size()) {
        return nullptr;
    }

    auto &slot = meshes_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}

auto Renderer::mesh_slot(MeshHandle handle) const noexcept -> MeshSlot const * {
    if (handle.index >= meshes_.size()) {
        return nullptr;
    }

    auto const &slot = meshes_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}

auto Renderer::model_slot(ModelHandle handle) noexcept -> ModelSlot * {
    if (handle.index >= models_.size()) {
        return nullptr;
    }

    auto &slot = models_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}

auto Renderer::model_slot(ModelHandle handle) const noexcept -> ModelSlot const * {
    if (handle.index >= models_.size()) {
        return nullptr;
    }

    auto const &slot = models_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}

void Renderer::record_bloom_pass(VkCommandBuffer command_buffer, std::uint32_t frame_index, ImageHandle input_hdr,
                                 VkExtent2D target_extent) {
    auto const &bloom = frames_[frame_index].bloom_target;

    auto const *bloom_image = image_storage_.get(bloom.image);
    if (bloom_image == nullptr || !bloom_image->valid()) {
        return;
    }

    auto const bloom_downsample_pipeline = resolve_layout(pipeline_graph_, bloom_downsample_pipeline_);
    auto const bloom_upsample_pipeline = resolve_layout(pipeline_graph_, bloom_upsample_pipeline_);

    if (bloom_downsample_pipeline == VK_NULL_HANDLE || bloom_upsample_pipeline == VK_NULL_HANDLE) {
        return;
    }

    auto const linear_clamp_sampler = sampler_storage_.linear_clamp().index;

    transition_image_layout(command_buffer, bloom_image->image(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0,
                            VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 4);

    bind_compute_node(pipeline_graph_, bloom_downsample_pipeline_, command_buffer);
    gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, bloom_downsample_pipeline);

    VkExtent2D const mip0_extent = bloom_image->mip_extent(0);


    DownsamplePushConstants const downsample_pc{
            .src_texture_index = input_hdr.index,
            .linear_sampler_index = linear_clamp_sampler,
            .dst_mip0_storage_index = bloom.mip_slots[0].index,
            .dst_mip1_storage_index = bloom.mip_slots[1].index,
            .dst_mip2_storage_index = bloom.mip_slots[2].index,
            .dst_mip3_storage_index = bloom.mip_slots[3].index,
            .src_texel_size_x = 1.0F / static_cast<float>(target_extent.width),
            .src_texel_size_y = 1.0F / static_cast<float>(target_extent.height),
            .mip0_size_x = static_cast<std::int32_t>(mip0_extent.width),
            .mip0_size_y = static_cast<std::int32_t>(mip0_extent.height),
            .threshold = bloom_settings_.threshold,
            .knee = bloom_settings_.knee,
    };

    vkCmdPushConstants(command_buffer, bloom_downsample_pipeline, VK_SHADER_STAGE_ALL, 0,
                       sizeof(DownsamplePushConstants), &downsample_pc);

    std::uint32_t const downsample_groups_x = (mip0_extent.width + 15U) / 16U;
    std::uint32_t const downsample_groups_y = (mip0_extent.height + 15U) / 16U;
    vkCmdDispatch(command_buffer, downsample_groups_x, downsample_groups_y, 1);

    transition_image_layout(command_buffer, bloom_image->image(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT, 0, 4);

    bind_compute_node(pipeline_graph_, bloom_upsample_pipeline_, command_buffer);
    gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, bloom_upsample_pipeline);

    for (std::int32_t i = 2; i >= 0; --i) {
        auto const target_mip = static_cast<std::uint32_t>(i);
        auto const lower_mip = target_mip + 1U;

        VkExtent2D const lower_extent = bloom_image->mip_extent(lower_mip);
        VkExtent2D const target_extent_mip = bloom_image->mip_extent(target_mip);

        UpsamplePushConstants const upsample_pc{
                .lower_mip_texture_index = bloom.mip_slots[lower_mip].index,
                .target_mip_storage_index = bloom.mip_slots[target_mip].index,
                .linear_sampler_index = linear_clamp_sampler,
                .lower_texel_size_x = 1.0F / static_cast<float>(lower_extent.width),
                .lower_texel_size_y = 1.0F / static_cast<float>(lower_extent.height),
                .target_size_x = static_cast<std::int32_t>(target_extent_mip.width),
                .target_size_y = static_cast<std::int32_t>(target_extent_mip.height),
                .filter_radius = bloom_settings_.filter_radius,
        };

        vkCmdPushConstants(command_buffer, bloom_upsample_pipeline, VK_SHADER_STAGE_ALL, 0,
                           sizeof(UpsamplePushConstants), &upsample_pc);

        std::uint32_t const upsample_groups_x = (target_extent_mip.width + 15U) / 16U;
        std::uint32_t const upsample_groups_y = (target_extent_mip.height + 15U) / 16U;
        vkCmdDispatch(command_buffer, upsample_groups_x, upsample_groups_y, 1);

        transition_image_layout(command_buffer, bloom_image->image(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_WRITE_BIT,
                                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                                target_mip, 1);
    }

    transition_image_layout(command_buffer, bloom_image->image(), VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                            VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
}

auto Renderer::upload_frame_data(VkCommandBuffer command_buffer, RendererFrame &frame)
        -> std::expected<void, RendererError> {

    auto const draw_size = static_cast<VkDeviceSize>(frame.draws.size()) * sizeof(GpuDraw);
    auto const transform_size = static_cast<VkDeviceSize>(frame.transforms.size()) * sizeof(glm::mat4);
    auto const indirect_size =
            static_cast<VkDeviceSize>(frame.indirect_commands.size()) * sizeof(VkDrawIndexedIndirectCommand);
    auto const batch_bounds_size = static_cast<VkDeviceSize>(frame.batch_bounds.size()) * sizeof(GpuCullBounds);

    if (draw_size != 0) {
        auto const data_span = std::as_bytes(std::span{frame.draws});
        if (auto const result = frame.upload_buffer.write(frame.draw_upload_offset, data_span); !result) {
            return std::unexpected(make_error(RendererErrorType::device_error));
        }
    }

    if (transform_size != 0) {
        auto const data_span = std::as_bytes(std::span{frame.transforms});
        if (auto const result = frame.upload_buffer.write(frame.transform_upload_offset, data_span); !result) {
            return std::unexpected(make_error(RendererErrorType::device_error));
        }
    }

    if (indirect_size != 0) {
        auto const data_span = std::as_bytes(std::span{frame.indirect_commands});
        if (auto const result = frame.upload_buffer.write(frame.indirect_upload_offset, data_span); !result) {
            return std::unexpected(make_error(RendererErrorType::device_error));
        }
    }

    if (batch_bounds_size != 0) {
        auto const data_span = std::as_bytes(std::span{frame.batch_bounds});
        if (auto const result = frame.upload_buffer.write(frame.batch_bounds_upload_offset, data_span); !result) {
            return std::unexpected(make_error(RendererErrorType::device_error));
        }
    }

    struct CopyOperation {
        VkBuffer destination = VK_NULL_HANDLE;
        VkBufferCopy2 region{};
    };

    std::array<CopyOperation, 4> copies{};
    std::uint32_t copy_count = 0;

    if (draw_size != 0) {
        copies[copy_count++] = CopyOperation{
                .destination = frame.draw_buffer.buffer,
                .region =
                        VkBufferCopy2{
                                .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                .pNext = nullptr,
                                .srcOffset = frame.draw_upload_offset,
                                .dstOffset = 0,
                                .size = draw_size,
                        },
        };
    }

    if (transform_size != 0) {
        copies[copy_count++] = CopyOperation{
                .destination = frame.transform_buffer.buffer,
                .region =
                        VkBufferCopy2{
                                .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                .pNext = nullptr,
                                .srcOffset = frame.transform_upload_offset,
                                .dstOffset = 0,
                                .size = transform_size,
                        },
        };
    }

    if (indirect_size != 0) {
        copies[copy_count++] = CopyOperation{
                .destination = frame.indirect_buffer.buffer,
                .region =
                        VkBufferCopy2{
                                .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                .pNext = nullptr,
                                .srcOffset = frame.indirect_upload_offset,
                                .dstOffset = 0,
                                .size = indirect_size,
                        },
        };
    }

    if (batch_bounds_size != 0) {
        copies[copy_count++] = CopyOperation{
                .destination = frame.batch_bounds_buffer.buffer,
                .region =
                        VkBufferCopy2{
                                .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                .pNext = nullptr,
                                .srcOffset = frame.batch_bounds_upload_offset,
                                .dstOffset = 0,
                                .size = batch_bounds_size,
                        },
        };
    }

    for (std::uint32_t index = 0; index < copy_count; ++index) {
        auto const &copy = copies[index];

        VkCopyBufferInfo2 const copy_info{
                .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                .pNext = nullptr,
                .srcBuffer = frame.upload_buffer.buffer,
                .dstBuffer = copy.destination,
                .regionCount = 1,
                .pRegions = &copy.region,
        };

        vkCmdCopyBuffer2(command_buffer, &copy_info);
    }

    if (copy_count == 0) {
        return {};
    }

    std::array<VkBufferMemoryBarrier2, 4> barriers{};
    std::uint32_t barrier_count = 0;

    if (draw_size != 0) {
        barriers[barrier_count++] = VkBufferMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = frame.draw_buffer.buffer,
                .offset = 0,
                .size = draw_size,
        };
    }

    if (transform_size != 0) {
        barriers[barrier_count++] = VkBufferMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = frame.transform_buffer.buffer,
                .offset = 0,
                .size = transform_size,
        };
    }

    // Read by two consumers now: the shadow pass's raw vkCmdDrawIndexedIndirect
    // (DRAW_INDIRECT / INDIRECT_COMMAND_READ) and mainCs's src_indirect Ptr<>,
    // which reads each batch's un-culled firstInstance/instanceCount
    // (COMPUTE_SHADER / SHADER_STORAGE_READ).
    if (indirect_size != 0) {
        barriers[barrier_count++] = VkBufferMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = frame.indirect_buffer.buffer,
                .offset = 0,
                .size = indirect_size,
        };
    }

    // batch_bounds_buffer is a read-only input to the frustum-cull compute
    // pass. culled_indirect_buffer and visible_draw_buffer/
    // visible_transform_buffer need no transfer barrier here -- they are
    // never host-uploaded; mainCs is their sole writer (see the dispatch's
    // own post-cull barrier in prepare_frame).
    if (batch_bounds_size != 0) {
        barriers[barrier_count++] = VkBufferMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = frame.batch_bounds_buffer.buffer,
                .offset = 0,
                .size = batch_bounds_size,
        };
    }

    VkDependencyInfo const dependency_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = barrier_count,
            .pBufferMemoryBarriers = barriers.data(),
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr,
    };

    vkCmdPipelineBarrier2(command_buffer, &dependency_info);

    return {};
}

auto Renderer::clear_submissions() noexcept -> void {
    submissions_.clear();
    model_submissions_.clear();
    point_light_submissions_.clear();
    spot_light_submissions_.clear();
}
