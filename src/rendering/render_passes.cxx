#include "rendering/render_passes.hxx"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

#include "gpu/vk_barrier.hxx"
#include "rendering/render_stage.hxx"
#include "rendering/shadow_cascades.hxx"
#include "shader_push_constants.hxx"

namespace render_pass {

    namespace detail {

        enum class ForwardDynamicStateMode : std::uint8_t {
            prepass,
            main,
            blend,
        };

        [[nodiscard]] auto make_error(RendererErrorType type) -> RendererError { return RendererError{.type = type}; }

        auto transition_forward_target_to_attachments(VkCommandBuffer command_buffer, Image const &hdr,
                                                      Image const &depth, Image const *resolved_hdr,
                                                      Image const *resolved_depth) noexcept -> void {
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
                    .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
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
                        .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
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
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
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
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &barrier,
            };

            vkCmdPipelineBarrier2(command_buffer, &dependency_info);
        }

        auto transition_shadow_atlas_to_attachment(VkCommandBuffer command_buffer, Image const &atlas,
                                                   bool preserve_contents) noexcept -> void {
            VkImageMemoryBarrier2 const barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask =
                            preserve_contents ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
                    .srcAccessMask = preserve_contents ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
                    .dstStageMask =
                            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .oldLayout =
                            preserve_contents ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
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
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &barrier,
            };

            vkCmdPipelineBarrier2(command_buffer, &dependency_info);
        }

        auto transition_shadow_atlas_to_shader_read(VkCommandBuffer command_buffer, Image const &atlas) noexcept
                -> void {
            VkImageMemoryBarrier2 const barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask =
                            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
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
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &barrier,
            };

            vkCmdPipelineBarrier2(command_buffer, &dependency_info);
        }

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

            VkRect2D const scissor{.offset = {0, 0}, .extent = extent};

            vkCmdSetViewportWithCount(command_buffer, 1, &viewport);
            vkCmdSetScissorWithCount(command_buffer, 1, &scissor);
            vkCmdSetPrimitiveTopology(command_buffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            vkCmdSetPrimitiveRestartEnable(command_buffer, VK_FALSE);
            vkCmdSetRasterizerDiscardEnable(command_buffer, VK_FALSE);
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_BACK_BIT);
            vkCmdSetFrontFace(command_buffer, VK_FRONT_FACE_CLOCKWISE);
            vkCmdSetDepthTestEnable(command_buffer, VK_TRUE);
            vkCmdSetDepthWriteEnable(command_buffer, mode == ForwardDynamicStateMode::blend ? VK_FALSE : VK_TRUE);
            vkCmdSetDepthCompareOp(command_buffer, mode == ForwardDynamicStateMode::main
                                                           ? VK_COMPARE_OP_EQUAL
                                                           : VK_COMPARE_OP_GREATER_OR_EQUAL);
            vkCmdSetDepthBiasEnable(command_buffer, VK_FALSE);
            vkCmdSetStencilTestEnable(command_buffer, VK_FALSE);
        }

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
            vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);
            vkCmdSetFrontFace(command_buffer, VK_FRONT_FACE_CLOCKWISE);
            vkCmdSetDepthTestEnable(command_buffer, VK_TRUE);
            vkCmdSetDepthWriteEnable(command_buffer, VK_TRUE);
            vkCmdSetDepthCompareOp(command_buffer, VK_COMPARE_OP_GREATER_OR_EQUAL);
            vkCmdSetDepthBiasEnable(command_buffer, VK_TRUE);
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

            VkRect2D const scissor{.offset = {0, 0}, .extent = extent};

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

        auto set_shader_object_color_blend_state(VkCommandBuffer command_buffer, std::uint32_t attachment_count,
                                                 bool blending) noexcept -> void {
            constexpr std::uint32_t max_supported_attachments = 8;

            if (attachment_count == 0) {
                return;
            }

            attachment_count = std::min(attachment_count, max_supported_attachments);

            std::array<VkBool32, max_supported_attachments> blend_enable{};
            blend_enable.fill(blending ? VK_TRUE : VK_FALSE);

            std::array<VkColorBlendEquationEXT, max_supported_attachments> blend_equation{};
            blend_equation.fill(VkColorBlendEquationEXT{
                    .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                    .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                    .colorBlendOp = VK_BLEND_OP_ADD,
                    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                    .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                    .alphaBlendOp = VK_BLEND_OP_ADD,
            });

            std::array<VkColorComponentFlags, max_supported_attachments> write_mask{};
            write_mask.fill(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                            VK_COLOR_COMPONENT_A_BIT);

            vkCmdSetColorBlendEnableEXT(command_buffer, 0, attachment_count, blend_enable.data());
            vkCmdSetColorBlendEquationEXT(command_buffer, 0, attachment_count, blend_equation.data());
            vkCmdSetColorWriteMaskEXT(command_buffer, 0, attachment_count, write_mask.data());
        }

        [[nodiscard]] auto resolve_layout(PipelineGraphRepository const &graph, PipelineNodeHandle handle) noexcept
                -> VkPipelineLayout {
            if (auto const *shader_objects = graph.resolve_shader_objects(handle); shader_objects != nullptr) {
                return shader_objects->layout();
            }

            return VK_NULL_HANDLE;
        }

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
        }

        auto bind_compute_node(PipelineGraphRepository const &graph, PipelineNodeHandle handle,
                               VkCommandBuffer command_buffer) noexcept -> void {
            if (auto const *shader_objects = graph.resolve_shader_objects(handle); shader_objects != nullptr) {
                shader_objects->bind(command_buffer);
                return;
            }
        }

    } // namespace detail

    auto prepare_forward_targets(Context const &context, ForwardTargets const &targets) noexcept -> void {
        detail::transition_forward_target_to_attachments(context.command_buffer, targets.hdr, targets.depth,
                                                         targets.resolved_hdr, targets.resolved_depth);
    }

    auto shadow(Context const &context, ShadowPassInfo const &info) -> std::expected<void, RendererError> {
        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::ShadowPass);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, context.timestamp_query_pool,
                             stage * 2);

        if (info.update_mask == 0) {
            vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                 context.timestamp_query_pool, stage * 2 + 1);
            return {};
        }

        auto has_dirty_draws = [&](auto const &counts) noexcept {
            for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
                if ((info.update_mask & (1U << cascade)) != 0 && counts[cascade] != 0) {
                    return true;
                }
            }
            return false;
        };

        auto const has_dirty_opaque = has_dirty_draws(info.opaque_cascade_counts);
        auto const has_dirty_mask = has_dirty_draws(info.mask_cascade_counts);
        auto const opaque_layout = has_dirty_opaque
                                           ? detail::resolve_layout(context.pipeline_graph, info.opaque_pipeline)
                                           : VK_NULL_HANDLE;
        auto const mask_layout =
                has_dirty_mask ? detail::resolve_layout(context.pipeline_graph, info.mask_pipeline) : VK_NULL_HANDLE;

        if ((has_dirty_opaque && opaque_layout == VK_NULL_HANDLE) ||
            (has_dirty_mask && mask_layout == VK_NULL_HANDLE)) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        detail::transition_shadow_atlas_to_attachment(context.command_buffer, info.shadow_atlas,
                                                      info.preserve_contents);

        VkRenderingAttachmentInfo shadow_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = info.shadow_atlas.view(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = info.preserve_contents ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        };

        VkRenderingInfo const rendering_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea = VkRect2D{.offset = {0, 0}, .extent = {shadow_atlas_width, shadow_atlas_height}},
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 0,
                .pColorAttachments = nullptr,
                .pDepthAttachment = &shadow_attachment,
                .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(context.command_buffer, &rendering_info);

        // LOAD keeps every cached tile intact. Explicitly clear only the tiles
        // that this frame is about to rebuild. Reverse-Z's empty value is zero.
        VkClearAttachment const clear_attachment{
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .colorAttachment = 0,
                .clearValue = {.depthStencil = {.depth = 0.0F, .stencil = 0}},
        };
        for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
            if ((info.update_mask & (1U << cascade)) == 0) {
                continue;
            }

            VkClearRect const clear_rect{
                    .rect =
                            VkRect2D{
                                    .offset = {static_cast<std::int32_t>(shadow_cascade_offset_x[cascade]), 0},
                                    .extent = {shadow_cascade_resolutions[cascade],
                                               shadow_cascade_resolutions[cascade]},
                            },
                    .baseArrayLayer = 0,
                    .layerCount = 1,
            };
            vkCmdClearAttachments(context.command_buffer, 1, &clear_attachment, 1, &clear_rect);
        }

        if (has_dirty_opaque || has_dirty_mask) {
            vkCmdBindIndexBuffer(context.command_buffer, info.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        }

        if (has_dirty_opaque) {
            detail::bind_graphics_node(context.pipeline_graph, info.opaque_pipeline, context.command_buffer,
                                       VK_SAMPLE_COUNT_1_BIT, 0, false);
            context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        opaque_layout);

            ShadowPushConstants pc{
                    .draws_address = info.draws.draws.device_address,
                    .transforms_address = info.draws.transforms.device_address,
                    .materials_address = info.materials_address,
                    .ubo_address = info.ubo_address,
                    .lights_address = info.lights_address,
                    .light_count = 0,
                    ._padding = 0,
                    .cascade_index = 0,
                    .padding = 0,
            };

            vkCmdPushConstants(context.command_buffer, opaque_layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);

            for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
                if ((info.update_mask & (1U << cascade)) == 0) {
                    continue;
                }

                auto const cascade_draw_count = info.opaque_cascade_counts[cascade];
                if (cascade_draw_count == 0) {
                    continue;
                }

                detail::set_shadow_dynamic_state(context.command_buffer, cascade, info.depth_bias_constant,
                                                 info.depth_bias_slope);
                vkCmdPushConstants(context.command_buffer, opaque_layout, VK_SHADER_STAGE_ALL,
                                   offsetof(ShadowPushConstants, cascade_index), sizeof(cascade), &cascade);
                vkCmdDrawIndexedIndirect(context.command_buffer, info.draws.indirect.buffer, 0, cascade_draw_count,
                                         sizeof(VkDrawIndexedIndirectCommand));
            }
        }

        if (has_dirty_mask) {
            detail::bind_graphics_node(context.pipeline_graph, info.mask_pipeline, context.command_buffer,
                                       VK_SAMPLE_COUNT_1_BIT, 0, false);
            context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        mask_layout);

            ShadowPushConstants pc{
                    .draws_address = info.draws.draws.device_address,
                    .transforms_address = info.draws.transforms.device_address,
                    .materials_address = info.materials_address,
                    .ubo_address = info.ubo_address,
                    .lights_address = info.lights_address,
                    .light_count = 0,
                    ._padding = 0,
                    .cascade_index = 0,
                    .padding = 0,
            };

            vkCmdPushConstants(context.command_buffer, mask_layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);

            auto const mask_offset =
                    static_cast<VkDeviceSize>(info.counts.opaque) * sizeof(VkDrawIndexedIndirectCommand);

            for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
                if ((info.update_mask & (1U << cascade)) == 0) {
                    continue;
                }

                auto const cascade_draw_count = info.mask_cascade_counts[cascade];
                if (cascade_draw_count == 0) {
                    continue;
                }

                detail::set_shadow_dynamic_state(context.command_buffer, cascade, info.depth_bias_constant,
                                                 info.depth_bias_slope);
                vkCmdPushConstants(context.command_buffer, mask_layout, VK_SHADER_STAGE_ALL,
                                   offsetof(ShadowPushConstants, cascade_index), sizeof(cascade), &cascade);
                vkCmdDrawIndexedIndirect(context.command_buffer, info.draws.indirect.buffer, mask_offset,
                                         cascade_draw_count, sizeof(VkDrawIndexedIndirectCommand));
            }
        }

        vkCmdEndRendering(context.command_buffer);
        detail::transition_shadow_atlas_to_shader_read(context.command_buffer, info.shadow_atlas);

        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);
        return {};
    }

    auto depth_prepass(Context const &context, DepthPrepassInfo const &info) -> std::expected<void, RendererError> {
        auto const opaque_layout = info.counts.opaque != 0
                                           ? detail::resolve_layout(context.pipeline_graph, info.opaque_pipeline)
                                           : VK_NULL_HANDLE;
        auto const mask_layout = info.counts.mask != 0
                                         ? detail::resolve_layout(context.pipeline_graph, info.mask_pipeline)
                                         : VK_NULL_HANDLE;

        if ((info.counts.opaque != 0 && opaque_layout == VK_NULL_HANDLE) ||
            (info.counts.mask != 0 && mask_layout == VK_NULL_HANDLE)) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::DepthPrepass);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, context.timestamp_query_pool,
                             stage * 2);

        VkRenderingAttachmentInfo depth_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = info.depth.view(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {},
        };

        if (info.resolved_depth != nullptr) {
            depth_attachment.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
            depth_attachment.resolveImageView = info.resolved_depth->view();
            depth_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        }

        VkRenderingInfo const rendering_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea = VkRect2D{.offset = {0, 0}, .extent = info.extent},
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 0,
                .pColorAttachments = nullptr,
                .pDepthAttachment = &depth_attachment,
                .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(context.command_buffer, &rendering_info);
        vkCmdBindIndexBuffer(context.command_buffer, info.index_buffer, 0, VK_INDEX_TYPE_UINT32);

        ForwardPushConstants const pc{
                .draws_address = info.draws.draws.device_address,
                .transforms_address = info.draws.transforms.device_address,
                .materials_address = info.materials_address,
                .ubo_address = info.ubo_address,
                .lights_address = info.lights_address,
                .light_count = 0,
                ._padding = 0,
        };

        if (info.counts.opaque != 0) {
            detail::bind_graphics_node(context.pipeline_graph, info.opaque_pipeline, context.command_buffer,
                                       info.samples, 0, false);
            context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        opaque_layout);
            detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                              detail::ForwardDynamicStateMode::prepass);
            vkCmdPushConstants(context.command_buffer, opaque_layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
            vkCmdDrawIndexedIndirect(context.command_buffer, info.draws.indirect.buffer, 0, info.counts.opaque,
                                     sizeof(VkDrawIndexedIndirectCommand));
        }

        if (info.counts.mask != 0) {
            detail::bind_graphics_node(context.pipeline_graph, info.mask_pipeline, context.command_buffer, info.samples,
                                       0, false);
            context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        mask_layout);
            detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                              detail::ForwardDynamicStateMode::prepass);
            vkCmdSetCullMode(context.command_buffer, VK_CULL_MODE_NONE);
            vkCmdPushConstants(context.command_buffer, mask_layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);

            auto const mask_offset =
                    static_cast<VkDeviceSize>(info.counts.opaque) * sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(context.command_buffer, info.draws.indirect.buffer, mask_offset, info.counts.mask,
                                     sizeof(VkDrawIndexedIndirectCommand));
        }

        vkCmdEndRendering(context.command_buffer);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);
        return {};
    }

    auto ambient_occlusion(Context const &context, AmbientOcclusionInfo const &info)
            -> std::expected<std::optional<AoTextureIndex>, RendererError> {
        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::AmbientOcclusion);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             context.timestamp_query_pool, stage * 2);

        if (!info.enabled) {
            vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                 context.timestamp_query_pool, stage * 2 + 1);
            return std::optional<AoTextureIndex>{};
        }

        auto const gtao_layout = detail::resolve_layout(context.pipeline_graph, info.gtao_pipeline);
        auto const denoise_layout = detail::resolve_layout(context.pipeline_graph, info.denoise_pipeline);

        if (gtao_layout == VK_NULL_HANDLE || denoise_layout == VK_NULL_HANDLE) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        // depth arrives in DEPTH_ATTACHMENT_OPTIMAL (written by the depth
        // prepass); both compute passes below only ever read it.
        transition_image_layout(context.command_buffer, info.depth.image(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1);

        transition_image_layout(context.command_buffer, info.raw_ao.image(), VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, VK_ACCESS_2_SHADER_WRITE_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

        detail::bind_compute_node(context.pipeline_graph, info.gtao_pipeline, context.command_buffer);
        context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    gtao_layout);

        GtaoPushConstants const gtao_pc{
                .ubo_address = info.ubo_address,
                .depth_texture_index = info.depth_texture_index,
                .output_storage_index = info.raw_ao_texture_index,
                .point_sampler_index = info.point_sampler_index,
                .width = info.extent.width,
                .height = info.extent.height,
                .inv_extent_x = 1.0F / static_cast<float>(info.extent.width),
                .inv_extent_y = 1.0F / static_cast<float>(info.extent.height),
                .radius_view = info.radius_view,
                .falloff_range = info.falloff_range,
                .slice_count = info.slice_count,
                .step_count = info.step_count,
        };

        vkCmdPushConstants(context.command_buffer, gtao_layout, VK_SHADER_STAGE_ALL, 0, sizeof(gtao_pc), &gtao_pc);
        vkCmdDispatch(context.command_buffer, (info.extent.width + 7U) / 8U, (info.extent.height + 7U) / 8U, 1);

        transition_image_layout(context.command_buffer, info.raw_ao.image(), VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                                VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

        transition_image_layout(context.command_buffer, info.denoised_ao.image(), VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, VK_ACCESS_2_SHADER_WRITE_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

        detail::bind_compute_node(context.pipeline_graph, info.denoise_pipeline, context.command_buffer);
        context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    denoise_layout);

        GtaoDenoisePushConstants const denoise_pc{
                .ubo_address = info.ubo_address,
                .input_texture_index = info.raw_ao_texture_index,
                .depth_texture_index = info.depth_texture_index,
                .output_storage_index = info.denoised_ao_texture_index,
                .point_sampler_index = info.point_sampler_index,
                .width = info.extent.width,
                .height = info.extent.height,
                .inv_extent_x = 1.0F / static_cast<float>(info.extent.width),
                .inv_extent_y = 1.0F / static_cast<float>(info.extent.height),
                .depth_sigma = info.denoise_depth_sigma,
        };

        vkCmdPushConstants(context.command_buffer, denoise_layout, VK_SHADER_STAGE_ALL, 0, sizeof(denoise_pc),
                           &denoise_pc);
        vkCmdDispatch(context.command_buffer, (info.extent.width + 7U) / 8U, (info.extent.height + 7U) / 8U, 1);

        transition_image_layout(context.command_buffer, info.denoised_ao.image(), VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                                VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

        // Restore the layout forward_geometry's LOAD_OP_LOAD depth
        // attachment expects.
        transition_image_layout(
                context.command_buffer, info.depth.image(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_SHADER_READ_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1);

        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);

        return std::optional<AoTextureIndex>{AoTextureIndex{.index = info.denoised_ao_texture_index}};
    }

    auto forward_geometry(Context const &context, ForwardGeometryInfo const &info, Callback debug_overlay)
            -> std::expected<HdrTextureIndex, RendererError> {
        auto const opaque_layout = detail::resolve_layout(context.pipeline_graph, info.opaque_pipeline);
        if (opaque_layout == VK_NULL_HANDLE) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        auto const blend_layout = info.counts.blend != 0
                                          ? detail::resolve_layout(context.pipeline_graph, info.blend_pipeline)
                                          : VK_NULL_HANDLE;
        if (info.counts.blend != 0 && blend_layout == VK_NULL_HANDLE) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::ForwardPass);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             context.timestamp_query_pool, stage * 2);

        VkRenderingAttachmentInfo hdr_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = info.hdr.view(),
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = VkClearValue{.color = VkClearColorValue{.float32 = {0.015F, 0.025F, 0.050F, 1.0F}}},
        };

        if (info.resolved_hdr != nullptr) {
            hdr_attachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            hdr_attachment.resolveImageView = info.resolved_hdr->view();
            hdr_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            hdr_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        }

        VkRenderingAttachmentInfo const depth_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = info.depth.view(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {},
        };

        VkRenderingInfo const rendering_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea = VkRect2D{.offset = {0, 0}, .extent = info.extent},
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 1,
                .pColorAttachments = &hdr_attachment,
                .pDepthAttachment = &depth_attachment,
                .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(context.command_buffer, &rendering_info);
        vkCmdBeginQuery(context.command_buffer, info.pipeline_statistics_query_pool, 0, 0);
        vkCmdBindIndexBuffer(context.command_buffer, info.index_buffer, 0, VK_INDEX_TYPE_UINT32);

        ForwardPushConstants const pc{
                .draws_address = info.draws.draws.device_address,
                .transforms_address = info.draws.transforms.device_address,
                .materials_address = info.materials_address,
                .ubo_address = info.ubo_address,
                .lights_address = info.lights_address,
                .light_count = info.light_count,
                ._padding = 0,
                .ao_texture_index = info.ao_texture_index,
                .ao_sampler_index = info.ao_sampler_index,
                .screen_size_x = static_cast<float>(info.extent.width),
                .screen_size_y = static_cast<float>(info.extent.height),
        };

        detail::bind_graphics_node(context.pipeline_graph, info.opaque_pipeline, context.command_buffer, info.samples,
                                   1, false);
        context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    opaque_layout);
        vkCmdPushConstants(context.command_buffer, opaque_layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);

        if (info.counts.opaque != 0) {
            detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                              detail::ForwardDynamicStateMode::main);
            vkCmdDrawIndexedIndirect(context.command_buffer, info.draws.indirect.buffer, 0, info.counts.opaque,
                                     sizeof(VkDrawIndexedIndirectCommand));
        }

        if (info.counts.mask != 0) {
            detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                              detail::ForwardDynamicStateMode::main);
            vkCmdSetCullMode(context.command_buffer, VK_CULL_MODE_NONE);
            auto const mask_offset =
                    static_cast<VkDeviceSize>(info.counts.opaque) * sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(context.command_buffer, info.draws.indirect.buffer, mask_offset, info.counts.mask,
                                     sizeof(VkDrawIndexedIndirectCommand));
        }

        if (info.counts.blend != 0) {
            detail::bind_graphics_node(context.pipeline_graph, info.blend_pipeline, context.command_buffer,
                                       info.samples, 1, true);
            context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        blend_layout);
            vkCmdPushConstants(context.command_buffer, blend_layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
            detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                              detail::ForwardDynamicStateMode::blend);
            vkCmdSetCullMode(context.command_buffer, VK_CULL_MODE_NONE);

            auto const blend_offset = static_cast<VkDeviceSize>(info.counts.opaque + info.counts.mask) *
                                      sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(context.command_buffer, info.draws.indirect.buffer, blend_offset,
                                     info.counts.blend, sizeof(VkDrawIndexedIndirectCommand));
        }

        vkCmdEndQuery(context.command_buffer, info.pipeline_statistics_query_pool, 0);

        if (info.draw_light_icons && info.light_count != 0) {
            if (auto const light_icon_layout = detail::resolve_layout(context.pipeline_graph, info.light_icon_pipeline);
                light_icon_layout != VK_NULL_HANDLE) {
                detail::bind_graphics_node(context.pipeline_graph, info.light_icon_pipeline, context.command_buffer,
                                           info.samples, 1, true, false);
                context.resource_table.bind(context.command_buffer, context.frame_index,
                                            VK_PIPELINE_BIND_POINT_GRAPHICS, light_icon_layout);

                vkCmdSetDepthTestEnable(context.command_buffer, VK_TRUE);
                vkCmdSetDepthWriteEnable(context.command_buffer, VK_FALSE);
                vkCmdSetDepthCompareOp(context.command_buffer, VK_COMPARE_OP_GREATER_OR_EQUAL);
                vkCmdSetCullMode(context.command_buffer, VK_CULL_MODE_NONE);

                LightIconPushConstants const light_pc{
                        .lights_address = info.lights_address,
                        .ubo_address = info.ubo_address,
                        .light_count = info.light_count,
                        .icon_texture_index = info.light_icon_texture_index,
                        .sampler_index = info.linear_sampler_index,
                        .icon_world_size = info.light_icon_world_size,
                };

                vkCmdPushConstants(context.command_buffer, light_icon_layout, VK_SHADER_STAGE_ALL, 0, sizeof(light_pc),
                                   &light_pc);
                vkCmdDrawMeshTasksEXT(context.command_buffer, (info.light_count + 31U) / 32U, 1, 1);
            }
        }

        debug_overlay();

        vkCmdEndRendering(context.command_buffer);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);

        detail::transition_hdr_to_shader_read(context.command_buffer,
                                              info.resolved_hdr != nullptr ? *info.resolved_hdr : info.hdr);
        return info.output_hdr;
    }

    auto bloom(Context const &context, BloomPassInfo const &info)
            -> std::expected<std::optional<BloomTextureIndex>, RendererError> {
        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::BloomPass);
        auto const timestamp_stage =
                info.enabled ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;

        vkCmdWriteTimestamp2(context.command_buffer, timestamp_stage, context.timestamp_query_pool, stage * 2);

        if (!info.enabled) {
            vkCmdWriteTimestamp2(context.command_buffer, timestamp_stage, context.timestamp_query_pool, stage * 2 + 1);
            return std::optional<BloomTextureIndex>{};
        }

        if (info.target == nullptr || !info.target->valid()) {
            return std::unexpected(detail::make_error(RendererErrorType::image_error));
        }

        auto const downsample_layout = detail::resolve_layout(context.pipeline_graph, info.downsample_pipeline);
        auto const upsample_layout = detail::resolve_layout(context.pipeline_graph, info.upsample_pipeline);

        if (downsample_layout == VK_NULL_HANDLE || upsample_layout == VK_NULL_HANDLE) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        transition_image_layout(context.command_buffer, info.target->image(), VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, VK_ACCESS_2_SHADER_WRITE_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT, 0, 4);

        detail::bind_compute_node(context.pipeline_graph, info.downsample_pipeline, context.command_buffer);
        context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    downsample_layout);

        auto const mip0_extent = info.target->mip_extent(0);

        DownsamplePushConstants const downsample_pc{
                .src_texture_index = info.input_hdr.index,
                .linear_sampler_index = info.linear_sampler_index,
                .dst_mip0_storage_index = info.mip_texture_indices[0],
                .dst_mip1_storage_index = info.mip_texture_indices[1],
                .dst_mip2_storage_index = info.mip_texture_indices[2],
                .dst_mip3_storage_index = info.mip_texture_indices[3],
                .src_texel_size_x = 1.0F / static_cast<float>(info.input_extent.width),
                .src_texel_size_y = 1.0F / static_cast<float>(info.input_extent.height),
                .mip0_size_x = static_cast<std::int32_t>(mip0_extent.width),
                .mip0_size_y = static_cast<std::int32_t>(mip0_extent.height),
                .threshold = info.threshold,
                .knee = info.knee,
        };

        vkCmdPushConstants(context.command_buffer, downsample_layout, VK_SHADER_STAGE_ALL, 0, sizeof(downsample_pc),
                           &downsample_pc);
        vkCmdDispatch(context.command_buffer, (mip0_extent.width + 15U) / 16U, (mip0_extent.height + 15U) / 16U, 1);

        transition_image_layout(context.command_buffer, info.target->image(), VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                                0, 4);

        detail::bind_compute_node(context.pipeline_graph, info.upsample_pipeline, context.command_buffer);
        context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    upsample_layout);

        for (std::int32_t i = 2; i >= 0; --i) {
            auto const target_mip = static_cast<std::uint32_t>(i);
            auto const lower_mip = target_mip + 1U;
            auto const lower_extent = info.target->mip_extent(lower_mip);
            auto const target_extent = info.target->mip_extent(target_mip);

            UpsamplePushConstants const upsample_pc{
                    .lower_mip_texture_index = info.mip_texture_indices[lower_mip],
                    .target_mip_storage_index = info.mip_texture_indices[target_mip],
                    .linear_sampler_index = info.linear_sampler_index,
                    .lower_texel_size_x = 1.0F / static_cast<float>(lower_extent.width),
                    .lower_texel_size_y = 1.0F / static_cast<float>(lower_extent.height),
                    .target_size_x = static_cast<std::int32_t>(target_extent.width),
                    .target_size_y = static_cast<std::int32_t>(target_extent.height),
                    .filter_radius = info.filter_radius,
            };

            vkCmdPushConstants(context.command_buffer, upsample_layout, VK_SHADER_STAGE_ALL, 0, sizeof(upsample_pc),
                               &upsample_pc);
            vkCmdDispatch(context.command_buffer, (target_extent.width + 15U) / 16U, (target_extent.height + 15U) / 16U,
                          1);

            transition_image_layout(context.command_buffer, info.target->image(), VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                    VK_IMAGE_ASPECT_COLOR_BIT, target_mip, 1);
        }

        transition_image_layout(context.command_buffer, info.target->image(), VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                                VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);

        return std::optional<BloomTextureIndex>{BloomTextureIndex{.index = info.mip_texture_indices[0]}};
    }

    auto composite(Context const &context, CompositePassInfo const &info, Callback ui_overlay)
            -> std::expected<void, RendererError> {
        auto const layout = detail::resolve_layout(context.pipeline_graph, info.pipeline);
        if (layout == VK_NULL_HANDLE) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        detail::transition_swapchain_to_attachment(context.command_buffer, info.swapchain_image);

        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::Composition);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             context.timestamp_query_pool, stage * 2);

        VkRenderingAttachmentInfo const swapchain_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = info.swapchain_view,
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {},
        };

        VkRenderingInfo const rendering_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea = VkRect2D{.offset = {0, 0}, .extent = info.extent},
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 1,
                .pColorAttachments = &swapchain_attachment,
                .pDepthAttachment = nullptr,
                .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(context.command_buffer, &rendering_info);
        detail::bind_graphics_node(context.pipeline_graph, info.pipeline, context.command_buffer, VK_SAMPLE_COUNT_1_BIT,
                                   1, false);
        context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    layout);
        detail::set_composite_dynamic_state(context.command_buffer, info.extent);

        auto const bloom_texture_index = info.bloom.has_value() ? info.bloom->index : info.bloom_fallback_texture_index;
        auto const bloom_intensity = info.bloom.has_value() ? info.bloom_intensity : 0.0F;

        CompositePushConstants const pc{
                .hdr_texture_index = info.hdr.index,
                .bloom_texture_index = bloom_texture_index,
                .sampler_index = info.linear_sampler_index,
                .exposure = info.exposure,
                .bloom_intensity = bloom_intensity,
        };

        vkCmdPushConstants(context.command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
        vkCmdDraw(context.command_buffer, 3, 1, 0, 0);

        ui_overlay();

        vkCmdEndRendering(context.command_buffer);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);
        return {};
    }

    auto present_swapchain(VkCommandBuffer command_buffer, VkImage image) noexcept -> void {
        detail::transition_swapchain_to_present(command_buffer, image);
    }

} // namespace render_pass
