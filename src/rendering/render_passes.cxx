#include "rendering/render_passes.hxx"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

#include "assets/meshlet.hxx"
#include "gpu/vk_barrier.hxx"
#include "rendering/hiz_occlusion.hxx"
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

        // One scene draw: the pipeline pair and the attachment state to bind it with.
        struct SceneDraw {
            PipelineNodeHandle meshlet_pipeline{};
            PipelineNodeHandle instanced_pipeline{};
            VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
            std::uint32_t colour_attachment_count = 0;
            bool blending = false;
        };

        [[nodiscard]] auto scene_layouts_valid(PipelineGraphRepository const &graph, SceneDraw const &draw) noexcept
                -> bool {
            return resolve_layout(graph, draw.meshlet_pipeline) != VK_NULL_HANDLE &&
                   resolve_layout(graph, draw.instanced_pipeline) != VK_NULL_HANDLE;
        }

        // Draws `command_count` commands from `first_command` down both paths: vkCmdDrawMeshTasksIndirectEXT with the
        // task/mesh pipeline, then vkCmdDrawIndexedIndirect with the instanced one. Each command has one live half.
        //
        // SV_DrawIndex restarts at 0 per indirect call, so PC::task_commands points at this call's first command.
        template<typename PushConstants>
        auto draw_scene_commands(Context const &context, SceneDraw const &draw, DrawBuffers const &buffers,
                                 std::uint32_t first_command, std::uint32_t command_count, PushConstants pc) noexcept
                -> void {
            if (command_count == 0) {
                return;
            }

            auto const offset = static_cast<VkDeviceSize>(first_command) * sizeof(GpuDrawCommand);
            pc.task_commands_address = buffers.indirect.device_address + offset;

            auto const bind = [&](PipelineNodeHandle pipeline, bool has_vertex_input_stage) {
                auto const layout = resolve_layout(context.pipeline_graph, pipeline);

                bind_graphics_node(context.pipeline_graph, pipeline, context.command_buffer, draw.samples,
                                   draw.colour_attachment_count, draw.blending, has_vertex_input_stage);
                context.resource_table.bind(context.command_buffer, context.frame_index,
                                            VK_PIPELINE_BIND_POINT_GRAPHICS, layout);
                vkCmdPushConstants(context.command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
            };

            bind(draw.meshlet_pipeline, false);
            vkCmdDrawMeshTasksIndirectEXT(context.command_buffer, buffers.indirect.buffer, offset, command_count,
                                          sizeof(GpuDrawCommand));

            bind(draw.instanced_pipeline, true);
            vkCmdBindIndexBuffer(context.command_buffer, buffers.index_buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexedIndirect(context.command_buffer, buffers.indirect.buffer, offset + indexed_command_offset,
                                     command_count, sizeof(GpuDrawCommand));
        }

        // Frustum-only without back-face culling (mask, blend, shadows); opaque main-view draws also cone-cull.
        // Mirrors cull_*_bit in scene_types.slang.
        constexpr std::uint32_t cull_frustum = 1U;
        constexpr std::uint32_t cull_frustum_and_backface = 1U | 2U;

    } // namespace detail

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

        detail::SceneDraw const opaque_draw{
                .meshlet_pipeline = info.opaque_pipeline,
                .instanced_pipeline = info.opaque_instanced_pipeline,
        };
        detail::SceneDraw const mask_draw{
                .meshlet_pipeline = info.mask_pipeline,
                .instanced_pipeline = info.mask_instanced_pipeline,
        };

        if ((has_dirty_opaque && !detail::scene_layouts_valid(context.pipeline_graph, opaque_draw)) ||
            (has_dirty_mask && !detail::scene_layouts_valid(context.pipeline_graph, mask_draw))) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        // The frame graph has begun rendering into the atlas, loading it when `preserve_contents`.

        // LOAD keeps cached tiles; clear only the ones being redrawn. Reverse-Z clears to zero.
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

        if (has_dirty_opaque) {
            ShadowPushConstants pc{
                    .draws_address = info.draws.draws.device_address,
                    .transforms_address = info.draws.transforms.device_address,
                    .materials_address = info.materials_address,
                    .ubo_address = info.ubo_address,
                    .lights_address = info.lights_address,
                    .light_count = 0,
                    ._padding = 0,
                    .cull_planes_address = info.cascade_cull_planes_address,
                    .cull_flags = info.meshlet_culling ? detail::cull_frustum : 0U,
                    .cascade_index = 0,
                    .padding = 0,
            };

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
                pc.cascade_index = cascade;
                detail::draw_scene_commands(context, opaque_draw, info.draws, 0, cascade_draw_count, pc);
            }
        }

        if (has_dirty_mask) {
            ShadowPushConstants pc{
                    .draws_address = info.draws.draws.device_address,
                    .transforms_address = info.draws.transforms.device_address,
                    .materials_address = info.materials_address,
                    .ubo_address = info.ubo_address,
                    .lights_address = info.lights_address,
                    .light_count = 0,
                    ._padding = 0,
                    .cull_planes_address = info.cascade_cull_planes_address,
                    .cull_flags = info.meshlet_culling ? detail::cull_frustum : 0U,
                    .cascade_index = 0,
                    .padding = 0,
            };

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
                pc.cascade_index = cascade;
                detail::draw_scene_commands(context, mask_draw, info.draws, info.counts.opaque, cascade_draw_count, pc);
            }
        }

        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);
        return {};
    }

    auto depth_prepass(Context const &context, DepthPrepassInfo const &info) -> std::expected<void, RendererError> {
        detail::SceneDraw const opaque_draw{
                .meshlet_pipeline = info.opaque_pipeline,
                .instanced_pipeline = info.opaque_instanced_pipeline,
                .samples = info.samples,
        };
        detail::SceneDraw const mask_draw{
                .meshlet_pipeline = info.mask_pipeline,
                .instanced_pipeline = info.mask_instanced_pipeline,
                .samples = info.samples,
        };

        if ((info.counts.opaque != 0 && !detail::scene_layouts_valid(context.pipeline_graph, opaque_draw)) ||
            (info.counts.mask != 0 && !detail::scene_layouts_valid(context.pipeline_graph, mask_draw))) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        bool const late = info.phase == DepthPrepassPhase::late;
        auto const stage = static_cast<std::uint32_t>(late ? RenderStage::DepthPrepassLate : RenderStage::DepthPrepass);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, context.timestamp_query_pool,
                             stage * 2);

        // The frame graph has begun rendering.

        ForwardPushConstants const pc{
                .draws_address = info.draws.draws.device_address,
                .transforms_address = info.draws.transforms.device_address,
                .materials_address = info.materials_address,
                .ubo_address = info.ubo_address,
                .lights_address = info.lights_address,
                .light_count = 0,
                ._padding = 0,
                .cull_planes_address = info.cull_planes_address,
                .occlusion_address = info.occlusion_view_address,
        };

        auto opaque_pc = pc;
        opaque_pc.cull_flags = info.meshlet_culling ? detail::cull_frustum_and_backface | info.extra_cull_flags : 0U;

        auto mask_pc = pc;
        mask_pc.cull_flags = info.meshlet_culling ? detail::cull_frustum | info.extra_cull_flags : 0U;

        if (info.counts.opaque != 0) {
            detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                              detail::ForwardDynamicStateMode::prepass);
            detail::draw_scene_commands(context, opaque_draw, info.draws, 0, info.counts.opaque, opaque_pc);
        }

        if (info.counts.mask != 0) {
            detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                              detail::ForwardDynamicStateMode::prepass);
            vkCmdSetCullMode(context.command_buffer, VK_CULL_MODE_NONE);
            detail::draw_scene_commands(context, mask_draw, info.draws, info.counts.opaque, info.counts.mask, mask_pc);
        }

        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);
        return {};
    }

    auto build_hiz(Context const &context, HizBuildInfo const &info) -> std::expected<void, RendererError> {
        auto const mip_count = static_cast<std::uint32_t>(info.mip_texture_indices.size());
        HizExtent const depth_extent{.width = info.depth_extent.width, .height = info.depth_extent.height};

        if (mip_count == 0 || mip_count != hiz_mip_count(depth_extent) || info.hiz.mip_levels() < mip_count) {
            return std::unexpected(detail::make_error(RendererErrorType::image_error));
        }

        auto const layout = detail::resolve_layout(context.pipeline_graph, info.pipeline);

        if (layout == VK_NULL_HANDLE) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        auto const command_buffer = context.command_buffer;
        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::HiZBuild);

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, context.timestamp_query_pool,
                             stage * 2);

        // The occlusion tests (main_cs, late_cs, and task shaders for meshlet occlusion) and the debug view.
        // (On a compute-only family only the compute reader can be named; the graph covers the others.)
        VkPipelineStageFlags2 const hiz_reader_stages = context.compute_only
                                                                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                                                : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                                          VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                                                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

        // The frame graph has put the source depth in SHADER_READ_ONLY_OPTIMAL and every level of the pyramid in
        // GENERAL (it is rebuilt from scratch), and takes the depth back for the late prepass afterwards.
        detail::bind_compute_node(context.pipeline_graph, info.pipeline, command_buffer);
        context.resource_table.bind(command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, layout);

        for (std::uint32_t level = 0; level < mip_count; ++level) {
            bool const first_level = level == 0;
            auto const source_extent = first_level ? depth_extent : hiz_level_extent(depth_extent, level - 1);
            auto const level_extent = hiz_level_extent(depth_extent, level);

            HizBuildPushConstants const build_pc{
                    .src_texture_index = first_level ? info.source_texture_index : info.mip_texture_indices[level - 1],
                    .dst_storage_index = info.mip_texture_indices[level],
                    .src_width = source_extent.width,
                    .src_height = source_extent.height,
                    .dst_width = level_extent.width,
                    .dst_height = level_extent.height,
            };

            vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(build_pc), &build_pc);
            vkCmdDispatch(command_buffer, (level_extent.width + 7U) / 8U, (level_extent.height + 7U) / 8U, 1);

            // Read by the next level's dispatch and, once the chain is done, by the occlusion tests.
            transition_image_layout(command_buffer, info.hiz.image(), VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    hiz_reader_stages, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, level, 1);
        }

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, context.timestamp_query_pool,
                             stage * 2 + 1);

        return {};
    }

    auto gtao(Context const &context, AmbientOcclusionInfo const &info) -> std::expected<void, RendererError> {
        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::AmbientOcclusion);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             context.timestamp_query_pool, stage * 2);

        auto const gtao_layout = detail::resolve_layout(context.pipeline_graph, info.gtao_pipeline);
        if (gtao_layout == VK_NULL_HANDLE) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

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
        return {};
    }

    auto gtao_denoise(Context const &context, AmbientOcclusionInfo const &info) -> std::expected<void, RendererError> {
        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::AmbientOcclusion);

        auto const denoise_layout = detail::resolve_layout(context.pipeline_graph, info.denoise_pipeline);
        if (denoise_layout == VK_NULL_HANDLE) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

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

        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);
        return {};
    }

    auto forward_geometry(Context const &context, ForwardGeometryInfo const &info, Callback scene_overlays)
            -> std::expected<HdrTextureIndex, RendererError> {
        detail::SceneDraw const opaque_draw{
                .meshlet_pipeline = info.opaque_pipeline,
                .instanced_pipeline = info.opaque_instanced_pipeline,
                .samples = info.samples,
                .colour_attachment_count = 1,
                .blending = false,
        };
        detail::SceneDraw const blend_draw{
                .meshlet_pipeline = info.blend_pipeline,
                .instanced_pipeline = info.blend_instanced_pipeline,
                .samples = info.samples,
                .colour_attachment_count = 1,
                .blending = true,
        };

        if (!detail::scene_layouts_valid(context.pipeline_graph, opaque_draw) ||
            (info.counts.blend != 0 && !detail::scene_layouts_valid(context.pipeline_graph, blend_draw))) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::ForwardPass);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             context.timestamp_query_pool, stage * 2);

        // The frame graph has begun rendering.
        vkCmdBeginQuery(context.command_buffer, info.pipeline_statistics_query_pool, 0, 0);

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
                .cull_planes_address = info.cull_planes_address,
                .cluster_lights_address = info.cluster_lights_address,
                .occlusion_address = info.occlusion_view_address,
        };

        // Must cull exactly like the prepass's opaque draw, since this pass depth-tests EQUAL. With meshlet occlusion
        // the extra flags make it replay the meshlets the prepass phases recorded instead of testing them again.
        auto opaque_pc = pc;
        opaque_pc.cull_flags = info.meshlet_culling ? detail::cull_frustum_and_backface | info.extra_cull_flags : 0U;

        auto mask_pc = pc;
        mask_pc.cull_flags = info.meshlet_culling ? detail::cull_frustum | info.extra_cull_flags : 0U;

        // Blend draws never enter the prepass, so they take no occlusion bits.
        auto unculled_backface_pc = pc;
        unculled_backface_pc.cull_flags = info.meshlet_culling ? detail::cull_frustum : 0U;

        if (info.counts.opaque != 0) {
            detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                              detail::ForwardDynamicStateMode::main);
            detail::draw_scene_commands(context, opaque_draw, info.draws, 0, info.counts.opaque, opaque_pc);
        }

        if (info.counts.mask != 0) {
            detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                              detail::ForwardDynamicStateMode::main);
            vkCmdSetCullMode(context.command_buffer, VK_CULL_MODE_NONE);
            detail::draw_scene_commands(context, opaque_draw, info.draws, info.counts.opaque, info.counts.mask,
                                        mask_pc);
        }

        // The background goes in before blending so blended surfaces composite over it, and inside this rendering
        // scope because under MSAA the colour attachment is not stored, only resolved when the scope ends. Reversed Z
        // stores the sky's ndc.z = 1 as depth 0, so with GREATER_OR_EQUAL and no depth write it only touches pixels no
        // geometry reached.
        if (info.draw_skybox) {
            auto const sky_layout = detail::resolve_layout(context.pipeline_graph, info.skybox_pipeline);

            if (sky_layout != VK_NULL_HANDLE) {
                detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                                  detail::ForwardDynamicStateMode::blend);
                vkCmdSetCullMode(context.command_buffer, VK_CULL_MODE_NONE);

                detail::bind_graphics_node(context.pipeline_graph, info.skybox_pipeline, context.command_buffer,
                                           info.samples, 1, false, true);
                context.resource_table.bind(context.command_buffer, context.frame_index,
                                            VK_PIPELINE_BIND_POINT_GRAPHICS, sky_layout);

                SkyboxPushConstants const sky_pc{.ubo_address = info.ubo_address};

                vkCmdPushConstants(context.command_buffer, sky_layout, VK_SHADER_STAGE_ALL, 0, sizeof(sky_pc), &sky_pc);
                vkCmdDraw(context.command_buffer, 3, 1, 0, 0);
            }
        }

        if (info.counts.blend != 0) {
            detail::set_forward_dynamic_state(context.command_buffer, info.extent,
                                              detail::ForwardDynamicStateMode::blend);
            vkCmdSetCullMode(context.command_buffer, VK_CULL_MODE_NONE);
            detail::draw_scene_commands(context, blend_draw, info.draws, info.counts.opaque + info.counts.mask,
                                        info.counts.blend, unculled_backface_pc);
        }

        vkCmdEndQuery(context.command_buffer, info.pipeline_statistics_query_pool, 0);

        scene_overlays();

        // Inside the rendering scope, so it no longer covers the end-of-scope resolve.
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);

        return info.output_hdr;
    }

    auto set_overlay_baseline_state(VkCommandBuffer command_buffer, OverlayStage stage,
                                    OverlayScope const &scope) noexcept -> void {
        auto const width = static_cast<float>(scope.extent.width);
        auto const height = static_cast<float>(scope.extent.height);

        VkViewport const viewport = stage == OverlayStage::scene ? VkViewport{.x = 0.0F,
                                                                              .y = height,
                                                                              .width = width,
                                                                              .height = -height,
                                                                              .minDepth = 1.0F,
                                                                              .maxDepth = 0.0F}
                                                                 : VkViewport{.x = 0.0F,
                                                                              .y = 0.0F,
                                                                              .width = width,
                                                                              .height = height,
                                                                              .minDepth = 0.0F,
                                                                              .maxDepth = 1.0F};

        VkRect2D const scissor{.offset = {0, 0}, .extent = scope.extent};

        vkCmdSetViewportWithCount(command_buffer, 1, &viewport);
        vkCmdSetScissorWithCount(command_buffer, 1, &scissor);

        detail::set_shader_object_vertex_input(command_buffer, {}, {});
        detail::set_shader_object_raster_state(command_buffer, VK_POLYGON_MODE_FILL, scope.samples, false);
        detail::set_shader_object_color_blend_state(command_buffer, 1, false);

        vkCmdSetPrimitiveTopology(command_buffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        vkCmdSetPrimitiveRestartEnable(command_buffer, VK_FALSE);
        vkCmdSetRasterizerDiscardEnable(command_buffer, VK_FALSE);
        vkCmdSetCullMode(command_buffer, VK_CULL_MODE_NONE);
        vkCmdSetFrontFace(command_buffer, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        vkCmdSetDepthBiasEnable(command_buffer, VK_FALSE);
        vkCmdSetStencilTestEnable(command_buffer, VK_FALSE);

        vkCmdSetDepthTestEnable(command_buffer, scope.has_depth() ? VK_TRUE : VK_FALSE);
        vkCmdSetDepthWriteEnable(command_buffer, VK_FALSE);
        vkCmdSetDepthCompareOp(command_buffer,
                               scope.has_depth() ? VK_COMPARE_OP_GREATER_OR_EQUAL : VK_COMPARE_OP_ALWAYS);
    }

    auto light_icons(Context const &context, LightIconsInfo const &info, OverlayScope const &scope) noexcept -> void {
        if (info.light_count == 0) {
            return;
        }

        auto const layout = detail::resolve_layout(context.pipeline_graph, info.pipeline);
        if (layout == VK_NULL_HANDLE) {
            return;
        }

        detail::bind_graphics_node(context.pipeline_graph, info.pipeline, context.command_buffer, scope.samples, 1,
                                   true, false);
        context.resource_table.bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    layout);

        LightIconPushConstants const pc{
                .lights_address = info.lights_address,
                .ubo_address = info.ubo_address,
                .light_count = info.light_count,
                .icon_texture_index = info.icon_texture_index,
                .sampler_index = info.sampler_index,
                .icon_world_size = info.icon_world_size,
        };

        vkCmdPushConstants(context.command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
        vkCmdDrawMeshTasksEXT(context.command_buffer, (info.light_count + 31U) / 32U, 1, 1);
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

        if (info.target == nullptr || !info.target->valid() || info.target->mip_levels() < bloom_mip_count) {
            return std::unexpected(detail::make_error(RendererErrorType::image_error));
        }

        auto const downsample_layout = detail::resolve_layout(context.pipeline_graph, info.downsample_pipeline);
        auto const upsample_layout = detail::resolve_layout(context.pipeline_graph, info.upsample_pipeline);

        if (downsample_layout == VK_NULL_HANDLE || upsample_layout == VK_NULL_HANDLE) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        auto const command_buffer = context.command_buffer;
        auto const bloom_image = info.target->image();

        // A level is GENERAL while written and SHADER_READ_ONLY_OPTIMAL while sampled. Each barrier moves one level,
        // so no dispatch samples and stores the same level.
        auto const to_storage = [&](std::uint32_t mip) {
            transition_image_layout(command_buffer, bloom_image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_NONE,
                                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                    VK_IMAGE_ASPECT_COLOR_BIT, mip, 1);
        };

        auto const to_sampled = [&](std::uint32_t mip) {
            transition_image_layout(command_buffer, bloom_image, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    context.compute_only ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                                         : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                    VK_IMAGE_ASPECT_COLOR_BIT, mip, 1);
        };

        // The frame graph has put every level in GENERAL, discarding last frame's contents: the chain is rebuilt each
        // frame. It takes the finished chain, every level SHADER_READ_ONLY_OPTIMAL, from here.

        // Downsample: HDR -> mip 0 -> mip 1 -> ..., each sampling the level above.
        detail::bind_compute_node(context.pipeline_graph, info.downsample_pipeline, command_buffer);
        context.resource_table.bind(command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    downsample_layout);

        for (std::uint32_t mip = 0; mip < bloom_mip_count; ++mip) {
            bool const first_level = mip == 0;
            auto const src_extent = first_level ? info.input_extent : info.target->mip_extent(mip - 1);
            auto const dst_extent = info.target->mip_extent(mip);

            DownsamplePushConstants const downsample_pc{
                    .src_texture_index = first_level ? info.input_hdr.index : info.mip_texture_indices[mip - 1],
                    .dst_storage_index = info.mip_texture_indices[mip],
                    .linear_sampler_index = info.linear_sampler_index,
                    .is_first_level = first_level ? 1U : 0U,
                    .src_texel_size_x = 1.0F / static_cast<float>(src_extent.width),
                    .src_texel_size_y = 1.0F / static_cast<float>(src_extent.height),
                    .dst_size_x = static_cast<std::int32_t>(dst_extent.width),
                    .dst_size_y = static_cast<std::int32_t>(dst_extent.height),
                    .threshold = info.threshold,
                    .knee = info.knee,
            };

            vkCmdPushConstants(command_buffer, downsample_layout, VK_SHADER_STAGE_ALL, 0, sizeof(downsample_pc),
                               &downsample_pc);
            vkCmdDispatch(command_buffer, (dst_extent.width + 7U) / 8U, (dst_extent.height + 7U) / 8U, 1);

            to_sampled(mip);
        }

        // Upsample: add the tent-filtered level below onto each level. mip 0 ends up with the full bloom.
        detail::bind_compute_node(context.pipeline_graph, info.upsample_pipeline, command_buffer);
        context.resource_table.bind(command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    upsample_layout);

        for (auto target_mip = bloom_mip_count - 1U; target_mip-- > 0U;) {
            auto const lower_mip = target_mip + 1U;
            auto const lower_extent = info.target->mip_extent(lower_mip);
            auto const target_extent = info.target->mip_extent(target_mip);

            to_storage(target_mip);

            UpsamplePushConstants const upsample_pc{
                    .lower_texture_index = info.mip_texture_indices[lower_mip],
                    .target_storage_index = info.mip_texture_indices[target_mip],
                    .linear_sampler_index = info.linear_sampler_index,
                    .lower_texel_size_x = 1.0F / static_cast<float>(lower_extent.width),
                    .lower_texel_size_y = 1.0F / static_cast<float>(lower_extent.height),
                    .target_size_x = static_cast<std::int32_t>(target_extent.width),
                    .target_size_y = static_cast<std::int32_t>(target_extent.height),
                    .filter_radius = info.filter_radius,
            };

            vkCmdPushConstants(command_buffer, upsample_layout, VK_SHADER_STAGE_ALL, 0, sizeof(upsample_pc),
                               &upsample_pc);
            vkCmdDispatch(command_buffer, (target_extent.width + 7U) / 8U, (target_extent.height + 7U) / 8U, 1);

            to_sampled(target_mip);
        }

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, context.timestamp_query_pool,
                             stage * 2 + 1);

        return std::optional<BloomTextureIndex>{BloomTextureIndex{.index = info.mip_texture_indices[0]}};
    }

    auto composite(Context const &context, CompositePassInfo const &info, Callback ui_overlay)
            -> std::expected<void, RendererError> {
        auto const layout = detail::resolve_layout(context.pipeline_graph, info.pipeline);
        if (layout == VK_NULL_HANDLE) {
            return std::unexpected(detail::make_error(RendererErrorType::invalid_pipeline));
        }

        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::Composition);
        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             context.timestamp_query_pool, stage * 2);

        // The frame graph has begun rendering into the swapchain or the viewport target.
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

        vkCmdWriteTimestamp2(context.command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             context.timestamp_query_pool, stage * 2 + 1);
        return {};
    }

} // namespace render_pass
