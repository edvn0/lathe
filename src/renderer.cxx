#include "renderer.hxx"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <expected>
#include <fstream>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "buffer.hxx"
#include "context.hxx"
#include "device_error.hxx"
#include "gpu_resource_table.hxx"
#include "logger.hxx"
#include "material_storage.hxx"
#include "sampler_storage.hxx"
#include "slang_compiler.hxx"

namespace {
    struct ForwardPushConstants {
        VkDeviceAddress draw_buffer_address;
        VkDeviceAddress transform_buffer_address;
        VkDeviceAddress material_buffer_address;
        VkDeviceAddress ubo_buffer_address;
    };

    struct CompositePushConstants {
        std::uint32_t hdr_texture_index;
        std::uint32_t sampler_index;
        float exposure;
        std::uint32_t padding;
    };

    static_assert(sizeof(CompositePushConstants) == 16);
} // namespace

namespace {
    auto transition_forward_target_to_attachments(VkCommandBuffer command_buffer, Image const &hdr,
                                                  Image const &depth) noexcept -> void {
        std::array<VkImageMemoryBarrier2, 2> barriers{
                VkImageMemoryBarrier2{
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
                },
                VkImageMemoryBarrier2{
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
                        .image = depth.image(),
                        .subresourceRange =
                                VkImageSubresourceRange{
                                        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                        .baseMipLevel = 0,
                                        .levelCount = depth.mip_levels(),
                                        .baseArrayLayer = 0,
                                        .layerCount = depth.array_layers(),
                                },
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
                .imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size()),
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

    auto set_forward_dynamic_state(VkCommandBuffer command_buffer, VkExtent2D extent, bool is_prepass) noexcept
            -> void {
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

        vkCmdSetViewport(command_buffer, 0, 1, &viewport);

        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        vkCmdSetPrimitiveTopology(command_buffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        vkCmdSetPrimitiveRestartEnable(command_buffer, VK_FALSE);

        vkCmdSetRasterizerDiscardEnable(command_buffer, VK_FALSE);

        vkCmdSetCullMode(command_buffer, VK_CULL_MODE_BACK_BIT);

        vkCmdSetFrontFace(command_buffer, VK_FRONT_FACE_CLOCKWISE);

        vkCmdSetDepthTestEnable(command_buffer, VK_TRUE);

        vkCmdSetDepthWriteEnable(command_buffer, VK_TRUE);

        vkCmdSetDepthCompareOp(command_buffer, is_prepass ? VK_COMPARE_OP_GREATER_OR_EQUAL : VK_COMPARE_OP_EQUAL);

        vkCmdSetDepthBiasEnable(command_buffer, VK_FALSE);

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

        vkCmdSetViewport(command_buffer, 0, 1, &viewport);

        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

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
} // namespace

namespace {
    auto make_error(RendererErrorType type) -> RendererError {
        return RendererError{
                .type = type,
        };
    }

    auto make_resource_table_error(const GpuResourceTableError &err) -> RendererError {
        return RendererError{
                .type = RendererErrorType::gpu_resource_table_error,
                .gpu_resource_table_error = err,
        };
    }

    auto make_pipeline_graph_error(PipelineGraphError const &error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::pipeline_graph_error,
                .pipeline_graph_error = error,
        };
    }

    auto make_device_error(DeviceError const &error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::device_error,
                .device_error = error,
        };
    }

    auto make_geometry_error(GeometryArenaError const &error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::geometry_error,
                .geometry_error = error,
        };
    }

    auto make_compiler_error(renderer::ShaderCompileError const &error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::compiler_error,
                .compiler_error = error,
        };
    }

    auto make_material_error(MaterialStorageError const &error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::material_error,
                .material_error = error,
        };
    }

    auto make_model_load_error(ModelLoadError const &error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::model_load_error,
                .model_load_error = error,
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

auto Renderer::initialize(RendererCreateInfo const &create_info) -> std::expected<void, RendererError> {
    if (initialized_ || create_info.extent.width == 0 || create_info.extent.height == 0 ||
        create_info.frames_in_flight == 0 || create_info.material_capacity < 2 || create_info.mesh_capacity < 2 ||
        create_info.model_capacity < 2 || create_info.maximum_draw_count == 0 ||
        create_info.maximum_submission_count == 0) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

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
        destroy();

        return std::unexpected(make_geometry_error(geometry_arena.error()));
    }

    auto material_storage = MaterialStorage::create(context_, MaterialStorageCreateInfo{
                                                                      .capacity = create_info.material_capacity,
                                                                      .debug_name = "renderer.materials",
                                                              });

    if (!material_storage) {
        destroy();

        return std::unexpected(make_material_error(material_storage.error()));
    }

    auto image_storage = ImageStorage::create(context_, ImageStorageCreateInfo{
                                                                .capacity = create_info.image_capacity,
                                                                .debug_name = "renderer.images",
                                                        });

    if (!image_storage) {
        destroy();

        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    auto sampler_storage = SamplerStorage::create(context_, create_info.sampler_capacity);

    if (!sampler_storage) {
        destroy();

        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    auto gpu_resource_table =
            GpuResourceTable::create(context_, GpuResourceTableCreateInfo{
                                                       .frames_in_flight = create_info.frames_in_flight,
                                                       .image_capacity = create_info.image_capacity,
                                                       .sampler_capacity = create_info.sampler_capacity,
                                                       .debug_name = "renderer.resources",
                                               });

    if (!gpu_resource_table) {
        destroy();

        return std::unexpected(make_resource_table_error(gpu_resource_table.error()));
    }

    gpu_resource_table_ = std::move(*gpu_resource_table);

    auto pipeline_graph = PipelineGraphRepository::create(
            context_, PipelineGraphCreateInfo{
                              .pipeline_capacity = create_info.pipeline_capacity,
                              .frames_in_flight = create_info.frames_in_flight,
                              .global_descriptor_set_layout = gpu_resource_table_.layout(),
                              .debug_name = "renderer.pipelines",
                      });

    if (!pipeline_graph) {
        destroy();

        return std::unexpected(make_pipeline_graph_error(pipeline_graph.error()));
    }


    pipeline_graph_ = std::move(*pipeline_graph);
    image_storage_ = std::move(*image_storage);
    sampler_storage_ = std::move(*sampler_storage);
    geometry_arena_ = std::move(*geometry_arena);
    material_storage_ = std::move(*material_storage);

    {
        VkPushConstantRange const forward_push_constant_range{
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(ForwardPushConstants),
        };

        auto registered_forward = pipeline_graph_.register_pipeline(
                compiler, PipelineRegisterInfo{
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
                                  .push_constant_ranges = {forward_push_constant_range},
                                  .colour_formats = {create_info.hdr_format},
                                  .depth_format = create_info.depth_format,
                                  .stencil_format = VK_FORMAT_UNDEFINED,
                                  .samples = create_info.samples,
                                  .debug_name = "renderer.forward_pipeline",
                          });

        if (!registered_forward) {
            destroy();

            return std::unexpected(make_pipeline_graph_error(registered_forward.error()));
        }

        forward_pipeline_ = *registered_forward;
    }
    {
        VkPushConstantRange const depth_prepass_pc{
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                .offset = 0,
                .size = sizeof(ForwardPushConstants),
        };

        auto registered_predepth = pipeline_graph_.register_pipeline(
                compiler, PipelineRegisterInfo{
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
                                  .push_constant_ranges = {depth_prepass_pc},
                                  .colour_formats = {},
                                  .depth_format = create_info.depth_format,
                                  .stencil_format = VK_FORMAT_UNDEFINED,
                                  .samples = create_info.samples,
                                  .debug_name = "renderer.depth_prepass_pipeline",
                          });

        if (!registered_predepth) {
            destroy();

            return std::unexpected(make_pipeline_graph_error(registered_predepth.error()));
        }

        depth_prepass_pipeline_ = *registered_predepth;
    }
    {
        VkPushConstantRange const composite_push_constant_range{
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(CompositePushConstants),
        };

        auto registered_composite = pipeline_graph_.register_pipeline(
                compiler, PipelineRegisterInfo{
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
                                  .push_constant_ranges = {composite_push_constant_range},
                                  .colour_formats = {create_info.swapchain_format},
                                  .depth_format = VK_FORMAT_UNDEFINED,
                                  .stencil_format = VK_FORMAT_UNDEFINED,
                                  .samples = VK_SAMPLE_COUNT_1_BIT,
                                  .debug_name = "renderer.composite_pipeline",
                          });

        if (!registered_composite) {
            destroy();

            return std::unexpected(make_pipeline_graph_error(registered_composite.error()));
        }

        composite_pipeline_ = *registered_composite;
    }

    auto const white = image_storage_.white();

    auto const flat_normal = image_storage_.flat_normal();

    auto const metallic_roughness = image_storage_.metallic_roughness();

    auto const occlusion = image_storage_.occlusion();

    auto const emissive = image_storage_.emissive();

    constexpr auto def_mat = MaterialHandle{0, 1};
    const auto mat = GpuMaterial{
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

    if (!draw_size_result || !transform_size_result || !indirect_size_result) {
        destroy();

        return std::unexpected(make_error(RendererErrorType::size_overflow));
    }

    auto const draw_size = *draw_size_result;

    auto const transform_size = *transform_size_result;

    auto const indirect_size = *indirect_size_result;

    auto const transform_offset = align_up(draw_size, 16);

    auto const indirect_offset = align_up(transform_offset + transform_size, 16);

    if (indirect_offset > std::numeric_limits<VkDeviceSize>::max() - indirect_size) {
        destroy();

        return std::unexpected(make_error(RendererErrorType::size_overflow));
    }

    auto const upload_size = indirect_offset + indirect_size;

    frames_.resize(create_info.frames_in_flight);

    for (std::uint32_t frame_index = 0; frame_index < static_cast<std::uint32_t>(frames_.size()); ++frame_index) {
        auto &frame = frames_[frame_index];

        auto upload = Buffer::create(context_, BufferCreateInfo{
                                                       .size = upload_size,
                                                       .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                       .memory = BufferMemory::upload,
                                                       .debug_name = "renderer.frame_upload",
                                               });

        if (!upload) {
            destroy();

            return std::unexpected(make_device_error(upload.error()));
        }

        auto draws = Buffer::create(context_, BufferCreateInfo{
                                                      .size = draw_size,
                                                      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                      .memory = BufferMemory::device,
                                                      .debug_name = "renderer.frame_draws",
                                              });

        if (!draws) {
            upload->destroy();
            destroy();

            return std::unexpected(make_device_error(draws.error()));
        }

        auto transforms = Buffer::create(context_, BufferCreateInfo{
                                                           .size = transform_size,
                                                           .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                           .memory = BufferMemory::device,
                                                           .debug_name = "renderer.frame_transforms",
                                                   });

        if (!transforms) {
            draws->destroy();
            upload->destroy();
            destroy();

            return std::unexpected(make_device_error(transforms.error()));
        }

        auto indirect = Buffer::create(context_, BufferCreateInfo{
                                                         .size = indirect_size,
                                                         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                                  VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                         .memory = BufferMemory::device,
                                                         .debug_name = "renderer.frame_indirect",
                                                 });

        if (!indirect) {
            transforms->destroy();
            draws->destroy();
            upload->destroy();
            destroy();

            return std::unexpected(make_device_error(indirect.error()));
        }

        auto const target_name = std::string{"renderer.forward_target_"} + std::to_string(frame_index);

        auto forward_target = ForwardTarget::create(image_storage_, ForwardTargetCreateInfo{
                                                                            .extent = create_info.extent,
                                                                            .hdr_format = create_info.hdr_format,
                                                                            .depth_format = create_info.depth_format,
                                                                            .samples = create_info.samples,
                                                                            .debug_name = target_name,
                                                                    });

        if (!forward_target) {
            indirect->destroy();
            transforms->destroy();
            draws->destroy();
            upload->destroy();

            destroy();

            return std::unexpected(RendererError{
                    .type = RendererErrorType::forward_target_error,
                    .forward_target_error = forward_target.error(),
            });
        }

        frame.forward_target = std::move(*forward_target);

        frame.upload_buffer = std::move(*upload);

        frame.draw_buffer = std::move(*draws);

        frame.transform_buffer = std::move(*transforms);

        frame.indirect_buffer = std::move(*indirect);

        frame.draw_upload_offset = 0;

        frame.transform_upload_offset = transform_offset;

        frame.indirect_upload_offset = indirect_offset;

        frame.draws.reserve(maximum_draw_count_);

        frame.transforms.reserve(maximum_submission_count_);

        frame.indirect_commands.reserve(maximum_draw_count_);
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context_.physical_device, &properties);

    if (!properties.limits.timestampComputeAndGraphics) {
        error("Physical device does not support timestamps on graphics/compute queues!");
    }

    this->timestamp_period_ = properties.limits.timestampPeriod;

    timestamp_queries_.resize(create_info.frames_in_flight);

    VkQueryPoolCreateInfo query_pool_info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = static_cast<std::uint32_t>(RenderStage::Count) * 2,
            .pipelineStatistics = 0,
    };

    for (std::uint32_t frame_index = 0; frame_index < create_info.frames_in_flight; ++frame_index) {
        VkQueryPool query_pool = VK_NULL_HANDLE;
        VkResult result = vkCreateQueryPool(context_.device, &query_pool_info, nullptr, &query_pool);

        if (result != VK_SUCCESS) {
            error("Failed to create timestamp query pool for frame index {}", frame_index);
            destroy();
            return std::unexpected(make_error(RendererErrorType::device_error));
        }

        timestamp_queries_[frame_index] = FrameTimestamps{.query_pool = query_pool, .has_results = false};
        vkResetQueryPool(context_.device, query_pool, 0, query_count);
    }

    constexpr auto size = sizeof(UBO);
    ubos_.resize(create_info.frames_in_flight);
    for (auto &ubo: ubos_) {
        auto maybe_ubo = Buffer::create(context_, BufferCreateInfo{
                                                          .size = size,
                                                          .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                          .memory = BufferMemory::device,
                                                          .debug_name = "renderer.ubo",
                                                  });
        if (!maybe_ubo || !maybe_ubo->zero()) {
            error("Failed to create ubo");
            destroy();
            return std::unexpected(make_error(RendererErrorType::device_error));
        }

        ubo = std::move(*maybe_ubo);
    }

    hdr_format_ = create_info.hdr_format;

    depth_format_ = create_info.depth_format;

    samples_ = create_info.samples;

    extent_ = create_info.extent;
    initialized_ = true;

    return {};
}

auto Renderer::destroy() noexcept -> void {
    pipeline_graph_.destroy();

    gpu_resource_table_.destroy();

    sampler_storage_.destroy();

    for (auto &query: timestamp_queries_) {
        if (query.query_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(context_.device, query.query_pool, nullptr);
            query.query_pool = VK_NULL_HANDLE;
        }
    }

    for (auto &ubo: ubos_) {
        ubo.destroy();
    }

    for (auto &frame: frames_) {
        frame.forward_target.destroy(image_storage_);

        frame.indirect_buffer.destroy();
        frame.transform_buffer.destroy();
        frame.draw_buffer.destroy();
        frame.upload_buffer.destroy();

        frame.draws.clear();
        frame.transforms.clear();
        frame.indirect_commands.clear();

        frame.indirect_command_count = 0;
    }

    frames_.clear();

    material_storage_.destroy();

    /*
     * Forward targets have already returned their image
     * handles, so the remaining images can now be destroyed.
     */
    image_storage_.destroy();

    geometry_arena_.buffer.destroy();

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

    std::expected<Model, ModelLoadError> imported_model{
            std::unexpected(ModelLoadError{.type = ModelLoadErrorType::invalid_argument})};

    context_.one_time_submit([&](VkCommandBuffer command_buffer) {
        imported_model =
                record_model_gpu_upload(*cpu_data, command_buffer, geometry_arena_, image_storage_, material_storage_);
    });

    if (!imported_model) {
        return std::unexpected(make_model_load_error(imported_model.error()));
    }

    image_storage_.release_completed_uploads();

    auto model_result = create_model(*imported_model, default_material_handle_);

    if (!model_result) {
        return std::unexpected{model_result.error()};
    }

    model_cache_[file_hash] = *model_result;

    return model_result;
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

    destination.next_free = 0;
    destination.occupied = true;

    return ModelHandle{
            .index = model_index,
            .generation = destination.generation,
    };
}

auto Renderer::submit_model(ModelHandle model, glm::mat4 const &transform) -> std::expected<void, RendererError> {
    if (model_slot(model) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (model_submissions_.size() >= maximum_submission_count_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    model_submissions_.push_back(ModelSubmission{
            .model = model,
            .transform = transform,
    });

    return {};
}

auto Renderer::submit_model(ModelHandle model, glm::mat4 &&transform) -> std::expected<void, RendererError> {
    if (model_slot(model) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (model_submissions_.size() >= maximum_submission_count_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    model_submissions_.push_back(ModelSubmission{
            .model = model,
            .transform = std::move(transform),
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

auto Renderer::submit_mesh(MeshHandle mesh, glm::mat4 const &transform) -> std::expected<void, RendererError> {
    if (mesh_slot(mesh) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_mesh));
    }

    if (submissions_.size() >= maximum_submission_count_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    submissions_.push_back(Submission{
            .mesh = mesh,
            .transform = transform,
    });

    return {};
}

auto Renderer::prepare_frame(VkCommandBuffer command_buffer, const CameraMatrices &matrices, std::uint32_t frame_index)
        -> std::expected<void, RendererError> {
    if (!initialized_ || command_buffer == VK_NULL_HANDLE || frame_index >= frames_.size()) {
        clear_submissions();

        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

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
    frame.indirect_command_count = 0;

    struct batch_key {
        std::uint32_t mesh_index;
        std::uint32_t submesh_index;

        auto operator==(batch_key const &) const noexcept -> bool = default;
    };

    struct batch_key_hash {
        auto operator()(batch_key const &key) const noexcept -> std::size_t {
            return std::hash<std::uint64_t>{}((static_cast<std::uint64_t>(key.mesh_index) << 32) | key.submesh_index);
        }
    };

    struct batch_entry {
        MeshHandle mesh{};
        std::uint32_t submesh_index = 0;
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
                auto const key = batch_key{
                        .mesh_index = model_draw.mesh.index,
                        .submesh_index = submesh_index,
                };

                auto [entry, inserted] = batches.try_emplace(key);

                if (inserted) {
                    entry->second.mesh = model_draw.mesh;
                    entry->second.submesh_index = submesh_index;
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
            auto const key = batch_key{
                    .mesh_index = submission.mesh.index,
                    .submesh_index = submesh_index,
            };

            auto [entry, inserted] = batches.try_emplace(key);

            if (inserted) {
                entry->second.mesh = submission.mesh;
                entry->second.submesh_index = submesh_index;
            }

            entry->second.transforms.push_back(submission.transform);
        }
    }

    for (auto const &[key, batch]: batches) {
        auto const *mesh = mesh_slot(batch.mesh);

        if (mesh == nullptr) {
            clear_submissions();

            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }

        auto const &submesh = mesh->submeshes[batch.submesh_index];

        auto const instance_count = static_cast<std::uint32_t>(batch.transforms.size());

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
                    .material_index = material_storage_.gpu_index(submesh.material),
                    .transform_index = base_transform_index + instance,
            });
        }

        frame.indirect_commands.push_back(VkDrawIndexedIndirectCommand{
                .indexCount = submesh.geometry.indices.index_count,
                .instanceCount = instance_count,
                .firstIndex = static_cast<std::uint32_t>(first_index_u64),
                .vertexOffset = 0,
                .firstInstance = first_instance,
        });
    }

    frame.indirect_command_count = static_cast<std::uint32_t>(frame.indirect_commands.size());

    auto material_result = material_storage_.prepare_frame(command_buffer, frame_index);

    if (!material_result) {
        clear_submissions();

        return std::unexpected(make_material_error(material_result.error()));
    }

    auto upload_result = upload_frame_data(command_buffer, frame);

    auto &&[view, projection] = matrices;
    UBO ubo{
            .view_projection = projection * view,
            .view = view,
            .projection = projection,
            .camera_position = glm::vec3(glm::inverse(view)[3]),
            .fog_colour = glm::vec3{0.5F},
    };
    if (!ubos_[frame_index].write(0, std::as_bytes(std::span{&ubo, 1}))) {
        clear_submissions();

        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    clear_submissions();

    auto &frame_query = timestamp_queries_[frame_index];
    if (frame_query.has_results) {
        std::vector<std::uint64_t> results(query_count);

        VkResult query_res = vkGetQueryPoolResults(context_.device, frame_query.query_pool, 0, query_count,
                                                   sizeof(std::uint64_t) * query_count, results.data(),
                                                   sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);

        if (query_res == VK_SUCCESS) {
            for (std::uint32_t i = 0; i < stage_count; ++i) {
                std::uint64_t const start = results[i * 2];
                std::uint64_t const end = results[i * 2 + 1];

                last_frame_timings_.milliseconds[i] =
                        static_cast<float>(end - start) * timestamp_period_ / 1000000.0f;
            }

            last_frame_timings_.valid = true;
        }

        frame_query.has_results = false;
    }

    return upload_result;
}

auto Renderer::record_frame(VkCommandBuffer command_buffer, SwapchainImage const &swapchain_image,
                            std::uint32_t frame_index, std::function<void(VkCommandBuffer)> const &overlay)
        -> std::expected<void, RendererError> {
    if (!initialized_ || command_buffer == VK_NULL_HANDLE || swapchain_image.image == VK_NULL_HANDLE ||
        swapchain_image.view == VK_NULL_HANDLE || swapchain_image.format == VK_FORMAT_UNDEFINED ||
        swapchain_image.extent.width == 0 || swapchain_image.extent.height == 0 || frame_index >= frames_.size()) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto &frame_query = timestamp_queries_[frame_index];
    vkCmdResetQueryPool(command_buffer, frame_query.query_pool, 0, query_count);

    vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame_query.query_pool,
                         static_cast<std::uint32_t>(RenderStage::FullFrame) * 2);

    auto &frame = frames_[frame_index];
    auto const *hdr = image_storage_.get(frame.forward_target.hdr());
    auto const *depth = image_storage_.get(frame.forward_target.depth());

    if (hdr == nullptr || depth == nullptr || !hdr->valid() || !depth->valid()) {
        return std::unexpected(make_error(RendererErrorType::image_error));
    }

    auto const *forward_pipeline = pipeline_graph_.resolve(forward_pipeline_);
    if (forward_pipeline == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const *depth_prepass_pipeline = pipeline_graph_.resolve(depth_prepass_pipeline_);
    if (depth_prepass_pipeline == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const *composite_pipeline = pipeline_graph_.resolve(composite_pipeline_);
    if (composite_pipeline == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const target_extent = frame.forward_target.extent();

    if (target_extent.width == 0 || target_extent.height == 0 || hdr->extent_2d().width != target_extent.width ||
        hdr->extent_2d().height != target_extent.height || depth->extent_2d().width != target_extent.width ||
        depth->extent_2d().height != target_extent.height) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    transition_forward_target_to_attachments(command_buffer, *hdr, *depth);

#pragma region Predepth pass
    {
        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame_query.query_pool,
                             static_cast<std::uint32_t>(RenderStage::DepthPrepass) * 2);

        VkRenderingAttachmentInfo const depth_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = depth->view(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue =
                        VkClearValue{
                                .depthStencil =
                                        VkClearDepthStencilValue{
                                                .depth = 0.0F,
                                                .stencil = 0,
                                        },
                        },
        };

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
        vkCmdBindPipeline(command_buffer, depth_prepass_pipeline->bind_point(), depth_prepass_pipeline->pipeline());
        gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 depth_prepass_pipeline->layout());
        set_forward_dynamic_state(command_buffer, target_extent, true);

        ForwardPushConstants pc{
                .draw_buffer_address = frame.draw_buffer.device_address,
                .transform_buffer_address = frame.transform_buffer.device_address,
                .material_buffer_address = material_storage_.device_address(),
                .ubo_buffer_address = ubos_[frame_index].device_address,
        };

        vkCmdPushConstants(command_buffer, depth_prepass_pipeline->layout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(ForwardPushConstants), &pc);
        vkCmdBindIndexBuffer(command_buffer, geometry_arena_.buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        if (frame.indirect_command_count != 0) {
            vkCmdDrawIndexedIndirect(command_buffer, frame.indirect_buffer.buffer, 0, frame.indirect_command_count,
                                     sizeof(VkDrawIndexedIndirectCommand));
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

        VkRenderingAttachmentInfo const hdr_attachment{
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
        vkCmdBindPipeline(command_buffer, forward_pipeline->bind_point(), forward_pipeline->pipeline());
        gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 forward_pipeline->layout());
        set_forward_dynamic_state(command_buffer, target_extent, false);

        ForwardPushConstants pc{
                .draw_buffer_address = frame.draw_buffer.device_address,
                .transform_buffer_address = frame.transform_buffer.device_address,
                .material_buffer_address = material_storage_.device_address(),
                .ubo_buffer_address = ubos_[frame_index].device_address,
        };

        vkCmdPushConstants(command_buffer, forward_pipeline->layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ForwardPushConstants),
                           &pc);

        vkCmdBindIndexBuffer(command_buffer, geometry_arena_.buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        if (frame.indirect_command_count != 0) {
            vkCmdDrawIndexedIndirect(command_buffer, frame.indirect_buffer.buffer, 0, frame.indirect_command_count,
                                     sizeof(VkDrawIndexedIndirectCommand));
        }

        vkCmdEndRendering(command_buffer);

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, frame_query.query_pool,
                             (static_cast<std::uint32_t>(RenderStage::ForwardPass) * 2) + 1);
    }
#pragma endregion

    transition_hdr_to_shader_read(command_buffer, *hdr);
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
        vkCmdBindPipeline(command_buffer, composite_pipeline->bind_point(), composite_pipeline->pipeline());
        gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 composite_pipeline->layout());
        set_composite_dynamic_state(command_buffer, swapchain_image.extent);

        struct CompositePC {
            std::uint32_t hdr_texture_index;
            std::uint32_t sampler_index;
            float exposure;
            std::uint32_t padding;
        };

        CompositePC const composite_pc{
                .hdr_texture_index = frame.forward_target.hdr().index,
                .sampler_index = sampler_storage_.linear_clamp().index,
                .exposure = 1.0F,
                .padding = 0,
        };

        vkCmdPushConstants(command_buffer, composite_pipeline->layout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(composite_pc), &composite_pc);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);

        if (overlay) {
            overlay(command_buffer);
        }

        vkCmdEndRendering(command_buffer);

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, frame_query.query_pool,
                             (static_cast<std::uint32_t>(RenderStage::Composition) * 2) + 1);
    }
#pragma endregion

    transition_swapchain_to_present(command_buffer, swapchain_image.image);

    // Full frame end
    vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, frame_query.query_pool,
                         (static_cast<std::uint32_t>(RenderStage::FullFrame) * 2) + 1);

    frame_query.has_results = true;
    return {};
}

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
                    .forward_target_error = replacement.error(),
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

void Renderer::queue_render_thread_event(std::function<void()> &&task) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    event_queue_.push(std::move(task));
    queued_events_.fetch_add(1);

    context_.render_wake_condition.notify_one();
}

auto Renderer::wait_idle() -> std::expected<void, RendererError> {
    auto result = vkDeviceWaitIdle(context_.device);
    return result == VK_SUCCESS ? std::expected<void, RendererError>{}
                                : std::unexpected<RendererError>(RendererError{
                                          .type = RendererErrorType::device_error,
                                          .device_error =
                                                  DeviceError{
                                                          .type = DeviceError::Type::Unknown,
                                                          .message = FlyString{"Could not wait"},
                                                          .vk_result = result,
                                                  },
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

auto Renderer::upload_frame_data(VkCommandBuffer command_buffer, RendererFrame &frame)
        -> std::expected<void, RendererError> {

    auto const draw_size = static_cast<VkDeviceSize>(frame.draws.size()) * sizeof(GpuDraw);
    auto const transform_size = static_cast<VkDeviceSize>(frame.transforms.size()) * sizeof(glm::mat4);
    auto const indirect_size =
            static_cast<VkDeviceSize>(frame.indirect_commands.size()) * sizeof(VkDrawIndexedIndirectCommand);

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

    struct CopyOperation {
        VkBuffer destination = VK_NULL_HANDLE;
        VkBufferCopy2 region{};
    };

    std::array<CopyOperation, 3> copies{};
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

    std::array<VkBufferMemoryBarrier2, 3> barriers{};
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

    if (indirect_size != 0) {
        barriers[barrier_count++] = VkBufferMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = frame.indirect_buffer.buffer,
                .offset = 0,
                .size = indirect_size,
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
}
