#include "renderer.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <expected>
#include <future>
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
#include "render_passes.hxx"
#include "renderer_application_policy.hxx"
#include "sampler_storage.hxx"
#include "slang_compiler.hxx"
#include "vk_barrier.hxx"

// ForwardPushConstants, ShadowPushConstants, CompositePushConstants,
// LightIconPushConstants, CullPushConstants, DownsamplePushConstants, and
// UpsamplePushConstants are generated at build time by reflecting the
// corresponding .slang shader's push_constant block -- see the "Shader
// push-constant reflection" section of CMakeLists.txt.
#include "shader_push_constants.hxx"

namespace {
    template<typename Action>
    struct FinalAction {
        Action action;

        ~FinalAction() { action(); }
    };

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
    static auto thread_pool_ = std::make_unique<BS::priority_thread_pool>(std::thread::hardware_concurrency());
    return *thread_pool_;
}

auto Renderer::compiler() noexcept -> renderer::SlangCompiler & {
    static auto compiler_ =
            std::make_unique<renderer::SlangCompiler>(std::move(renderer::SlangCompiler::create().value()));
    return *compiler_;
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
    pipeline_infos.reserve(11);

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
    auto registered_pipelines = pipeline_graph_.register_pipelines_parallel(pipeline_infos);
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

    auto mesh_storage = MeshStorage::create(MeshStorageCreateInfo{.capacity = create_info.mesh_capacity});

    if (!mesh_storage) {
        error("Could not create mesh storage");
        return std::unexpected(RendererError{.type = RendererErrorType::invalid_argument});
    }

    mesh_storage_ = std::move(*mesh_storage);

    auto model_storage = ModelStorage::create(ModelStorageCreateInfo{.capacity = create_info.model_capacity});

    if (!model_storage) {
        error("Could not create model storage");
        return std::unexpected(RendererError{.type = RendererErrorType::invalid_argument});
    }

    model_storage_ = std::move(*model_storage);

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

    // Must wait before destroying anything a still-running background load
    // could touch when it finishes: model_streamer_'s finish_model_load()
    // creates meshes/materials (mesh_storage_/material_storage_) and writes
    // into model_storage_/geometry_arena_, so it has to drain first.
    model_streamer_.wait_all();

    material_storage_.destroy();
    texture_streamer_.wait_all();
    image_storage_.destroy();
    geometry_arena_.destroy(context_);
    compiler().destroy();

    clear_submissions();

    model_storage_.destroy();
    mesh_storage_.destroy();

    default_material_handle_ = {};

    forward_pipeline_ = {};
    composite_pipeline_ = {};

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

    // Keyed on the resolved path rather than file content: hashing only the
    // first 64 bytes (the previous approach) let two different models with
    // the same header -- e.g. two glTFs from the same exporter -- collide
    // and silently return each other's cached ModelHandle. Models aren't
    // hot-reloaded, so a path-based key loses nothing while removing both
    // the collision risk and a redundant file open+read on every call.
    std::error_code canonicalize_error;
    auto const canonical_path = std::filesystem::weakly_canonical(path, canonicalize_error);
    auto const &cache_key_path = canonicalize_error ? path : canonical_path;
    std::size_t const file_hash = std::filesystem::hash_value(cache_key_path);

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

auto Renderer::load_model_cpu_async(std::filesystem::path path)
        -> std::future<std::expected<ModelCpuData, ModelLoadError>> {
    // sampler_storage_ is only read here (nearest/linear clamp/repeat are
    // fixed defaults resolved once at Renderer::initialize, never mutated
    // afterwards -- see SamplerStorage), so calling load_model_cpu()
    // concurrently with anything the render thread does to sampler_storage_
    // is safe without additional synchronization.
    return thread_pool().submit_task([this, path = std::move(path)] { return load_model_cpu(path, sampler_storage_); });
}

auto Renderer::create_model_from_cpu_data(ModelCpuData const &cpu_data) -> std::expected<ModelHandle, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    std::expected<Model, ModelLoadError> imported_model{
            std::unexpected(ModelLoadError{.type = ModelLoadErrorType::invalid_argument}),
    };

    context_.one_time_submit([&](VkCommandBuffer command_buffer) {
        imported_model = record_model_gpu_upload(cpu_data, command_buffer, geometry_arena_, image_storage_,
                                                 texture_streamer_, material_storage_);
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
    auto const model_capacity_exceeded = model_storage_.size() >= model_storage_.capacity();

    if (!initialized_ || model_capacity_exceeded) {
        return std::unexpected(make_error(model_capacity_exceeded ? RendererErrorType::capacity_exceeded
                                                                  : RendererErrorType::invalid_argument));
    }

    return create_model_common(model, fallback_material,
                               [this](ModelSlotData data) { return model_storage_.create_model(std::move(data)); });
}

auto Renderer::create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto handle = model_storage_.create_pending_model(fallback);

    if (!handle) {
        return std::unexpected(make_error(handle.error().type == ModelStorageErrorType::capacity_exceeded
                                                  ? RendererErrorType::capacity_exceeded
                                                  : RendererErrorType::invalid_argument));
    }

    return *handle;
}

auto Renderer::finish_model_load(ModelHandle pending, ModelCpuData const &cpu_data, VkCommandBuffer command_buffer)
        -> std::expected<void, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto imported_model = record_model_gpu_upload(cpu_data, command_buffer, geometry_arena_, image_storage_,
                                                  texture_streamer_, material_storage_);

    if (!imported_model) {
        return std::unexpected(make_model_load_error(imported_model.error()));
    }

    auto handle = create_model_common(*imported_model, default_material_handle_, [this, pending](ModelSlotData data) {
        return model_storage_.upgrade_pending_model(pending, std::move(data));
    });

    if (!handle) {
        return std::unexpected(handle.error());
    }

    return {};
}

auto Renderer::create_model_common(
        Model const &model, MaterialHandle fallback_material,
        std::move_only_function<std::expected<ModelHandle, ModelStorageError>(ModelSlotData)> install)
        -> std::expected<ModelHandle, RendererError> {
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
                    .lods = source_submesh.lods,
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

    auto handle = install(ModelSlotData{
            .draws = std::move(flattened_draws),
            .bounds_min = model.bounds_min,
            .bounds_max = model.bounds_max,
            .lights = model.lights,
    });

    if (!handle) {
        rollback_meshes();

        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    return *handle;
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

    if (mesh_storage_.size() >= mesh_storage_.capacity()) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    std::vector<Submesh> submeshes;
    submeshes.reserve(create_info.submeshes.size());

    for (auto const &submesh_info: create_info.submeshes) {
        auto const &lod0 = submesh_info.lods[0];

        if (!lod0.vertices.bytes.valid() || !lod0.indices.bytes.valid() || lod0.vertices.vertex_count == 0 ||
            lod0.indices.index_count == 0) {
            return std::unexpected(make_error(RendererErrorType::invalid_argument));
        }

        if (material_storage_.get(submesh_info.material) == nullptr) {
            return std::unexpected(make_error(RendererErrorType::invalid_material));
        }

        auto stride = index_stride(lod0.indices.index_type);

        if (!stride) {
            return std::unexpected(stride.error());
        }

        if (lod0.indices.bytes.offset % *stride != 0) {
            return std::unexpected(make_error(RendererErrorType::invalid_argument));
        }

        submeshes.push_back(Submesh{
                .lods = submesh_info.lods,
                .material = submesh_info.material,
                .bounds_min = submesh_info.bounds_min,
                .bounds_max = submesh_info.bounds_max,
        });
    }

    auto handle = mesh_storage_.create_mesh(std::move(submeshes));

    if (!handle) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    return *handle;
}

auto Renderer::destroy_mesh(MeshHandle handle) -> std::expected<void, RendererError> {
    auto result = mesh_storage_.destroy_mesh(handle);

    if (!result) {
        return std::unexpected(make_error(RendererErrorType::invalid_mesh));
    }

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

auto Renderer::prepare_frame(VkCommandBuffer command_buffer, CameraMatrices const &matrices, std::uint32_t frame_index)
        -> std::expected<void, RendererError> {
    ZoneScopedNC("PrepareFrame", tracy::Color::RoyalBlue);

    if (!initialized_ || command_buffer == VK_NULL_HANDLE || frame_index >= frames_.size()) {
        clear_submissions();
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    std::uint32_t submitted_triangle_count = 0;

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

    pipeline_graph_.process_dirty();

    auto image_result = image_storage_.prepare_frame(command_buffer);

    if (!image_result) {
        clear_submissions();

        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    texture_streamer_.process_ready(image_storage_, command_buffer, frame_index);
    model_streamer_.process_ready(*this, command_buffer);

    auto resource_result = gpu_resource_table_.prepare_frame(frame_index, image_storage_, sampler_storage_);

    if (!resource_result) {
        clear_submissions();
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

    auto const camera_position = glm::vec3(glm::inverse(matrices.view)[3]);

    ++batch_frame_;

    if (batch_frame_ == 0) {
        for (auto &[key, batch]: batches_) {
            static_cast<void>(key);
            batch.frame_stamp = 0;
        }
        batch_frame_ = 1;
    }

    active_batches_.clear();

    opaque_batches_.clear();
    mask_batches_.clear();
    blend_batches_.clear();

    auto const select_lod_index = [&camera_position](glm::vec3 const &instance_position) -> std::uint32_t {
        auto const delta = instance_position - camera_position;
        auto const distance_sq = glm::dot(delta, delta);
        std::uint32_t lod_index = 0;
        for (; lod_index < lod_distances.size(); ++lod_index) {
            auto const lod_distance = lod_distances[lod_index];
            if (distance_sq < lod_distance * lod_distance) {
                break;
            }
        }
        return lod_index;
    };

    auto const append_batch_transform = [this](BatchKey const &key, MeshHandle mesh, std::uint32_t submesh_index,
                                               MaterialHandle material, std::uint32_t lod_index,
                                               glm::mat4 const &transform) {
        auto iterator = batches_.try_emplace(key).first;
        auto &batch = iterator->second;

        if (batch.frame_stamp != batch_frame_) {
            batch.mesh = mesh;
            batch.submesh_index = submesh_index;
            batch.material = material;
            batch.lod_index = lod_index;
            batch.transforms.clear();
            batch.frame_stamp = batch_frame_;
            active_batches_.push_back(&batch);
        }
        batch.transforms.push_back(transform);
    };

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
            auto const lod_index = select_lod_index(glm::vec3(instance_transform[3]));
            for (std::uint32_t submesh_index = 0; submesh_index < mesh->submeshes.size(); ++submesh_index) {
                auto const &submesh = mesh->submeshes[submesh_index];
                auto const material = model_submission.material_override.valid() ? model_submission.material_override
                                                                                 : submesh.material;
                auto const key = BatchKey{
                        .mesh_index = model_draw.mesh.index,
                        .submesh_index = submesh_index,
                        .material_index = material_storage_.gpu_index(material),
                        .lod_index = lod_index,
                };

                append_batch_transform(key, model_draw.mesh, submesh_index, material, lod_index, instance_transform);
            }
        }
    }

    for (auto const &submission: submissions_) {
        auto const *mesh = mesh_slot(submission.mesh);
        if (mesh == nullptr) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }

        auto const lod_index = select_lod_index(glm::vec3(submission.transform[3]));

        for (std::uint32_t submesh_index = 0; submesh_index < mesh->submeshes.size(); ++submesh_index) {
            auto const &submesh = mesh->submeshes[submesh_index];
            auto const material =
                    submission.material_override.valid() ? submission.material_override : submesh.material;
            auto const key = BatchKey{
                    .mesh_index = submission.mesh.index,
                    .submesh_index = submesh_index,
                    .material_index = material_storage_.gpu_index(material),
                    .lod_index = lod_index,
            };

            append_batch_transform(key, submission.mesh, submesh_index, material, lod_index, submission.transform);
        }
    }

    opaque_batches_.reserve(active_batches_.size());
    mask_batches_.reserve(active_batches_.size());
    blend_batches_.reserve(active_batches_.size());

    auto const emit_batch = [this, &frame,
                             &submitted_triangle_count](BatchEntry const &batch) -> std::expected<void, RendererError> {
        auto const *mesh = mesh_slot(batch.mesh);

        if (mesh == nullptr) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }

        auto const &submesh = mesh->submeshes[batch.submesh_index];
        auto const &geometry = submesh.lods[batch.lod_index];
        auto const instance_count = static_cast<std::uint32_t>(batch.transforms.size());
        submitted_triangle_count += (geometry.indices.index_count / 3) * instance_count;
        if (frame.transforms.size() + instance_count > maximum_submission_count_ ||
            frame.draws.size() + instance_count > maximum_draw_count_) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
        }
        auto const base_transform_index = static_cast<std::uint32_t>(frame.transforms.size());

        frame.transforms.insert(frame.transforms.end(), batch.transforms.begin(), batch.transforms.end());
        auto const stride = index_stride(geometry.indices.index_type);
        if (!stride) {
            clear_submissions();
            return std::unexpected(stride.error());
        }

        auto const first_index_u64 = geometry.indices.bytes.offset / *stride;
        if (first_index_u64 > std::numeric_limits<std::uint32_t>::max()) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::size_overflow));
        }

        auto const first_instance = static_cast<std::uint32_t>(frame.draws.size());
        auto const vertex_address = geometry_arena_.vertex_address(geometry.vertices);
        auto const material_index = material_storage_.gpu_index(batch.material);
        for (std::uint32_t instance = 0; instance < instance_count; ++instance) {
            frame.draws.push_back(GpuDraw{
                    .vertex_address = vertex_address,
                    .material_index = material_index,
                    .transform_index = base_transform_index + instance,
            });
        }

        frame.indirect_commands.push_back(VkDrawIndexedIndirectCommand{
                .indexCount = geometry.indices.index_count,
                .instanceCount = instance_count,
                .firstIndex = static_cast<std::uint32_t>(first_index_u64),
                .vertexOffset = 0,
                .firstInstance = first_instance,
        });

        auto const *material = material_storage_.get(batch.material);

        auto const wind_padding = material != nullptr ? material->wind_strength : 0.0F;

        frame.batch_bounds.push_back(GpuCullBounds{
                .bounds_min = submesh.bounds_min,
                .wind_padding = wind_padding,
                .bounds_max = submesh.bounds_max,
        });

        return {};
    };

    for (auto const *batch: active_batches_) {
        auto const *material = material_storage_.get(batch->material);
        auto const alpha_mode = material != nullptr ? material->alpha_mode : AlphaMode::opaque;

        switch (alpha_mode) {
            case AlphaMode::opaque:
                opaque_batches_.push_back(batch);
                break;

            case AlphaMode::mask:
                mask_batches_.push_back(batch);
                break;

            case AlphaMode::blend: {
                auto const *mesh = mesh_slot(batch->mesh);

                if (mesh == nullptr) {
                    clear_submissions();
                    return std::unexpected(make_error(RendererErrorType::invalid_mesh));
                }

                auto const &submesh = mesh->submeshes[batch->submesh_index];
                auto const local_centre = (submesh.bounds_min + submesh.bounds_max) * 0.5F;
                auto const world_centre = glm::vec3(batch->transforms.front() * glm::vec4(local_centre, 1.0F));
                auto const distance = world_centre - camera_position;
                blend_batches_.push_back(PendingBlendBatch{
                        .entry = batch,
                        .camera_distance_sq = glm::dot(distance, distance),
                });

                break;
            }
        }
    }

    // Transparent batches need a true back-to-front ordering. For sufficiently
    // large lists, let the renderer's persistent worker pool sort them while
    // the render thread orders the opaque/masked shadow batches below.
    constexpr std::size_t parallel_blend_sort_threshold = 4'096;

    auto sort_blend_batches = [this] {
        std::sort(blend_batches_.begin(), blend_batches_.end(),
                  [](PendingBlendBatch const &lhs, PendingBlendBatch const &rhs) {
                      return lhs.camera_distance_sq > rhs.camera_distance_sq;
                  });
    };

    std::future<void> blend_sort_future;
    if (blend_batches_.size() >= parallel_blend_sort_threshold) {
        blend_sort_future = thread_pool().submit_task(sort_blend_batches);
    } else {
        sort_blend_batches();
    }

    auto const batch_max_shadow_cascade = [this](BatchEntry const *batch) noexcept -> std::int32_t {
        auto const *material = material_storage_.get(batch->material);
        auto const cascade = material != nullptr ? material->max_shadow_cascade : shadow_cascade_count - 1;
        return cascade == GpuMaterial::no_shadow_cascade ? -1 : static_cast<std::int32_t>(cascade);
    };

    // Opaque and masked batches only have shadow_cascade_count + 1 possible
    // ordering keys: max cascade 3..0 plus "does not cast shadows". A general
    // O(n log n) sort is unnecessary. Repeated in-place partitions form those
    // buckets in descending cascade order with no temporary allocations. The
    // partition boundaries are exactly the indirect-command prefix counts
    // needed by each shadow cascade, so there is no later find_if scan either.
    auto const order_shadow_batches = [&batch_max_shadow_cascade](auto &batches) {
        std::array<std::uint32_t, shadow_cascade_count> prefix_counts{};
        auto bucket_begin = batches.begin();

        for (std::int32_t cascade = static_cast<std::int32_t>(shadow_cascade_count) - 1; cascade >= 0; --cascade) {
            bucket_begin = std::partition(bucket_begin, batches.end(), [&](BatchEntry const *batch) {
                return batch_max_shadow_cascade(batch) == cascade;
            });

            prefix_counts[static_cast<std::size_t>(cascade)] =
                    static_cast<std::uint32_t>(std::distance(batches.begin(), bucket_begin));
        }

        return prefix_counts;
    };

    frame.shadow_opaque_indirect_count = order_shadow_batches(opaque_batches_);
    frame.shadow_mask_indirect_count = order_shadow_batches(mask_batches_);

    for (auto const *batch: opaque_batches_) {
        if (auto result = emit_batch(*batch); !result) {
            return std::unexpected(result.error());
        }
    }

    frame.opaque_indirect_count = static_cast<std::uint32_t>(frame.indirect_commands.size());

    for (auto const *batch: mask_batches_) {
        if (auto result = emit_batch(*batch); !result) {
            return std::unexpected(result.error());
        }
    }

    frame.mask_indirect_count =
            static_cast<std::uint32_t>(frame.indirect_commands.size()) - frame.opaque_indirect_count;

    if (blend_sort_future.valid()) {
        blend_sort_future.wait();
    }

    for (auto const &pending: blend_batches_) {
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
            .cascade_atlas_offset_u =
                    glm::vec4{
                            static_cast<float>(shadow_cascade_offset_x[0]),
                            static_cast<float>(shadow_cascade_offset_x[1]),
                            static_cast<float>(shadow_cascade_offset_x[2]),
                            static_cast<float>(shadow_cascade_offset_x[3]),
                    } /
                    static_cast<float>(shadow_atlas_width),
            .cascade_atlas_scale_u =
                    glm::vec4{
                            static_cast<float>(shadow_cascade_resolutions[0]),
                            static_cast<float>(shadow_cascade_resolutions[1]),
                            static_cast<float>(shadow_cascade_resolutions[2]),
                            static_cast<float>(shadow_cascade_resolutions[3]),
                    } /
                    static_cast<float>(shadow_atlas_width),
            .cascade_atlas_scale_v =
                    glm::vec4{
                            static_cast<float>(shadow_cascade_resolutions[0]),
                            static_cast<float>(shadow_cascade_resolutions[1]),
                            static_cast<float>(shadow_cascade_resolutions[2]),
                            static_cast<float>(shadow_cascade_resolutions[3]),
                    } /
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

#pragma region Culling
    {
        TracyVkZoneC(context_.host_query_context.context, command_buffer, "Culling", tracy::Color::SlateBlue);

        constexpr auto stage = static_cast<std::uint32_t>(RenderStage::Culling);
        constexpr auto start_query = stage * 2;
        constexpr auto end_query = start_query + 1;

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, frame_query.query_pool,
                             start_query);

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

            gpu_resource_table_.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     frustum_cull_pipeline);

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

            vkCmdPushConstants(command_buffer, frustum_cull_pipeline, VK_SHADER_STAGE_ALL, 0, sizeof(cull_pc),
                               &cull_pc);

            vkCmdDispatch(command_buffer, frame.indirect_command_count, 1, 1);

            std::array<VkBufferMemoryBarrier2, 3> const post_cull_barriers{
                    VkBufferMemoryBarrier2{
                            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
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

            VkDependencyInfo const dependency_info{
                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .bufferMemoryBarrierCount = static_cast<std::uint32_t>(post_cull_barriers.size()),
                    .pBufferMemoryBarriers = post_cull_barriers.data(),
            };

            vkCmdPipelineBarrier2(command_buffer, &dependency_info);
        }

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, frame_query.query_pool, end_query);
    }
#pragma endregion

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
        last_frame_timings_.valid = false;
        std::array<std::uint64_t, query_count> results{};
        auto const query_result =
                vkGetQueryPoolResults(context_.device, frame_query.query_pool, 0, query_count, sizeof(results),
                                      results.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);

        if (query_result == VK_NOT_READY) {
            warn("Timestamp queries not ready for frame {}", frame_index);
        }

        if (query_result == VK_SUCCESS) {
            for (std::uint32_t i = 0; i < stage_count; ++i) {
                auto const start = results[i * 2];
                auto const end = results[i * 2 + 1];
                last_frame_timings_.milliseconds[i] =
                        static_cast<float>(end - start) * timestamp_period_ / 1'000'000.0F;
            }

            last_frame_timings_.valid = true;
        }

        frame_query.has_results = false;
    }

    if (frame_pipeline_query.has_results) {
        last_frame_pipeline_stats_.valid = false;
        std::array<std::uint64_t, pipeline_stat_count> results{};

        auto const query_result =
                vkGetQueryPoolResults(context_.device, frame_pipeline_query.query_pool, 0, 1, sizeof(results),
                                      results.data(), sizeof(results), VK_QUERY_RESULT_64_BIT);

        if (query_result == VK_SUCCESS) {
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
    ZoneScopedNC("RecordFrame", tracy::Color::RoyalBlue);

    if (!initialized_ || command_buffer == VK_NULL_HANDLE || swapchain_image.image == VK_NULL_HANDLE ||
        swapchain_image.view == VK_NULL_HANDLE || swapchain_image.format == VK_FORMAT_UNDEFINED ||
        swapchain_image.extent.width == 0 || swapchain_image.extent.height == 0 || frame_index >= frames_.size()) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    screenshot_.try_resolve(frame_index);

    auto &frame = frames_[frame_index];
    auto &frame_query = timestamp_queries_[frame_index];

    auto const hdr_handle = frame.forward_target.hdr();
    auto const depth_handle = frame.forward_target.depth();
    auto const resolved_hdr_handle =
            frame.forward_target.is_multisampled() ? frame.forward_target.resolved_hdr() : hdr_handle;
    auto const resolved_depth_handle =
            frame.forward_target.is_multisampled() ? frame.forward_target.resolved_depth() : depth_handle;

    auto const *hdr = image_storage_.get(hdr_handle);
    auto const *depth = image_storage_.get(depth_handle);
    auto const *resolved_hdr = image_storage_.get(resolved_hdr_handle);
    auto const *resolved_depth = image_storage_.get(resolved_depth_handle);
    auto const *shadow_atlas = image_storage_.get(frame.shadow_atlas);

    if (hdr == nullptr || depth == nullptr || resolved_hdr == nullptr || resolved_depth == nullptr ||
        shadow_atlas == nullptr || !hdr->valid() || !depth->valid() || !resolved_hdr->valid() ||
        !resolved_depth->valid() || !shadow_atlas->valid()) {
        return std::unexpected(make_error(RendererErrorType::image_error));
    }

    auto const target_extent = frame.forward_target.extent();
    if (target_extent.width == 0 || target_extent.height == 0 || hdr->extent_2d().width != target_extent.width ||
        hdr->extent_2d().height != target_extent.height || depth->extent_2d().width != target_extent.width ||
        depth->extent_2d().height != target_extent.height) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto const pass_context = render_pass::Context{
            .command_buffer = command_buffer,
            .frame_index = frame_index,
            .pipeline_graph = pipeline_graph_,
            .resource_table = gpu_resource_table_,
            .timestamp_query_pool = frame_query.query_pool,
    };

    auto const main_view_buffers = render_pass::DrawBuffers{
            .draws = frame.visible_draw_buffer,
            .transforms = frame.visible_transform_buffer,
            .indirect = frame.culled_indirect_buffer,
    };

    auto const main_view_counts = render_pass::DrawCounts{
            .opaque = frame.opaque_indirect_count,
            .mask = frame.mask_indirect_count,
            .blend = frame.blend_indirect_count,
    };

    {
        TracyVkZoneC(context_.host_query_context.context, command_buffer, "Shadow Pass", tracy::Color::Purple);

        auto const result =
                render_pass::shadow(pass_context, render_pass::ShadowPassInfo{
                                                          .shadow_atlas = *shadow_atlas,
                                                          .draws =
                                                                  {
                                                                          .draws = frame.draw_buffer,
                                                                          .transforms = frame.transform_buffer,
                                                                          .indirect = frame.indirect_buffer,
                                                                  },
                                                          .counts =
                                                                  {
                                                                          .opaque = frame.opaque_indirect_count,
                                                                          .mask = frame.mask_indirect_count,
                                                                          .blend = frame.blend_indirect_count,
                                                                  },
                                                          .opaque_cascade_counts = frame.shadow_opaque_indirect_count,
                                                          .mask_cascade_counts = frame.shadow_mask_indirect_count,
                                                          .index_buffer = geometry_arena_.bindable_buffer(),
                                                          .materials_address = material_storage_.device_address(),
                                                          .ubo_address = ubos_[frame_index].device_address,
                                                          .lights_address = frame.lights_buffer.device_address,
                                                          .opaque_pipeline = shadow_pipeline_,
                                                          .mask_pipeline = shadow_mask_pipeline_,
                                                          .depth_bias_constant = shadow_settings_.depth_bias_constant,
                                                          .depth_bias_slope = shadow_settings_.depth_bias_slope,
                                                  });

        if (!result) {
            return std::unexpected(result.error());
        }
    }

    render_pass::prepare_forward_targets(
            pass_context, render_pass::ForwardTargets{
                                  .hdr = *hdr,
                                  .depth = *depth,
                                  .resolved_hdr = frame.forward_target.is_multisampled() ? resolved_hdr : nullptr,
                                  .resolved_depth = frame.forward_target.is_multisampled() ? resolved_depth : nullptr,
                          });

    {
        TracyVkZoneC(context_.host_query_context.context, command_buffer, "Depth Prepass", tracy::Color::SlateGray);

        auto const result = render_pass::depth_prepass(
                pass_context,
                render_pass::DepthPrepassInfo{
                        .depth = *depth,
                        .resolved_depth = frame.forward_target.is_multisampled() ? resolved_depth : nullptr,
                        .extent = target_extent,
                        .samples = samples_,
                        .draws = main_view_buffers,
                        .counts = main_view_counts,
                        .index_buffer = geometry_arena_.bindable_buffer(),
                        .materials_address = material_storage_.device_address(),
                        .ubo_address = ubos_[frame_index].device_address,
                        .lights_address = frame.lights_buffer.device_address,
                        .opaque_pipeline = depth_prepass_pipeline_,
                        .mask_pipeline = depth_prepass_mask_pipeline_,
                });

        if (!result) {
            return std::unexpected(result.error());
        }
    }

    auto debug_overlay = [&] { OverlayPolicy::render_debug(app, command_buffer, vp, frame_index); };

    auto hdr_output = [&]() -> std::expected<render_pass::HdrTextureIndex, RendererError> {
        TracyVkZoneC(context_.host_query_context.context, command_buffer, "Forward Pass", tracy::Color::RoyalBlue);

        return render_pass::forward_geometry(
                pass_context,
                render_pass::ForwardGeometryInfo{
                        .hdr = *hdr,
                        .depth = *depth,
                        .resolved_hdr = frame.forward_target.is_multisampled() ? resolved_hdr : nullptr,
                        .output_hdr = {.index = resolved_hdr_handle.index},
                        .extent = target_extent,
                        .samples = samples_,
                        .draws = main_view_buffers,
                        .counts = main_view_counts,
                        .index_buffer = geometry_arena_.bindable_buffer(),
                        .materials_address = material_storage_.device_address(),
                        .ubo_address = ubos_[frame_index].device_address,
                        .lights_address = frame.lights_buffer.device_address,
                        .light_count = frame.light_count,
                        .pipeline_statistics_query_pool = pipeline_stat_queries_[frame_index].query_pool,
                        .opaque_pipeline = forward_pipeline_,
                        .blend_pipeline = forward_blend_pipeline_,
                        .draw_light_icons = debug_draw_light_icons_,
                        .light_icon_pipeline = light_icon_pipeline_,
                        .light_icon_texture_index = light_icon_texture_.index,
                        .linear_sampler_index = sampler_storage_.linear_clamp().index,
                        .light_icon_world_size = light_icon_world_size_,
                },
                render_pass::Callback::bind(debug_overlay));
    }();

    if (!hdr_output) {
        return std::unexpected(hdr_output.error());
    }

    auto const *bloom_target = bloom_settings_.enabled ? image_storage_.get(frame.bloom_target.image) : nullptr;

    auto bloom_output = [&]() -> std::expected<std::optional<render_pass::BloomTextureIndex>, RendererError> {
        TracyVkZoneC(context_.host_query_context.context, command_buffer, "Bloom Pass", tracy::Color::Orange);

        return render_pass::bloom(pass_context, render_pass::BloomPassInfo{
                                                        .enabled = bloom_settings_.enabled,
                                                        .input_hdr = *hdr_output,
                                                        .target = bloom_target,
                                                        .mip_texture_indices =
                                                                {
                                                                        frame.bloom_target.mip_slots[0].index,
                                                                        frame.bloom_target.mip_slots[1].index,
                                                                        frame.bloom_target.mip_slots[2].index,
                                                                        frame.bloom_target.mip_slots[3].index,
                                                                },
                                                        .input_extent = target_extent,
                                                        .downsample_pipeline = bloom_downsample_pipeline_,
                                                        .upsample_pipeline = bloom_upsample_pipeline_,
                                                        .linear_sampler_index = sampler_storage_.linear_clamp().index,
                                                        .threshold = bloom_settings_.threshold,
                                                        .knee = bloom_settings_.knee,
                                                        .filter_radius = bloom_settings_.filter_radius,
                                                });
    }();

    if (!bloom_output) {
        return std::unexpected(bloom_output.error());
    }

    auto ui_overlay = [&] { OverlayPolicy::render_ui(app, command_buffer, frame_index); };
    {
        TracyVkZoneC(context_.host_query_context.context, command_buffer, "Composition", tracy::Color::SeaGreen);

        auto const result =
                render_pass::composite(pass_context,
                                       render_pass::CompositePassInfo{
                                               .swapchain_image = swapchain_image.image,
                                               .swapchain_view = swapchain_image.view,
                                               .extent = swapchain_image.extent,
                                               .hdr = *hdr_output,
                                               .bloom = *bloom_output,
                                               // The default emissive texture is the renderer's valid black texture.
                                               .bloom_fallback_texture_index = image_storage_.emissive().index,
                                               .linear_sampler_index = sampler_storage_.linear_clamp().index,
                                               .pipeline = composite_pipeline_,
                                               .exposure = 1.0F,
                                               .bloom_intensity = bloom_settings_.intensity,
                                       },
                                       render_pass::Callback::bind(ui_overlay));

        if (!result) {
            return std::unexpected(result.error());
        }
    }

    bool const screenshot_recorded = screenshot_.record(context_, command_buffer, swapchain_image.image,
                                                        swapchain_image.format, swapchain_image.extent, frame_index);

    if (!screenshot_recorded) {
        render_pass::present_swapchain(command_buffer, swapchain_image.image);
    }

    vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, frame_query.query_pool,
                         (static_cast<std::uint32_t>(RenderStage::FullFrame) * 2) + 1);

    frame_query.has_results = true;
    pipeline_stat_queries_[frame_index].has_results = true;

    TracyVkCollectHost(context_.host_query_context.context);
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

    // Destroying frames_[i].forward_target below is only safe once the GPU is
    // done with it. The caller (main.cxx) currently only ever calls resize()
    // right after Swapchain::recreate(), which itself does a full
    // vkDeviceWaitIdle -- but that ordering isn't enforced here, so a future
    // caller (or a reordering of the resize path) could destroy an image the
    // GPU is still rendering into. Wait unconditionally so this function is
    // correct on its own.
    if (auto waited = wait_idle(); !waited) {
        return std::unexpected(waited.error());
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

auto Renderer::mesh_slot(MeshHandle handle) noexcept -> MeshSlotData * { return mesh_storage_.get(handle); }

auto Renderer::mesh_slot(MeshHandle handle) const noexcept -> MeshSlotData const * { return mesh_storage_.get(handle); }

auto Renderer::model_slot(ModelHandle handle) noexcept -> ModelSlotData * { return model_storage_.get(handle); }

auto Renderer::model_slot(ModelHandle handle) const noexcept -> ModelSlotData const * {
    return model_storage_.get(handle);
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
