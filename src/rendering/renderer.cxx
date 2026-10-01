#include "rendering/renderer.hxx"

#include <algorithm>
#include <array>
#include <bit>
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

#include "assets/material_storage.hxx"
#include "assets/slang_compiler.hxx"
#include "core/logger.hxx"
#include "core/thread_pool.hxx"
#include "gpu/buffer.hxx"
#include "gpu/context.hxx"
#include "gpu/device_error.hxx"
#include "gpu/gpu_resource_table.hxx"
#include "gpu/sampler_storage.hxx"
#include "gpu/vk_barrier.hxx"
#include "maths/aabb.hxx"
#include "rendering/render_passes.hxx"
#include "rendering/screenshot.hxx"

// Generated at build time from the shaders' push_constant blocks (see CMakeLists.txt).
#include "shader_push_constants.hxx"

namespace {
    // Per overlay timing slot: prepare() begin/end and record() begin/end.
    constexpr std::uint32_t queries_per_overlay = 4;
    constexpr std::uint32_t overlay_query_base = query_count;
    constexpr std::uint32_t total_query_count = query_count + (OverlayRegistry::max_overlays * queries_per_overlay);

    [[nodiscard]] constexpr auto overlay_query(std::uint32_t slot, std::uint32_t which) noexcept -> std::uint32_t {
        return overlay_query_base + (slot * queries_per_overlay) + which;
    }

    [[nodiscard]] constexpr auto model_source_key(ModelHandle handle) noexcept -> std::uint64_t {
        return (static_cast<std::uint64_t>(handle.generation) << 32U) | handle.index;
    }
} // namespace

namespace {
    [[nodiscard]]
    inline auto compile_stages_parallel(std::span<renderer::ShaderCompileRequest const> requests)
            -> std::vector<std::expected<renderer::CompiledShader, renderer::ShaderCompileError>> {
        auto &pool = thread_pool();

        std::vector<std::future<std::expected<renderer::CompiledShader, renderer::ShaderCompileError>>> futures;
        futures.reserve(requests.size());

        for (auto const &request: requests) {
            futures.push_back(pool.submit_task([&request] { return Renderer::compiler().compile(request); }));
        }

        std::vector<std::expected<renderer::CompiledShader, renderer::ShaderCompileError>> results;
        results.reserve(requests.size());

        for (auto &future: futures) {
            results.push_back(future.get());
        }

        return results;
    }

    template<typename Action>
    struct FinalAction {
        Action action;

        ~FinalAction() { action(); }
    };

    // GpuDrawCommand's leading fields are read by Vulkan as a VkDrawMeshTasksIndirectCommandEXT.
    static_assert(sizeof(VkDrawMeshTasksIndirectCommandEXT) == 12);
    static_assert(offsetof(GpuDrawCommand, group_count_x) == offsetof(VkDrawMeshTasksIndirectCommandEXT, groupCountX));
    static_assert(offsetof(GpuDrawCommand, group_count_y) == offsetof(VkDrawMeshTasksIndirectCommandEXT, groupCountY));
    static_assert(offsetof(GpuDrawCommand, group_count_z) == offsetof(VkDrawMeshTasksIndirectCommandEXT, groupCountZ));
} // namespace

namespace {
    [[nodiscard]]
    auto resolve_layout(PipelineGraphRepository const &graph, PipelineNodeHandle handle) noexcept -> VkPipelineLayout {
        if (auto const *shader_objects = graph.resolve_shader_objects(handle); shader_objects != nullptr) {
            return shader_objects->layout();
        }

        return VK_NULL_HANDLE;
    }

    auto bind_compute_node(PipelineGraphRepository const &graph, PipelineNodeHandle handle,
                           VkCommandBuffer command_buffer) noexcept -> void {
        if (auto const *shader_objects = graph.resolve_shader_objects(handle); shader_objects != nullptr) {
            shader_objects->bind(command_buffer);
            return;
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
                .cause = ErrorCause{Boxed<DeviceError>{error}},
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

    // Owns a mesh through Renderer::destroy_mesh(), which also retires its geometry.
    using MeshHolder = Holder<Renderer, MeshHandle, &Renderer::destroy_mesh>;

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

    [[nodiscard]] auto nearly_equal(float lhs, float rhs, float epsilon = 1e-5F) noexcept -> bool {
        auto const scale = std::max({1.0F, std::abs(lhs), std::abs(rhs)});
        return std::abs(lhs - rhs) <= epsilon * scale;
    }

    [[nodiscard]] auto nearly_equal(glm::vec3 const &lhs, glm::vec3 const &rhs, float epsilon = 1e-5F) noexcept
            -> bool {
        return nearly_equal(lhs.x, rhs.x, epsilon) && nearly_equal(lhs.y, rhs.y, epsilon) &&
               nearly_equal(lhs.z, rhs.z, epsilon);
    }

    [[nodiscard]] auto nearly_equal(glm::mat4 const &lhs, glm::mat4 const &rhs, float epsilon = 1e-5F) noexcept
            -> bool {
        for (auto column = 0; column < glm::mat4::length(); ++column) {
            for (auto row = 0; row < glm::mat4::length(); ++row) {
                if (!nearly_equal(lhs[column][row], rhs[column][row], epsilon)) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] auto shadow_signature_mix(std::uint64_t value) noexcept -> std::uint64_t {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    [[nodiscard]] auto shadow_signature_combine(std::uint64_t seed, std::uint64_t value) noexcept -> std::uint64_t {
        return seed ^ (shadow_signature_mix(value) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
    }
} // namespace

Renderer::Renderer(VulkanContext &context) noexcept :
    context_(context), screenshot_(std::make_unique<ScreenshotCapture>()) {}
Renderer::~Renderer() noexcept = default;

auto Renderer::compiler() noexcept -> renderer::SlangCompiler & {
    static auto compiler_ =
            std::make_unique<renderer::SlangCompiler>(std::move(renderer::SlangCompiler::create().value()));
    return *compiler_;
}

auto Renderer::initialize(RendererCreateInfo const &create_info) -> std::expected<void, RendererError> {
    debug("[Renderer::initialize] enter");

    if (initialized_ || create_info.extent.width == 0 || create_info.extent.height == 0 || frames_in_flight == 0 ||
        create_info.material_capacity < 2 || create_info.mesh_capacity < 2 || create_info.model_capacity < 2 ||
        create_info.script_capacity < 2 || create_info.maximum_draw_count == 0 ||
        create_info.maximum_submission_count == 0) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    hdr_format_ = create_info.hdr_format;
    depth_format_ = create_info.depth_format;
    swapchain_format_ = create_info.swapchain_format;
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
                              .shader_binary_cache_directory = "cache/shader_binaries",
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

    // Scene pipelines are task + mesh (+ fragment), each with an instanced vertex-shader variant registered after
    // the rest for batches too small for meshlets. All startup pipelines are registered in one parallel batch;
    // the indices below must match the push_back() order.
    std::vector<PipelineRegisterInfo> pipeline_infos;
    pipeline_infos.reserve(13);

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/forward_geom.slang",
                                    .entry_point = "main_task",
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/forward_geom.slang",
                                    .entry_point = "main_mesh",
                                    .stage = renderer::ShaderStage::mesh,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/forward_geom.slang",
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
            .debug_name = "renderer.forward_pipeline",
    }); // index 0: forward

    // Forward with alpha blending.
    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/forward_geom.slang",
                                    .entry_point = "main_task",
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/forward_geom.slang",
                                    .entry_point = "main_mesh",
                                    .stage = renderer::ShaderStage::mesh,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/forward_geom.slang",
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
            .debug_name = "renderer.light_icon_pipeline",
    }); // index 2: light_icon

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/shadow_depth.slang",
                                    .entry_point = "main_task",
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/shadow_depth.slang",
                                    .entry_point = "main_mesh",
                                    .stage = renderer::ShaderStage::mesh,
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
            .debug_name = "renderer.shadow_pipeline",
    }); // index 3: shadow

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/shadow_depth.slang",
                                    .entry_point = "main_task",
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/shadow_depth.slang",
                                    .entry_point = "main_mesh",
                                    .stage = renderer::ShaderStage::mesh,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/shadow_depth.slang",
                                    .entry_point = "main_fs",
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
            .debug_name = "renderer.shadow_mask_pipeline",
    }); // index 4: shadow_mask

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/depth_prepass.slang",
                                    .entry_point = "main_task",
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/depth_prepass.slang",
                                    .entry_point = "main_mesh",
                                    .stage = renderer::ShaderStage::mesh,
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
            .debug_name = "renderer.depth_prepass_pipeline",
    }); // index 5: depth_prepass

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/depth_prepass.slang",
                                    .entry_point = "main_task",
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/depth_prepass.slang",
                                    .entry_point = "main_mesh",
                                    .stage = renderer::ShaderStage::mesh,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/depth_prepass.slang",
                                    .entry_point = "main_fs",
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
            .debug_name = "renderer.depth_prepass_mask_pipeline",
    }); // index 6: depth_prepass_mask

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/composite.slang",
                                    .entry_point = "main_vs",
                                    .stage = renderer::ShaderStage::vertex,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/composite.slang",
                                    .entry_point = "main_fs",
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
            .debug_name = "renderer.composite_pipeline",
    }); // index 7: composite

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/frustum_cull.slang",
                                    .entry_point = "main_cs",
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
            .debug_name = "renderer.frustum_cull_pipeline",
    }); // index 8: frustum_cull

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/bloom_downsample.slang",
                                    .entry_point = "main_cs",
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
            .debug_name = "renderer.bloom_downsample_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/bloom_upsample.slang",
                                    .entry_point = "main_cs",
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
            .debug_name = "renderer.bloom_upsample_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/gtao.slang",
                                    .entry_point = "main_cs",
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
            .debug_name = "renderer.gtao_pipeline",
    }); // index 11: gtao

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = "assets/shaders/gtao_denoise.slang",
                                    .entry_point = "main_cs",
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
            .debug_name = "renderer.gtao_denoise_pipeline",
    }); // index 12: gtao_denoise

    // Instanced variants of the scene pipelines (indices 13..18): main_task + main_mesh become main_vs.
    for (std::size_t const meshlet_index: {0U, 1U, 3U, 4U, 5U, 6U}) {
        auto instanced = pipeline_infos[meshlet_index];

        std::erase_if(instanced.stages, [](renderer::ShaderCompileRequest const &stage) {
            return stage.stage == renderer::ShaderStage::task;
        });

        for (auto &stage: instanced.stages) {
            if (stage.stage == renderer::ShaderStage::mesh) {
                stage.stage = renderer::ShaderStage::vertex;
                stage.entry_point = "main_vs";
            }
        }

        instanced.debug_name += "_instanced";
        pipeline_infos.push_back(std::move(instanced));
    }

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
    gtao_pipeline_ = *registered_pipelines[11];
    gtao_denoise_pipeline_ = *registered_pipelines[12];
    forward_instanced_pipeline_ = *registered_pipelines[13];
    forward_blend_instanced_pipeline_ = *registered_pipelines[14];
    shadow_instanced_pipeline_ = *registered_pipelines[15];
    shadow_mask_instanced_pipeline_ = *registered_pipelines[16];
    depth_prepass_instanced_pipeline_ = *registered_pipelines[17];
    depth_prepass_mask_instanced_pipeline_ = *registered_pipelines[18];

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
    MaterialCreateInfo const mat{
            .base_colour_factor = {1.0F, 1.0F, 1.0F, 1.0F},
            .emissive_factor = {0.0F, 0.0F, 0.0F},
            .emissive_strength = 1.0F,
            .metallic_factor = 0.0F,
            .roughness_factor = 1.0F,
            .normal_scale = 1.0F,
            .occlusion_strength = 1.0F,
            .alpha_cutoff = 0.5F,
            .base_colour_texture = white,
            .normal_texture = flat_normal,
            .metallic_roughness_texture = metallic_roughness,
            .occlusion_texture = occlusion,
            .emissive_texture = emissive,
            .sampler = sampler_storage_.linear_repeat(),
            .alpha_mode = AlphaMode::opaque,
    };

    if (!material_storage_.update_material(def_mat, mat)) {
        error("Could not update default material");
        return std::unexpected(RendererError{
                .type = RendererErrorType::material_error,
        });
    }

    default_material_handle_ = def_mat;
    static_cast<void>(assets_.materials().register_asset("Default", default_material_handle_));

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

    auto script_storage = ScriptStorage::create(ScriptStorageCreateInfo{.capacity = create_info.script_capacity});

    if (!script_storage) {
        error("Could not create script storage");
        return std::unexpected(RendererError{.type = RendererErrorType::invalid_argument});
    }

    script_storage_ = std::move(*script_storage);

    maximum_draw_count_ = create_info.maximum_draw_count;
    maximum_submission_count_ = create_info.maximum_submission_count;
    submissions_.reserve(maximum_submission_count_);
    model_submissions_.reserve(maximum_submission_count_);
    auto draw_size_result = checked_multiply(sizeof(GpuDraw), maximum_draw_count_);
    auto transform_size_result = checked_multiply(sizeof(glm::mat4), maximum_submission_count_);
    auto indirect_size_result = checked_multiply(sizeof(GpuDrawCommand), maximum_draw_count_);
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

    auto shadow_atlas = create_held_image(image_storage_, ImageCreateInfo{
            .extent = VkExtent3D{.width = shadow_atlas_width, .height = shadow_atlas_height, .depth = 1},
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
            .debug_name = "renderer.shadow_atlas",
    });

    if (!shadow_atlas) {
        return std::unexpected(make_image_error(shadow_atlas.error()));
    }

    shadow_atlas_ = std::move(*shadow_atlas);

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

        // main_cs is the only writer. TRANSFER_SRC for the visible-instance readback.
        auto culled_indirect = Buffer::create(
                context_, BufferCreateInfo{
                                  .size = culled_indirect_size,
                                  .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
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

        // Host-written every frame.
        auto frustum_planes_buffer =
                Buffer::create(context_, BufferCreateInfo{
                                                 .size = sizeof(glm::vec4) * cull_plane_count,
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

        auto targets = create_frame_targets(frame_index, create_info.extent);

        if (!targets) {
            return std::unexpected(targets.error());
        }

        frame.forward_target = std::move(targets->forward_target);
        frame.viewport_target = std::move(targets->viewport_target);
        frame.bloom_target = std::move(targets->bloom_target);
        frame.ao_target = std::move(targets->ao_target);

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
            .queryCount = total_query_count,
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
        vkResetQueryPool(context_.device, query_pool, 0, total_query_count);
    }

    pipeline_stat_queries_.resize(frames_in_flight);
    VkQueryPoolCreateInfo const pipeline_stat_pool_info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS,
            .queryCount = 1,
            // Results come back in bit order. Clipping primitives is incompatible with mesh draws
            // (VUID-vkCmdDrawMeshTasksIndirectEXT-pipelineStatistics-07076), so it is only used without mesh support.
            .pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
                                  (context_.mesh_shader_queries_supported
                                           ? VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT |
                                                     VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT
                                           : static_cast<VkQueryPipelineStatisticFlags>(
                                                     VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT)),
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

    if (auto light_icons = register_light_icon_overlay(); !light_icons) {
        return std::unexpected(light_icons.error());
    }

    rollback_on_failure = false;
    initialized_ = true;

    debug("[Renderer::initialize] exit: success");

    return {};
}

auto Renderer::destroy() noexcept -> void {
    debug("[Renderer::destroy] enter");

    light_icon_overlay_.reset();

    screenshot_->close();

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

    // The frames' image targets are Holders, destroyed by frames_.clear() below, before image_storage_.
    for (auto &frame: frames_) {
        frame.lights_buffer.destroy();
        frame.frustum_planes_buffer.destroy();
        frame.visible_transform_buffer.destroy();
        frame.visible_draw_buffer.destroy();
        frame.culled_readback_buffer.destroy();
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
        frame.culled_readback_capacity = 0;
        frame.culled_readback_count = 0;
        frame.culled_readback_pending = false;
    }

    frames_.clear();

    shadow_atlas_.reset();

    shadow_cascade_cache_ = {};
    shadow_frame_ = 0;
    shadow_caster_revision_ = 1;
    cached_shadow_caster_revision_ = 0;
    shadow_scene_signature_ = 0;
    cached_shadow_light_direction_ = glm::vec3{0.0F};
    cached_shadow_depth_bias_constant_ = 0.0F;
    cached_shadow_depth_bias_slope_ = 0.0F;
    shadow_scene_signature_valid_ = false;
    shadow_global_state_valid_ = false;
    shadow_atlas_initialized_ = false;
    dynamic_shadow_casters_dirty_ = false;

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

    std::error_code canonicalize_error;
    auto const canonical_path = std::filesystem::weakly_canonical(path, canonicalize_error);
    auto const &cache_key_path = canonicalize_error ? path : canonical_path;
    std::size_t const file_hash = std::filesystem::hash_value(cache_key_path);

    if (auto it = model_cache_.find(file_hash); it != model_cache_.end()) {
        retain_model(it->second);
        return it->second;
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
    model_sources_.insert_or_assign(model_source_key(*model_result), cache_key_path);
    register_model_name(*model_result, path.filename().string());

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

auto Renderer::install_model(ModelHandle pending, Model const &model) -> std::expected<void, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto const *pending_slot = model_storage_.get(pending);

    if (pending_slot == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    auto const borrowed_from = pending_slot->borrowed_from;

    auto handle = create_model_common(model, default_material_handle_, [this, pending](ModelSlotData data) {
        return model_storage_.upgrade_pending_model(pending, std::move(data));
    });

    if (!handle) {
        return std::unexpected(handle.error());
    }

    // The slot owns its own meshes now, so it no longer needs the fallback it was drawing.
    if (borrowed_from.valid()) {
        static_cast<void>(destroy_model(borrowed_from));
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

    // Held until the model is installed, so every early return below destroys the meshes made so far.
    std::vector<MeshHolder> created_meshes;
    created_meshes.reserve(model.meshes.size());

    for (std::size_t mesh_index = 0; mesh_index < model.meshes.size(); ++mesh_index) {
        auto const &source_mesh = model.meshes[mesh_index];

        if (source_mesh.primitives.empty()) {
            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }

        std::vector<SubmeshCreateInfo> submesh_infos;

        submesh_infos.reserve(source_mesh.primitives.size());

        for (auto const &source_submesh: source_mesh.primitives) {
            auto const index = source_submesh.material_index;

            if (source_submesh.material_index >= model.materials.size()) {
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
            return std::unexpected(mesh.error());
        }

        imported_meshes[mesh_index] = *mesh;
        created_meshes.emplace_back(*this, *mesh);
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
            return std::unexpected(result.error());
        }
    }

    if (flattened_draws.empty()) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    auto handle = install(ModelSlotData{
            .draws = std::move(flattened_draws),
            .bounds_min = model.bounds_min,
            .bounds_max = model.bounds_max,
            .lights = model.lights,
    });

    if (!handle) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    // The model's draws own the meshes now; destroy_model() releases them.
    for (auto &mesh: created_meshes) {
        static_cast<void>(mesh.detach());
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

auto Renderer::model_submesh_bounds(ModelHandle model) const
        -> std::optional<std::vector<std::pair<glm::vec3, glm::vec3>>> {
    auto const *slot = model_slot(model);

    if (slot == nullptr) {
        return std::nullopt;
    }

    std::vector<std::pair<glm::vec3, glm::vec3>> bounds;

    for (auto const &draw: slot->draws) {
        auto const *mesh = mesh_slot(draw.mesh);

        if (mesh == nullptr) {
            continue;
        }

        for (auto const &submesh: mesh->submeshes) {
            bounds.push_back(maths::transform_aabb(draw.local_transform, submesh.bounds_min, submesh.bounds_max));
        }
    }

    return bounds;
}

auto Renderer::model_lights(ModelHandle model) const -> std::span<ModelCpuLight const> {
    auto const *slot = model_slot(model);

    if (slot == nullptr) {
        return {};
    }

    return slot->lights;
}

auto Renderer::model_materials(ModelHandle model) const -> std::vector<MaterialHandle> {
    auto const *slot = model_slot(model);

    if (slot == nullptr) {
        return {};
    }

    std::vector<MaterialHandle> materials;

    for (auto const &draw: slot->draws) {
        auto const *mesh = mesh_slot(draw.mesh);

        if (mesh == nullptr) {
            continue;
        }

        for (auto const &submesh: mesh->submeshes) {
            if (submesh.material.valid() && std::ranges::find(materials, submesh.material) == materials.end()) {
                materials.push_back(submesh.material);
            }
        }
    }

    return materials;
}

auto Renderer::submit_model(ModelHandle model, glm::mat4 const &transform, MaterialHandle material_override,
                            std::span<MaterialSlotOverride const> slot_overrides)
        -> std::expected<void, RendererError> {
    if (model_slot(model) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (model_submissions_.size() >= maximum_submission_count_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    auto const slot_override_first = static_cast<std::uint32_t>(slot_override_submissions_.size());
    slot_override_submissions_.insert(slot_override_submissions_.end(), slot_overrides.begin(), slot_overrides.end());

    model_submissions_.push_back(ModelSubmission{
            .model = model,
            .transform = transform,
            .material_override = material_override,
            .slot_override_first = slot_override_first,
            .slot_override_count = static_cast<std::uint32_t>(slot_overrides.size()),
    });

    return {};
}

auto Renderer::submit_model(ModelHandle model, glm::mat4 &&transform, MaterialHandle material_override,
                            std::span<MaterialSlotOverride const> slot_overrides)
        -> std::expected<void, RendererError> {
    if (model_slot(model) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (model_submissions_.size() >= maximum_submission_count_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    auto const slot_override_first = static_cast<std::uint32_t>(slot_override_submissions_.size());
    slot_override_submissions_.insert(slot_override_submissions_.end(), slot_overrides.begin(), slot_overrides.end());

    model_submissions_.push_back(ModelSubmission{
            .model = model,
            .transform = transform,
            .material_override = material_override,
            .slot_override_first = slot_override_first,
            .slot_override_count = static_cast<std::uint32_t>(slot_overrides.size()),
    });

    return {};
}

auto Renderer::submit_model_instances(ModelHandle model, std::span<glm::mat4 const> transforms,
                                      MaterialHandle material_override) -> std::expected<void, RendererError> {
    if (model_slot(model) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (model_submissions_.size() + transforms.size() > maximum_submission_count_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    model_submissions_.reserve(model_submissions_.size() + transforms.size());
    for (auto const &transform: transforms) {
        model_submissions_.push_back(ModelSubmission{
                .model = model,
                .transform = transform,
                .material_override = material_override,
        });
    }

    return {};
}

auto Renderer::create_material(MaterialCreateInfo const &create_info, std::string debug_name)
        -> std::expected<MaterialHandle, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto material = material_storage_.create_material(create_info);

    if (!material) {
        return std::unexpected(make_material_error(material.error()));
    }

    if (!debug_name.empty()) {
        // A name collision is fine; the material just isn't registered under that name.
        static_cast<void>(assets_.materials().register_asset(std::move(debug_name), *material));
    }

    return *material;
}

auto Renderer::duplicate_material(MaterialHandle source, std::string debug_name)
        -> std::expected<MaterialHandle, RendererError> {
    auto const *source_info = material_storage_.create_info(source);

    if (source_info == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_material));
    }

    return create_material(*source_info, std::move(debug_name));
}

auto Renderer::update_material(MaterialHandle handle, MaterialCreateInfo const &create_info)
        -> std::expected<void, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto result = material_storage_.update_material(handle, create_info);

    if (!result) {
        return std::unexpected(make_material_error(result.error()));
    }

    mark_shadow_casters_dirty();
    return {};
}

auto Renderer::retain_material(MaterialHandle handle) -> void {
    if (!initialized_ || !handle.valid()) {
        return;
    }

    if (!material_storage_.retain_material(handle)) {
        warn("Renderer::retain_material: handle is not a live material");
    }
}

auto Renderer::release_material(MaterialHandle handle) -> void {
    // Scenes outlive destroy(), which already freed every material.
    if (!initialized_ || !handle.valid() || handle == default_material_handle_) {
        return;
    }

    bool const last_reference = material_storage_.ref_count(handle) == 1;

    if (auto const result = material_storage_.destroy_material(handle); !result) {
        warn("Renderer::release_material: handle is not a live material");
        return;
    }

    if (last_reference) {
        assets_.materials().unregister(handle);
        mark_shadow_casters_dirty();
    }
}

auto Renderer::destroy_material(MaterialHandle handle) -> std::expected<void, RendererError> {
    if (handle == default_material_handle_) {
        return std::unexpected(make_error(RendererErrorType::invalid_material));
    }

    if (material_storage_.get(handle) == nullptr) {
        return std::unexpected(make_material_error(MaterialStorageError{
                .type = MaterialStorageErrorType::invalid_handle,
        }));
    }

    assets_.materials().unregister(handle);
    release_material(handle);

    return {};
}

auto Renderer::register_material_name(MaterialHandle handle, std::string name) -> bool {
    if (material_storage_.get(handle) == nullptr || !assets_.materials().name_of(handle).empty()) {
        return false;
    }

    if (!assets_.materials().register_asset(std::move(name), handle)) {
        return false;
    }

    retain_material(handle);
    return true;
}

auto Renderer::request_texture(std::filesystem::path source_path, TextureRole role, ImageHandle fallback,
                               std::string debug_name) -> ImageHandle {
    // The handle is stable across the pending-to-loaded upgrade, so it can be named right away.
    auto const handle = texture_streamer_.request(image_storage_, std::move(source_path), role, fallback, debug_name);

    static_cast<void>(assets_.textures().register_asset(std::move(debug_name), handle));

    return handle;
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

        // Every LOD is drawn through task/mesh shaders and needs meshlets.
        if (!std::ranges::all_of(submesh_info.lods, [](MeshGeometry const &lod) { return lod.meshlets.valid(); })) {
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

    mark_shadow_casters_dirty();
    return *handle;
}

namespace {

    // LODs share the vertex slice and may alias an earlier level's indices/meshlets. Retire each distinct range
    // once, or GeometryArena's free-list gets the same range twice.
    auto retire_submesh_geometry(GeometryArena &geometry_arena, Submesh const &submesh) -> void {
        std::array<VkDeviceSize, std::size_t{lod_count} * 4> retired_offsets{};
        std::size_t retired_count = 0;

        auto retire_once = [&](GeometrySlice const &slice) {
            if (!slice.valid()) {
                return;
            }

            for (std::size_t i = 0; i < retired_count; ++i) {
                if (retired_offsets[i] == slice.offset) {
                    return;
                }
            }

            geometry_arena.retire(slice);
            retired_offsets[retired_count++] = slice.offset;
        };

        for (auto const &lod: submesh.lods) {
            retire_once(lod.vertices.bytes);
            retire_once(lod.indices.bytes);
            retire_once(lod.meshlets.descriptors);
            retire_once(lod.meshlets.data);
        }
    }

} // namespace

auto Renderer::destroy_mesh(MeshHandle handle) -> std::expected<void, RendererError> {
    if (auto const *slot = mesh_storage_.get(handle)) {
        for (auto const &submesh: slot->submeshes) {
            retire_submesh_geometry(geometry_arena_, submesh);
        }
    }

    auto result = mesh_storage_.destroy_mesh(handle);

    if (!result) {
        return std::unexpected(make_error(RendererErrorType::invalid_mesh));
    }

    mark_shadow_casters_dirty();
    return {};
}

auto Renderer::retain_model(ModelHandle handle) -> void {
    auto *slot = model_storage_.get(handle);

    if (slot == nullptr) {
        warn("Renderer::retain_model: handle is not a live model");
        return;
    }

    ++slot->ref_count;
}

auto Renderer::release_model(ModelHandle handle) -> void {
    // Scenes outlive destroy(), which already freed every model.
    if (!initialized_) {
        return;
    }

    static_cast<void>(destroy_model(handle));
}

auto Renderer::register_model_name(ModelHandle handle, std::string_view name) -> void {
    static_cast<void>(assets_.models().register_asset(std::string{name}, handle));
}

auto Renderer::register_model_source(ModelHandle handle, std::filesystem::path const &source) -> void {
    if (model_storage_.get(handle) == nullptr) {
        return;
    }

    std::error_code canonicalize_error;
    auto const canonical_path = std::filesystem::weakly_canonical(source, canonicalize_error);
    auto const &cache_key_path = canonicalize_error ? source : canonical_path;

    model_cache_.try_emplace(std::filesystem::hash_value(cache_key_path), handle);
    model_sources_.insert_or_assign(model_source_key(handle), cache_key_path);
}

auto Renderer::cached_model(std::filesystem::path const &source) const -> ModelHandle {
    std::error_code canonicalize_error;
    auto const canonical_path = std::filesystem::weakly_canonical(source, canonicalize_error);
    auto const &cache_key_path = canonicalize_error ? source : canonical_path;

    auto const it = model_cache_.find(std::filesystem::hash_value(cache_key_path));
    return it != model_cache_.end() && model_storage_.get(it->second) != nullptr ? it->second : ModelHandle{};
}

auto Renderer::model_source(ModelHandle handle) const noexcept -> std::filesystem::path const * {
    auto const it = model_sources_.find(model_source_key(handle));
    return it != model_sources_.end() ? &it->second : nullptr;
}

auto Renderer::destroy_model(ModelHandle handle) -> std::expected<void, RendererError> {
    auto *slot = model_storage_.get(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (slot->ref_count > 1) {
        --slot->ref_count;
        return {};
    }

    // A pending slot's draws are its fallback's meshes; only the reference on the fallback is its to drop.
    auto const borrowed_from = slot->borrowed_from;

    if (!borrowed_from.valid()) {
        for (auto const &draw: slot->draws) {
            static_cast<void>(destroy_mesh(draw.mesh));
        }
    }

    model_streamer_.forget(handle);
    std::erase_if(model_cache_, [handle](auto const &entry) { return entry.second == handle; });
    model_sources_.erase(model_source_key(handle));
    assets_.models().unregister(handle);

    if (auto released = model_storage_.release(handle); !released) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (borrowed_from.valid()) {
        static_cast<void>(destroy_model(borrowed_from));
    }

    mark_shadow_casters_dirty();
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

    vkCmdResetQueryPool(command_buffer, frame_query.query_pool, 0, total_query_count);

    auto &frame_pipeline_query = pipeline_stat_queries_[frame_index];

    vkCmdResetQueryPool(command_buffer, frame_pipeline_query.query_pool, 0, 1);

    vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame_query.query_pool,
                         static_cast<std::uint32_t>(RenderStage::FullFrame) * 2);

    pipeline_graph_.tick_retirement();
    geometry_arena_.tick_retirement();

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

    frame.view_projection = matrices.projection * matrices.view;

    frame.draws.clear();
    frame.transforms.clear();
    frame.indirect_commands.clear();
    frame.batch_bounds.clear();

    frame.indirect_command_count = 0;
    frame.opaque_indirect_count = 0;
    frame.mask_indirect_count = 0;
    frame.blend_indirect_count = 0;
    frame.shadow_update_mask = 0;

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

        // Destroyed after it was submitted this frame, e.g. by the editor swapping an entity's model.
        if (model == nullptr) {
            continue;
        }

        auto const slot_overrides = std::span{slot_override_submissions_}.subspan(
                model_submission.slot_override_first, model_submission.slot_override_count);

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
                auto material = model_submission.material_override.valid() ? model_submission.material_override
                                                                           : submesh.material;

                for (auto const &slot_override: slot_overrides) {
                    if (slot_override.source == submesh.material && slot_override.material.valid()) {
                        material = slot_override.material;
                        break;
                    }
                }
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

        if (!geometry.meshlets.valid()) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }

        frame.transforms.insert(frame.transforms.end(), batch.transforms.begin(), batch.transforms.end());

        auto const first_instance = static_cast<std::uint32_t>(frame.draws.size());
        auto const vertex_address = geometry_arena_.vertex_address(geometry.vertices);
        auto const meshlet_address = geometry_arena_.device_address(geometry.meshlets.descriptors);
        auto const meshlet_data_address = geometry_arena_.device_address(geometry.meshlets.data);
        auto const material_index = material_storage_.gpu_index(batch.material);
        for (std::uint32_t instance = 0; instance < instance_count; ++instance) {
            frame.draws.push_back(GpuDraw{
                    .vertex_address = vertex_address,
                    .meshlet_address = meshlet_address,
                    .meshlet_data_address = meshlet_data_address,
                    .material_index = material_index,
                    .transform_index = base_transform_index + instance,
            });
        }

        // Un-culled command: drawn as-is by the shadow pass and culled by main_cs for the main view. Exactly one of its
        // halves is live, see uses_meshlet_path().
        GpuDrawCommand command{
                .instance_count = instance_count,
                .first_instance = first_instance,
        };

        if (uses_meshlet_path(geometry.meshlets.meshlet_count)) {
            command.meshlet_count = geometry.meshlets.meshlet_count;
        } else {
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

            command.index_count = geometry.indices.index_count;
            command.first_index = static_cast<std::uint32_t>(first_index_u64);
        }

        set_task_group_counts(command);
        frame.indirect_commands.push_back(command);

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

    constexpr std::size_t parallel_blend_sort_threshold = 4'096;

    auto sort_blend_batches = [this] {
        std::ranges::sort(blend_batches_, [](PendingBlendBatch const &lhs, PendingBlendBatch const &rhs) {
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

    // Transforms are left out to keep this cheap; moving casters use mark_dynamic_shadow_casters_dirty().
    std::uint64_t current_shadow_scene_signature = 0;
    std::uint64_t shadow_caster_batch_count = 0;
    bool has_animated_shadow_casters = false;
    for (auto const *batch: active_batches_) {
        auto const *material = material_storage_.get(batch->material);
        auto const alpha_mode = material != nullptr ? material->alpha_mode : AlphaMode::opaque;
        auto const max_cascade = batch_max_shadow_cascade(batch);

        if (alpha_mode == AlphaMode::blend || max_cascade < 0) {
            continue;
        }

        std::uint64_t batch_signature = 0xcbf29ce484222325ULL;
        batch_signature = shadow_signature_combine(batch_signature, batch->mesh.index);
        batch_signature = shadow_signature_combine(batch_signature, batch->submesh_index);
        batch_signature = shadow_signature_combine(batch_signature, material_storage_.gpu_index(batch->material));
        batch_signature = shadow_signature_combine(batch_signature, batch->lod_index);
        batch_signature =
                shadow_signature_combine(batch_signature, static_cast<std::uint64_t>(batch->transforms.size()));
        batch_signature = shadow_signature_combine(batch_signature, static_cast<std::uint64_t>(max_cascade));

        // XOR makes the result independent of batch order. The count is mixed in separately so duplicates don't
        // cancel.
        current_shadow_scene_signature ^= shadow_signature_mix(batch_signature);
        ++shadow_caster_batch_count;

        has_animated_shadow_casters =
                has_animated_shadow_casters || (material != nullptr && std::abs(material->wind_strength) > 1e-6F);
    }
    current_shadow_scene_signature =
            shadow_signature_combine(current_shadow_scene_signature, shadow_caster_batch_count);

    // There are only shadow_cascade_count + 1 keys (max cascade 3..0, or no shadows), so repeated in-place
    // partitions replace a sort. The partition boundaries are the per-cascade indirect prefix counts.
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
    auto const light_direction = glm::normalize(light_.direction);
    auto const candidate_cascades = fit_shadow_cascades(ShadowCascadeFitInput{
            .camera_view = view,
            .camera_near = matrices.near_clip,
            .camera_far = matrices.far_clip,
            .vertical_fov_radians = matrices.vertical_fov_radians,
            .aspect_ratio = matrices.aspect_ratio,
            .light_direction = light_direction,
            .settings = shadow_settings_.cascades,
    });

    ++shadow_frame_;
    if (shadow_frame_ == 0) {
        shadow_frame_ = 1;
        for (auto &cached: shadow_cascade_cache_) {
            cached.valid = false;
        }
        shadow_atlas_initialized_ = false;
    }

    bool projection_shape_changed = false;
    for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
        auto const &cached = shadow_cascade_cache_[cascade];
        if (!cached.valid || !nearly_equal(candidate_cascades.split_far[cascade], cached.split_far) ||
            !nearly_equal(candidate_cascades.texel_world[cascade], cached.texel_world) ||
            !nearly_equal(candidate_cascades.depth_scale[cascade], cached.depth_scale)) {
            projection_shape_changed = true;
            break;
        }
    }

    auto const shadow_global_state_changed =
            !shadow_global_state_valid_ || !nearly_equal(light_direction, cached_shadow_light_direction_) ||
            !nearly_equal(shadow_settings_.depth_bias_constant, cached_shadow_depth_bias_constant_) ||
            !nearly_equal(shadow_settings_.depth_bias_slope, cached_shadow_depth_bias_slope_);
    auto const shadow_scene_changed =
            !shadow_scene_signature_valid_ || current_shadow_scene_signature != shadow_scene_signature_;
    auto const shadow_casters_changed = cached_shadow_caster_revision_ != shadow_caster_revision_;
    auto const force_all_cascades = !shadow_settings_.cache_enabled || !shadow_atlas_initialized_ ||
                                    shadow_global_state_changed || shadow_scene_changed || shadow_casters_changed ||
                                    projection_shape_changed;

    auto resolved_view_projection = candidate_cascades.view_projection;
    auto resolved_split_far = candidate_cascades.split_far;
    auto resolved_texel_world = candidate_cascades.texel_world;
    auto resolved_depth_scale = candidate_cascades.depth_scale;
    frame.pending_shadow_cache = shadow_cascade_cache_;

    for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
        auto const &cached = shadow_cascade_cache_[cascade];
        auto &pending = frame.pending_shadow_cache[cascade];
        auto const bit = ShadowCascadeMask{1} << cascade;
        auto const period = std::max(shadow_settings_.cache_update_periods[cascade], 1U);
        auto const age = cached.valid ? shadow_frame_ - cached.last_update_frame : std::uint64_t{period};
        auto const period_due = age >= period;
        auto const matrix_changed =
                !cached.valid || !nearly_equal(candidate_cascades.view_projection[cascade], cached.view_projection);

        // Cascade 0 is always fresh. Far cascades update when their snapped matrix changes or casters move, once
        // their minimum interval has elapsed.
        auto const dynamic_due = (dynamic_shadow_casters_dirty_ || has_animated_shadow_casters) && period_due;
        auto const update = force_all_cascades || cascade == 0U || (matrix_changed && period_due) || dynamic_due;

        if (update) {
            frame.shadow_update_mask |= bit;
            pending.view_projection = candidate_cascades.view_projection[cascade];
            pending.split_far = candidate_cascades.split_far[cascade];
            pending.texel_world = candidate_cascades.texel_world[cascade];
            pending.depth_scale = candidate_cascades.depth_scale[cascade];
            pending.last_update_frame = shadow_frame_;
            pending.valid = true;
        } else {
            // Never sample a cached tile with a matrix that did not create it.
            resolved_view_projection[cascade] = cached.view_projection;
            resolved_split_far[cascade] = cached.split_far;
            resolved_texel_world[cascade] = cached.texel_world;
            resolved_depth_scale[cascade] = cached.depth_scale;
        }
    }

    // Committed only after record_frame() records the tile updates, so a failed shadow pass leaves the cache
    // metadata untouched.
    frame.pending_shadow_light_direction = light_direction;
    frame.pending_shadow_depth_bias_constant = shadow_settings_.depth_bias_constant;
    frame.pending_shadow_depth_bias_slope = shadow_settings_.depth_bias_slope;
    frame.pending_shadow_caster_revision = shadow_caster_revision_;
    frame.pending_shadow_scene_signature = current_shadow_scene_signature;
    dynamic_shadow_casters_dirty_ = false;

    auto const view_projection = projection * view;

    auto const frustum_planes = extract_frustum_planes(view_projection);

    UBO const ubo{
            .view_projection = view_projection,
            .view = view,
            .projection = projection,
            .inverse_projection = glm::inverse(projection),
            .camera_position = camera_position,
            .fog_colour = fog_settings_.colour,
            .fog_extinction = fog_settings_.enabled ? fog_settings_.extinction : 0.0F,
            .fog_inscattering = fog_settings_.inscattering,
            .cascade_view_projection = resolved_view_projection,
            .cascade_split_far = glm::make_vec4(resolved_split_far.data()),
            .cascade_texel_world = glm::make_vec4(resolved_texel_world.data()),
            .cascade_depth_scale = glm::make_vec4(resolved_depth_scale.data()),
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
            .shadow_atlas_texture = shadow_atlas_.handle().index,
            .shadow_sampler = sampler_storage_.shadow_compare().index,
            .shadow_depth_bias_world = shadow_settings_.depth_bias_world,
            .shadow_pcf_radius_texels = shadow_settings_.pcf_radius_texels,
            .cascade_count = shadow_cascade_count,
            .shadow_atlas_texel_u = 1.0F / static_cast<float>(shadow_atlas_width),
            .shadow_atlas_texel_v = 1.0F / static_cast<float>(shadow_atlas_height),
            .shadow_debug_cascade_tint = shadow_settings_.debug_cascade_tint ? 1U : 0U,
            .time = matrices.time,
            .ambient_intensity = ambient_intensity_,
            .ao_intensity = ao_settings_.enabled ? ao_settings_.intensity : 0.0F,
    };

    if (!ubos_[frame_index].write(0, std::span{&ubo, 1})) {
        clear_submissions();
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    // Cascade planes come from the matrices the shadow pass renders with, so a reused tile culls against the
    // frustum it was drawn with.
    std::array<glm::vec4, cull_plane_count> cull_planes{};
    std::ranges::copy(frustum_planes, cull_planes.begin());

    for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
        std::ranges::copy(extract_frustum_planes(resolved_view_projection[cascade]),
                          cull_planes.begin() + (static_cast<std::ptrdiff_t>(1 + cascade) * 6));
    }

    if (!frame.frustum_planes_buffer.write(0, std::span{cull_planes})) {
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

            std::array<VkBufferMemoryBarrier2, 4> const post_cull_barriers{
                    VkBufferMemoryBarrier2{
                            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
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
                            .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                            VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
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
                            // Read as the indirect command and as the task shader's per-batch payload.
                            .dstStageMask =
                                    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
                            .dstAccessMask =
                                    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .buffer = frame.culled_indirect_buffer.buffer,
                            .offset = 0,
                            .size = VK_WHOLE_SIZE,
                    },
                    // Also read by the visible-instance readback copy.
                    VkBufferMemoryBarrier2{
                            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
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

            auto const readback_size = static_cast<VkDeviceSize>(frame.indirect_command_count) * sizeof(GpuDrawCommand);

            if (!frame.culled_readback_buffer.valid() ||
                frame.culled_readback_capacity < frame.indirect_command_count) {
                auto const capacity = std::bit_ceil(std::max(frame.indirect_command_count, 1U));

                auto readback = Buffer::create(
                        context_, BufferCreateInfo{
                                          .size = static_cast<VkDeviceSize>(capacity) * sizeof(GpuDrawCommand),
                                          .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          .memory = BufferMemory::readback,
                                          .debug_name = "renderer.culled_readback",
                                  });

                if (!readback) {
                    clear_submissions();
                    return std::unexpected(make_device_error(readback.error()));
                }

                frame.culled_readback_buffer = std::move(*readback);
                frame.culled_readback_capacity = capacity;
            }

            VkBufferCopy2 const readback_region{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                    .srcOffset = 0,
                    .dstOffset = 0,
                    .size = readback_size,
            };

            VkCopyBufferInfo2 const readback_copy{
                    .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    .srcBuffer = frame.culled_indirect_buffer.buffer,
                    .dstBuffer = frame.culled_readback_buffer.buffer,
                    .regionCount = 1,
                    .pRegions = &readback_region,
            };

            vkCmdCopyBuffer2(command_buffer, &readback_copy);

            VkBufferMemoryBarrier2 const readback_to_host{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                    .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .buffer = frame.culled_readback_buffer.buffer,
                    .offset = 0,
                    .size = readback_size,
            };

            VkDependencyInfo const readback_dependency_info{
                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .bufferMemoryBarrierCount = 1,
                    .pBufferMemoryBarriers = &readback_to_host,
            };

            vkCmdPipelineBarrier2(command_buffer, &readback_dependency_info);

            frame.culled_readback_count = frame.indirect_command_count;
            frame.culled_readback_pending = true;
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
                auto const start = results[static_cast<std::size_t>(i) * 2];
                auto const end = results[(static_cast<std::size_t>(i) * 2) + 1];
                last_frame_timings_.milliseconds[i] =
                        static_cast<float>(end - start) * timestamp_period_ / 1'000'000.0F;
            }

            read_overlay_timings(frame_query);
            last_frame_timings_.valid = true;
        }

        frame_query.has_results = false;
    }

    if (frame_pipeline_query.has_results) {
        last_frame_pipeline_stats_.valid = false;
        std::array<std::uint64_t, pipeline_stat_count> results{};

        auto const mesh_queries = context_.mesh_shader_queries_supported;
        auto const result_count = mesh_queries ? 3U : 2U;
        auto const result_size = static_cast<std::size_t>(result_count) * sizeof(std::uint64_t);

        auto const query_result =
                vkGetQueryPoolResults(context_.device, frame_pipeline_query.query_pool, 0, 1, result_size,
                                      results.data(), result_size, VK_QUERY_RESULT_64_BIT);

        if (query_result == VK_SUCCESS) {
            last_frame_pipeline_stats_ = PipelineStats{};
            if (mesh_queries) {
                last_frame_pipeline_stats_.fragment_shader_invocation_count = results[0];
                last_frame_pipeline_stats_.task_shader_invocation_count = results[1];
                last_frame_pipeline_stats_.mesh_shader_invocation_count = results[2];
            } else {
                last_frame_pipeline_stats_.clipped_primitive_count = results[0];
                last_frame_pipeline_stats_.fragment_shader_invocation_count = results[1];
            }
            last_frame_pipeline_stats_.mesh_stats_valid = mesh_queries;
            last_frame_pipeline_stats_.valid = true;
        }

        frame_pipeline_query.has_results = false;
    }

    return {};
}

auto Renderer::consume_culled_readback(RendererFrame &frame) -> void {
    // This frame slot's fence has been waited on, so last use's readback copy has completed.
    if (!frame.culled_readback_pending) {
        return;
    }

    frame.culled_readback_pending = false;

    if (auto invalidated = frame.culled_readback_buffer.invalidate(
                0, static_cast<VkDeviceSize>(frame.culled_readback_count) * sizeof(GpuDrawCommand));
        !invalidated) {
        error("[Renderer] Failed to invalidate culled-indirect readback buffer");
        return;
    }

    auto const *commands = frame.culled_readback_buffer.mapped_data_as<GpuDrawCommand const>();
    if (commands == nullptr) {
        return;
    }

    std::uint32_t visible_instance_count = 0;

    for (std::uint32_t i = 0; i < frame.culled_readback_count; ++i) {
        visible_instance_count += commands[i].instance_count;
    }

    last_frame_stats_.visible_instance_count = visible_instance_count;
}

auto Renderer::resolve_frame_targets(RendererFrame const &frame) const -> std::expected<FrameTargets, RendererError> {
    bool const multisampled = frame.forward_target.is_multisampled();

    auto const hdr_handle = frame.forward_target.hdr();
    auto const depth_handle = frame.forward_target.depth();
    auto const resolved_hdr_handle = multisampled ? frame.forward_target.resolved_hdr() : hdr_handle;
    auto const resolved_depth_handle = multisampled ? frame.forward_target.resolved_depth() : depth_handle;

    FrameTargets const targets{
            .hdr = image_storage_.get(hdr_handle),
            .depth = image_storage_.get(depth_handle),
            .resolved_hdr = image_storage_.get(resolved_hdr_handle),
            .resolved_depth = image_storage_.get(resolved_depth_handle),
            .resolved_hdr_handle = resolved_hdr_handle,
            .resolved_depth_handle = resolved_depth_handle,
            .shadow_atlas = shadow_atlas_.get(),
            .ao_raw = frame.ao_target.raw.get(),
            .ao_denoised = frame.ao_target.denoised.get(),
            .viewport = frame.viewport_target.get(),
            .extent = frame.forward_target.extent(),
            .multisampled = multisampled,
    };

    auto const usable = [](Image const *image) { return image != nullptr && image->valid(); };

    if (!usable(targets.hdr) || !usable(targets.depth) || !usable(targets.resolved_hdr) ||
        !usable(targets.resolved_depth) || !usable(targets.shadow_atlas) || !usable(targets.ao_raw) ||
        !usable(targets.ao_denoised) || !usable(targets.viewport)) {
        return std::unexpected(make_error(RendererErrorType::image_error));
    }

    auto const matches_extent = [&](Image const &image) {
        return image.extent_2d().width == targets.extent.width && image.extent_2d().height == targets.extent.height;
    };

    if (targets.extent.width == 0 || targets.extent.height == 0 || !matches_extent(*targets.hdr) ||
        !matches_extent(*targets.depth)) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    return targets;
}

auto Renderer::main_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers {
    return render_pass::DrawBuffers{
            .draws = frame.visible_draw_buffer,
            .transforms = frame.visible_transform_buffer,
            .indirect = frame.culled_indirect_buffer,
            .index_buffer = geometry_arena_.bindable_buffer(),
    };
}

auto Renderer::batch_counts(RendererFrame const &frame) noexcept -> render_pass::DrawCounts {
    return render_pass::DrawCounts{
            .opaque = frame.opaque_indirect_count,
            .mask = frame.mask_indirect_count,
            .blend = frame.blend_indirect_count,
    };
}

auto Renderer::record_shadow_pass(render_pass::Context const &pass_context, RendererFrame const &frame,
                                  FrameTargets const &targets) -> std::expected<void, RendererError> {
    TracyVkZoneC(context_.host_query_context.context, pass_context.command_buffer, "Shadow Pass", tracy::Color::Purple);

    auto const frame_index = pass_context.frame_index;

    // Shadows draw every caster, so this uses the un-culled buffers.
    auto const result = render_pass::shadow(
            pass_context,
            render_pass::ShadowPassInfo{
                    .shadow_atlas = *targets.shadow_atlas,
                    .draws =
                            {
                                    .draws = frame.draw_buffer,
                                    .transforms = frame.transform_buffer,
                                    .indirect = frame.indirect_buffer,
                                    .index_buffer = geometry_arena_.bindable_buffer(),
                            },
                    .counts = batch_counts(frame),
                    .opaque_cascade_counts = frame.shadow_opaque_indirect_count,
                    .mask_cascade_counts = frame.shadow_mask_indirect_count,
                    .update_mask = frame.shadow_update_mask,
                    .preserve_contents = shadow_atlas_initialized_,
                    .meshlet_culling = meshlet_culling_,
                    .cascade_cull_planes_address = frame.frustum_planes_buffer.device_address + 6 * sizeof(glm::vec4),
                    .materials_address = material_storage_.device_address(),
                    .ubo_address = ubos_[frame_index].device_address,
                    .lights_address = frame.lights_buffer.device_address,
                    .opaque_pipeline = shadow_pipeline_,
                    .mask_pipeline = shadow_mask_pipeline_,
                    .opaque_instanced_pipeline = shadow_instanced_pipeline_,
                    .mask_instanced_pipeline = shadow_mask_instanced_pipeline_,
                    .depth_bias_constant = shadow_settings_.depth_bias_constant,
                    .depth_bias_slope = shadow_settings_.depth_bias_slope,
            });

    if (!result) {
        return std::unexpected(result.error());
    }

    if (frame.shadow_update_mask == 0) {
        return {};
    }

    for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
        if ((frame.shadow_update_mask & (ShadowCascadeMask{1} << cascade)) != 0) {
            shadow_cascade_cache_[cascade] = frame.pending_shadow_cache[cascade];
        }
    }

    cached_shadow_light_direction_ = frame.pending_shadow_light_direction;
    cached_shadow_depth_bias_constant_ = frame.pending_shadow_depth_bias_constant;
    cached_shadow_depth_bias_slope_ = frame.pending_shadow_depth_bias_slope;
    cached_shadow_caster_revision_ = frame.pending_shadow_caster_revision;
    shadow_scene_signature_ = frame.pending_shadow_scene_signature;
    shadow_scene_signature_valid_ = true;
    shadow_global_state_valid_ = true;
    shadow_atlas_initialized_ = true;

    return {};
}

auto Renderer::record_depth_prepass(render_pass::Context const &pass_context, RendererFrame const &frame,
                                    FrameTargets const &targets) -> std::expected<void, RendererError> {
    render_pass::prepare_forward_targets(
            pass_context, render_pass::ForwardTargets{
                                  .hdr = *targets.hdr,
                                  .depth = *targets.depth,
                                  .resolved_hdr = targets.multisampled ? targets.resolved_hdr : nullptr,
                                  .resolved_depth = targets.multisampled ? targets.resolved_depth : nullptr,
                          });

    TracyVkZoneC(context_.host_query_context.context, pass_context.command_buffer, "Depth Prepass",
                 tracy::Color::SlateGray);

    auto const frame_index = pass_context.frame_index;

    return render_pass::depth_prepass(pass_context,
                                      render_pass::DepthPrepassInfo{
                                              .depth = *targets.depth,
                                              .resolved_depth = targets.multisampled ? targets.resolved_depth : nullptr,
                                              .extent = targets.extent,
                                              .samples = samples_,
                                              .draws = main_view_draws(frame),
                                              .counts = batch_counts(frame),
                                              .cull_planes_address = frame.frustum_planes_buffer.device_address,
                                              .materials_address = material_storage_.device_address(),
                                              .ubo_address = ubos_[frame_index].device_address,
                                              .lights_address = frame.lights_buffer.device_address,
                                              .opaque_pipeline = depth_prepass_pipeline_,
                                              .mask_pipeline = depth_prepass_mask_pipeline_,
                                              .opaque_instanced_pipeline = depth_prepass_instanced_pipeline_,
                                              .mask_instanced_pipeline = depth_prepass_mask_instanced_pipeline_,
                                              .meshlet_culling = meshlet_culling_,
                                      });
}

auto Renderer::record_ambient_occlusion_pass(render_pass::Context const &pass_context, RendererFrame const &frame,
                                             FrameTargets const &targets)
        -> std::expected<std::uint32_t, RendererError> {
    TracyVkZoneC(context_.host_query_context.context, pass_context.command_buffer, "Ambient Occlusion",
                 tracy::Color::DarkSlateGray);

    auto const ao_output = render_pass::ambient_occlusion(
            pass_context, render_pass::AmbientOcclusionInfo{
                                  .enabled = ao_settings_.enabled,
                                  .depth = *targets.resolved_depth,
                                  .raw_ao = *targets.ao_raw,
                                  .denoised_ao = *targets.ao_denoised,
                                  .extent = targets.extent,
                                  .depth_texture_index = targets.resolved_depth_handle.index,
                                  .raw_ao_texture_index = frame.ao_target.raw.handle().index,
                                  .denoised_ao_texture_index = frame.ao_target.denoised.handle().index,
                                  .point_sampler_index = sampler_storage_.nearest_clamp().index,
                                  .ubo_address = ubos_[pass_context.frame_index].device_address,
                                  .gtao_pipeline = gtao_pipeline_,
                                  .denoise_pipeline = gtao_denoise_pipeline_,
                                  .radius_view = ao_settings_.radius,
                                  .falloff_range = ao_settings_.falloff_range,
                                  .slice_count = ao_settings_.slice_count,
                                  .step_count = ao_settings_.step_count,
                                  .denoise_depth_sigma = ao_settings_.denoise_depth_sigma,
                          });

    if (!ao_output) {
        return std::unexpected(ao_output.error());
    }

    return ao_output->has_value() ? (*ao_output)->index : image_storage_.white().index;
}

auto Renderer::record_forward_pass(render_pass::Context const &pass_context, RendererFrame const &frame,
                                   FrameTargets const &targets, std::uint32_t ao_texture_index,
                                   render_pass::Callback scene_overlays)
        -> std::expected<render_pass::HdrTextureIndex, RendererError> {
    TracyVkZoneC(context_.host_query_context.context, pass_context.command_buffer, "Forward Pass",
                 tracy::Color::RoyalBlue);

    auto const frame_index = pass_context.frame_index;

    return render_pass::forward_geometry(
            pass_context,
            render_pass::ForwardGeometryInfo{
                    .hdr = *targets.hdr,
                    .depth = *targets.depth,
                    .resolved_hdr = targets.multisampled ? targets.resolved_hdr : nullptr,
                    .output_hdr = {.index = targets.resolved_hdr_handle.index},
                    .extent = targets.extent,
                    .samples = samples_,
                    .draws = main_view_draws(frame),
                    .counts = batch_counts(frame),
                    .cull_planes_address = frame.frustum_planes_buffer.device_address,
                    .materials_address = material_storage_.device_address(),
                    .ubo_address = ubos_[frame_index].device_address,
                    .lights_address = frame.lights_buffer.device_address,
                    .light_count = frame.light_count,
                    .pipeline_statistics_query_pool = pipeline_stat_queries_[frame_index].query_pool,
                    .meshlet_culling = meshlet_culling_,
                    .opaque_pipeline = forward_pipeline_,
                    .blend_pipeline = forward_blend_pipeline_,
                    .opaque_instanced_pipeline = forward_instanced_pipeline_,
                    .blend_instanced_pipeline = forward_blend_instanced_pipeline_,
                    .ao_texture_index = ao_texture_index,
                    .ao_sampler_index = sampler_storage_.linear_clamp().index,
            },
            scene_overlays);
}

auto Renderer::record_bloom_pass(render_pass::Context const &pass_context, RendererFrame const &frame,
                                 FrameTargets const &targets, render_pass::HdrTextureIndex hdr)
        -> std::expected<std::optional<render_pass::BloomTextureIndex>, RendererError> {
    TracyVkZoneC(context_.host_query_context.context, pass_context.command_buffer, "Bloom Pass", tracy::Color::Orange);

    std::array<std::uint32_t, render_pass::bloom_mip_count> mip_texture_indices{};
    for (std::uint32_t mip = 0; mip < render_pass::bloom_mip_count; ++mip) {
        mip_texture_indices[mip] = frame.bloom_target.mip_slots[mip].handle().index;
    }

    return render_pass::bloom(
            pass_context,
            render_pass::BloomPassInfo{
                    .enabled = bloom_settings_.enabled,
                    .input_hdr = hdr,
                    .target = bloom_settings_.enabled ? frame.bloom_target.image.get() : nullptr,
                    .mip_texture_indices = mip_texture_indices,
                    .input_extent = targets.extent,
                    .downsample_pipeline = bloom_downsample_pipeline_,
                    .upsample_pipeline = bloom_upsample_pipeline_,
                    .linear_sampler_index = sampler_storage_.linear_clamp().index,
                    .threshold = bloom_settings_.threshold,
                    .knee = bloom_settings_.knee,
                    .filter_radius = bloom_settings_.filter_radius,
            });
}

auto Renderer::record_composite_pass(render_pass::Context const &pass_context, FrameTargets const &targets,
                                     SwapchainImage const &swapchain_image, render_pass::HdrTextureIndex hdr,
                                     std::optional<render_pass::BloomTextureIndex> bloom, CompositeTarget target,
                                     render_pass::Callback ui_overlays) -> std::expected<void, RendererError> {
    TracyVkZoneC(context_.host_query_context.context, pass_context.command_buffer, "Composition",
                 tracy::Color::SeaGreen);

    // Fullscreen play composites straight into the swapchain with the UI on top. Otherwise the scene goes into the
    // viewport target the editor's Viewport panel samples, and a second pass draws the UI onto the swapchain.
    bool const fullscreen = target == CompositeTarget::swapchain;

    auto const result = render_pass::composite(
            pass_context,
            render_pass::CompositePassInfo{
                    .swapchain_image = fullscreen ? swapchain_image.image : targets.viewport->image(),
                    .swapchain_view = fullscreen ? swapchain_image.view : targets.viewport->view(),
                    .extent = fullscreen ? swapchain_image.extent : targets.extent,
                    .hdr = hdr,
                    .bloom = bloom,
                    .bloom_fallback_texture_index = image_storage_.emissive().index,
                    .linear_sampler_index = sampler_storage_.linear_clamp().index,
                    .pipeline = composite_pipeline_,
                    .exposure = 1.0F,
                    .bloom_intensity = bloom_settings_.intensity,
            },
            fullscreen ? ui_overlays : render_pass::Callback{});

    if (!result) {
        return std::unexpected(result.error());
    }

    if (fullscreen) {
        return {};
    }

    render_pass::transition_to_shader_read(pass_context.command_buffer, *targets.viewport);

    render_pass::ui_only(pass_context,
                         render_pass::UiOnlyPassInfo{
                                 .target_image = swapchain_image.image,
                                 .target_view = swapchain_image.view,
                                 .extent = swapchain_image.extent,
                         },
                         ui_overlays);

    return {};
}

auto Renderer::record_frame_end(VkCommandBuffer command_buffer, SwapchainImage const &swapchain_image,
                                Image const *viewport, std::uint32_t frame_index) -> void {
    auto const pending = screenshot_->pending_source();

    // The viewport target is left sampled by the UI pass; hand it back the same way so the next frame is unaffected.
    if (pending == ScreenshotSource::viewport && viewport != nullptr) {
        // The swapchain is presented as usual; the capture never touches it.
        (void) screenshot_->record(context_, command_buffer,
                                   ScreenshotImage{
                                           .image = viewport->image(),
                                           .format = viewport->format(),
                                           .extent = {viewport->extent().width, viewport->extent().height},
                                           .layout_before = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                           .stage_before = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                           .access_before = VK_ACCESS_2_NONE,
                                           .layout_after = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                           .stage_after = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                           .access_after = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                   },
                                   frame_index);
        render_pass::present_swapchain(command_buffer, swapchain_image.image);
    } else {
        bool const screenshot_recorded =
                pending.has_value() &&
                screenshot_->record(context_, command_buffer,
                                    ScreenshotImage{
                                            .image = swapchain_image.image,
                                            .format = swapchain_image.format,
                                            .extent = swapchain_image.extent,
                                            .layout_before = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            .stage_before = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                            .access_before = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                            .layout_after = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                    },
                                    frame_index);

        if (!screenshot_recorded) {
            render_pass::present_swapchain(command_buffer, swapchain_image.image);
        }
    }

    auto &frame_query = timestamp_queries_[frame_index];

    vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, frame_query.query_pool,
                         (static_cast<std::uint32_t>(RenderStage::FullFrame) * 2) + 1);

    frame_query.has_results = true;
    pipeline_stat_queries_[frame_index].has_results = true;
}

auto Renderer::make_pass_context(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> render_pass::Context {
    return render_pass::Context{
            .command_buffer = command_buffer,
            .frame_index = frame_index,
            .pipeline_graph = pipeline_graph_,
            .resource_table = gpu_resource_table_,
            .timestamp_query_pool = timestamp_queries_[frame_index].query_pool,
    };
}

auto Renderer::record_overlay_prepares(render_pass::Context const &pass_context) -> void {
    auto const command_buffer = pass_context.command_buffer;
    auto &frame_query = timestamp_queries_[pass_context.frame_index];

    frame_query.overlays.clear();

    bool any_gpu_writes = false;

    for (auto &entry: overlays_.all()) {
        frame_query.overlays.push_back(RecordedOverlay{
                .name = entry.desc.name,
                .stage = entry.desc.stage,
                .slot = entry.slot,
        });

        // Written even without a prepare() so all four queries are always available.
        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pass_context.timestamp_query_pool,
                             overlay_query(entry.slot, 0));

        if (entry.desc.prepare) {
            ZoneTransientN(cpu_zone, entry.desc.name.c_str(), true);
            TracyVkZoneTransient(context_.host_query_context.context, gpu_zone, command_buffer, entry.desc.name.c_str(),
                                 true);

            auto const result = entry.desc.prepare(OverlayPrepareContext{
                    .command_buffer = command_buffer,
                    .frame_index = pass_context.frame_index,
            });

            any_gpu_writes = any_gpu_writes || result == OverlayPrepareResult::recorded_gpu_writes;
        }

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pass_context.timestamp_query_pool,
                             overlay_query(entry.slot, 1));
    }

    if (!any_gpu_writes) {
        return;
    }

    // Makes every prepare() write visible to the stages an overlay's draw can read from.
    VkMemoryBarrier2 const barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
                            VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT |
                             VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT,
    };

    VkDependencyInfo const dependency_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr,
    };

    vkCmdPipelineBarrier2(command_buffer, &dependency_info);
}

auto Renderer::record_overlay_stage(render_pass::Context const &pass_context, OverlayStage stage,
                                    OverlayScope const &scope, glm::mat4 const &view_projection) -> void {
    auto const command_buffer = pass_context.command_buffer;

    for (auto &entry: overlays_.stage(stage)) {
        ZoneTransientN(cpu_zone, entry.desc.name.c_str(), true);
        TracyVkZoneTransient(context_.host_query_context.context, gpu_zone, command_buffer, entry.desc.name.c_str(),
                             true);

        render_pass::set_overlay_baseline_state(command_buffer, stage, scope);

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             pass_context.timestamp_query_pool, overlay_query(entry.slot, 2));

        entry.desc.record(OverlayRecordContext{
                .command_buffer = command_buffer,
                .frame_index = pass_context.frame_index,
                .scope = scope,
                .view_projection = view_projection,
        });

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             pass_context.timestamp_query_pool, overlay_query(entry.slot, 3));
    }
}

auto Renderer::read_overlay_timings(FrameTimestamps const &frame_query) -> void {
    last_frame_timings_.overlays.clear();

    auto const to_milliseconds = [&](std::uint64_t begin, std::uint64_t end) {
        return end > begin ? static_cast<float>(end - begin) * timestamp_period_ / 1'000'000.0F : 0.0F;
    };

    for (auto const &recorded: frame_query.overlays) {
        std::array<std::uint64_t, queries_per_overlay> results{};

        auto const query_result = vkGetQueryPoolResults(
                context_.device, frame_query.query_pool, overlay_query(recorded.slot, 0), queries_per_overlay,
                sizeof(results), results.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);

        if (query_result != VK_SUCCESS) {
            continue;
        }

        last_frame_timings_.overlays.push_back(OverlayTiming{
                .name = recorded.name,
                .stage = recorded.stage,
                .prepare_milliseconds = to_milliseconds(results[0], results[1]),
                .record_milliseconds = to_milliseconds(results[2], results[3]),
        });
    }
}

auto Renderer::register_overlay(OverlayDesc desc) -> std::expected<OverlayRegistration, RendererError> {
    auto registration = overlays_.add(std::move(desc));

    if (!registration) {
        switch (registration.error()) {
            case OverlayRegistryError::capacity_exceeded:
                return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
            case OverlayRegistryError::empty_name:
            case OverlayRegistryError::missing_record_callback:
            case OverlayRegistryError::invalid_stage:
                break;
        }

        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    return std::move(*registration);
}

auto Renderer::register_light_icon_overlay() -> std::expected<void, RendererError> {
    auto registration = register_overlay(OverlayDesc{
            .name = "Light icons",
            .stage = OverlayStage::scene,
            // Before the default-order overlays.
            .order = -100,
            .prepare = {},
            .record =
                    [this](OverlayRecordContext const &context) {
                        if (!debug_draw_light_icons_) {
                            return;
                        }

                        auto const &frame = frames_[context.frame_index];

                        render_pass::light_icons(make_pass_context(context.command_buffer, context.frame_index),
                                                 render_pass::LightIconsInfo{
                                                         .lights_address = frame.lights_buffer.device_address,
                                                         .ubo_address = ubos_[context.frame_index].device_address,
                                                         .light_count = frame.light_count,
                                                         .pipeline = light_icon_pipeline_,
                                                         .icon_texture_index = light_icon_texture_.index,
                                                         .sampler_index = sampler_storage_.linear_clamp().index,
                                                         .icon_world_size = light_icon_world_size_,
                                                 },
                                                 context.scope);
                    },
    });

    if (!registration) {
        return std::unexpected(registration.error());
    }

    light_icon_overlay_ = std::move(*registration);
    return {};
}

auto Renderer::record_frame(FrameRecordInfo const &info) -> std::expected<void, RendererError> {
    ZoneScopedNC("RecordFrame", tracy::Color::RoyalBlue);

    auto const command_buffer = info.command_buffer;
    auto const &swapchain_image = info.swapchain_image;
    auto const frame_index = info.frame_index;

    if (!initialized_ || command_buffer == VK_NULL_HANDLE || swapchain_image.image == VK_NULL_HANDLE ||
        swapchain_image.view == VK_NULL_HANDLE || swapchain_image.format == VK_FORMAT_UNDEFINED ||
        swapchain_image.extent.width == 0 || swapchain_image.extent.height == 0 || frame_index >= frames_.size()) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    screenshot_->try_resolve(frame_index);

    auto &frame = frames_[frame_index];
    consume_culled_readback(frame);

    auto const targets = resolve_frame_targets(frame);
    if (!targets) {
        return std::unexpected(targets.error());
    }

    auto const pass_context = make_pass_context(command_buffer, frame_index);

    // Registration changes made by overlay callbacks land after recording, so prepare, stages and timing all see
    // the same set.
    auto const overlay_iteration = overlays_.iterate();

    OverlayScope const scene_scope{
            .extent = targets->extent,
            .colour_format = frame.forward_target.hdr_format(),
            .depth_format = frame.forward_target.depth_format(),
            .samples = samples_,
    };

    OverlayScope const ui_scope{
            .extent = swapchain_image.extent,
            .colour_format = swapchain_image.format,
            .depth_format = VK_FORMAT_UNDEFINED,
            .samples = VK_SAMPLE_COUNT_1_BIT,
    };

    auto scene_overlays = [&] {
        record_overlay_stage(pass_context, OverlayStage::scene, scene_scope, frame.view_projection);
    };
    auto ui_overlays = [&] { record_overlay_stage(pass_context, OverlayStage::ui, ui_scope, frame.view_projection); };

    record_overlay_prepares(pass_context);

    if (auto shadows = record_shadow_pass(pass_context, frame, *targets); !shadows) {
        return shadows;
    }

    if (auto prepass = record_depth_prepass(pass_context, frame, *targets); !prepass) {
        return prepass;
    }

    auto const ao_texture_index = record_ambient_occlusion_pass(pass_context, frame, *targets);
    if (!ao_texture_index) {
        return std::unexpected(ao_texture_index.error());
    }

    auto const hdr = record_forward_pass(pass_context, frame, *targets, *ao_texture_index,
                                         render_pass::Callback::bind(scene_overlays));
    if (!hdr) {
        return std::unexpected(hdr.error());
    }

    auto const bloom = record_bloom_pass(pass_context, frame, *targets, *hdr);
    if (!bloom) {
        return std::unexpected(bloom.error());
    }

    if (auto composited = record_composite_pass(pass_context, *targets, swapchain_image, *hdr, *bloom,
                                                info.composite_target, render_pass::Callback::bind(ui_overlays));
        !composited) {
        return composited;
    }

    record_frame_end(command_buffer, swapchain_image,
                     info.composite_target == CompositeTarget::swapchain ? nullptr : targets->viewport, frame_index);

    TracyVkCollectHost(context_.host_query_context.context);
    return {};
}

auto Renderer::create_frame_targets(std::uint32_t frame_index, VkExtent2D extent)
        -> std::expected<OwnedFrameTargets, RendererError> {
    OwnedFrameTargets targets;

    auto const target_name = std::format("renderer.forward_target_{}", frame_index);
    auto forward_target = ForwardTarget::create(image_storage_, ForwardTargetCreateInfo{
                                                                        .extent = extent,
                                                                        .hdr_format = hdr_format_,
                                                                        .depth_format = depth_format_,
                                                                        .samples = samples_,
                                                                        .debug_name = target_name,
                                                                });

    if (!forward_target) {
        return std::unexpected(RendererError{
                .type = RendererErrorType::forward_target_error,
                .cause = ErrorCause{Boxed<ForwardTargetError>{forward_target.error()}},
        });
    }

    targets.forward_target = std::move(*forward_target);

    auto const viewport_target_name = std::format("renderer.viewport_target_{}", frame_index);
    auto viewport_target = create_held_image(image_storage_, ImageCreateInfo{
            .extent = VkExtent3D{.width = extent.width, .height = extent.height, .depth = 1},
            .format = swapchain_format_,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
            .image_type = VK_IMAGE_TYPE_2D,
            .view_type = VK_IMAGE_VIEW_TYPE_2D,
            .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
            .flags = 0,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .mip_levels = 1,
            .array_layers = 1,
            .debug_name = viewport_target_name,
    });

    if (!viewport_target) {
        return std::unexpected(make_image_error(viewport_target.error()));
    }

    targets.viewport_target = std::move(*viewport_target);

    auto const bloom_target_name = std::format("renderer.bloom_target_{}", frame_index);
    auto bloom_image = create_held_image(image_storage_, ImageCreateInfo{
            .extent = VkExtent3D{.width = extent.width / 2, .height = extent.height / 2, .depth = 1},
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
            .mip_levels = render_pass::bloom_mip_count,
            .array_layers = 1,
            .create_mip_layer_views = true,
            .debug_name = bloom_target_name,
    });

    if (!bloom_image) {
        return std::unexpected(make_image_error(bloom_image.error()));
    }

    // Into the BloomTarget before its mip slots, so a failed registration still releases the slots first.
    targets.bloom_target.image = std::move(*bloom_image);

    auto const *bloom_image_ptr = targets.bloom_target.image.get();

    for (std::uint32_t mip = 0; mip < render_pass::bloom_mip_count; ++mip) {
        auto const view = bloom_image_ptr->mip_layer_view(mip, 0);

        auto mip_slot = register_held_view(image_storage_, ImageViewRegistration{
                                                                   .sampled_2d = view,
                                                                   .storage_2d = view,
                                                           });

        if (!mip_slot) {
            return std::unexpected(make_image_error(mip_slot.error()));
        }

        targets.bloom_target.mip_slots[mip] = std::move(*mip_slot);
    }

    auto const create_ao_image = [&](std::string_view kind) -> std::expected<ImageHolder, RendererError> {
        auto const name = std::format("renderer.ao_{}_{}", kind, frame_index);

        auto image = create_held_image(image_storage_, ImageCreateInfo{
                .extent = VkExtent3D{.width = extent.width, .height = extent.height, .depth = 1},
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                .image_type = VK_IMAGE_TYPE_2D,
                .view_type = VK_IMAGE_VIEW_TYPE_2D,
                .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d) |
                                    image_descriptor_view_bit(ImageDescriptorView::storage_2d),
                .flags = 0,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .mip_levels = 1,
                .array_layers = 1,
                .debug_name = name,
        });

        if (!image) {
            return std::unexpected(make_image_error(image.error()));
        }

        return std::move(*image);
    };

    auto ao_raw = create_ao_image("raw");

    if (!ao_raw) {
        return std::unexpected(ao_raw.error());
    }

    targets.ao_target.raw = std::move(*ao_raw);

    auto ao_denoised = create_ao_image("denoised");

    if (!ao_denoised) {
        return std::unexpected(ao_denoised.error());
    }

    targets.ao_target.denoised = std::move(*ao_denoised);

    return targets;
}

auto Renderer::resize(VkExtent2D extent) -> std::expected<void, RendererError> {
    if (extent.width == 0 || extent.height == 0) {
        return {};
    }

    if (extent.width == extent_.width && extent.height == extent_.height) {
        return {};
    }

    // The forward targets below are destroyed, so wait for the GPU regardless of what the caller did.
    if (auto waited = wait_idle(); !waited) {
        return std::unexpected(waited.error());
    }

    // Every frame's replacements are built before any frame is touched: on failure the current targets stay, and the
    // replacements built so far are destroyed with `replacements`.
    std::vector<OwnedFrameTargets> replacements;
    replacements.reserve(frames_.size());

    for (std::size_t index = 0; index < frames_.size(); ++index) {
        auto targets = create_frame_targets(static_cast<std::uint32_t>(index), extent);

        if (!targets) {
            return std::unexpected(targets.error());
        }

        replacements.push_back(std::move(*targets));
    }

    // Moving in destroys the old targets.
    for (std::size_t index = 0; index < frames_.size(); ++index) {
        auto &frame = frames_[index];
        auto &targets = replacements[index];

        frame.forward_target = std::move(targets.forward_target);
        frame.viewport_target = std::move(targets.viewport_target);
        frame.bloom_target = std::move(targets.bloom_target);
        frame.ao_target = std::move(targets.ao_target);
    }

    extent_ = extent;

    return {};
}

auto Renderer::queue_render_thread_event(std::move_only_function<void()> &&task) -> void {
    std::lock_guard lock(queue_mutex_);
    event_queue_.push(std::move(task));
    queued_events_.fetch_add(1);
}

auto Renderer::drain_event_queue() -> void {
    if (queued_events_.load(std::memory_order_relaxed) == 0) [[likely]] {
        return;
    }

    std::queue<std::move_only_function<void()>> local_queue;

    {
        std::lock_guard lock(queue_mutex_);
        std::swap(event_queue_, local_queue);
    }

    while (!local_queue.empty()) {
        local_queue.front()();
        local_queue.pop();
    }
}

auto Renderer::mark_shadow_casters_dirty() noexcept -> void {
    ++shadow_caster_revision_;

    if (shadow_caster_revision_ == 0) {
        shadow_caster_revision_ = 1;
        cached_shadow_caster_revision_ = 0;
        for (auto &cached: shadow_cascade_cache_) {
            cached.valid = false;
        }
        shadow_atlas_initialized_ = false;
    }
}

auto Renderer::request_screenshot(ScreenshotSource source) noexcept -> void { screenshot_->request(source); }
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
    auto const indirect_size = static_cast<VkDeviceSize>(frame.indirect_commands.size()) * sizeof(GpuDrawCommand);
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
                .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
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
                .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = frame.transform_buffer.buffer,
                .offset = 0,
                .size = transform_size,
        };
    }

    // Read by the shadow pass's indirect draws, its task shader and main_cs.
    if (indirect_size != 0) {
        barriers[barrier_count++] = VkBufferMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = frame.indirect_buffer.buffer,
                .offset = 0,
                .size = indirect_size,
        };
    }

    // The culled and visible buffers are written only by main_cs, so only batch_bounds_buffer needs a barrier.
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
    slot_override_submissions_.clear();
    point_light_submissions_.clear();
    spot_light_submissions_.clear();
}
