#include "rendering/renderer.hxx"
#include "gpu/skinning.hxx"
#include "core/error_describe.hxx"
#include "core/paths.hxx"
#include "core/perf_events.hxx"
#include "core/resources.hxx"

#include "gpu/device_wait.hxx"
#include "rendering/frame_graph/compiler.hxx"
#include "rendering/frame_graph/executor.hxx"
#include "rendering/frame_graph/pass_context.hxx"

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
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "assets/material_storage.hxx"
#include "assets/slang_compiler.hxx"
#include "core/logger.hxx"
#include "assets/shader_pack.hxx"
#include "core/thread_pool.hxx"
#include <atomic>
#include <thread>
#include "gpu/buffer.hxx"
#include "gpu/context.hxx"
#include "gpu/device_error.hxx"
#include "gpu/gpu_resource_table.hxx"
#include "gpu/sampler_storage.hxx"
#include "gpu/vk_barrier.hxx"
#include "maths/aabb.hxx"
#include "rendering/render_passes.hxx"
#include "rendering/screenshot.hxx"
#include "rendering/sky_model.hxx"

#include "shader_push_constants.hxx"

namespace {
    // Folds `size` bytes into `hash`, a word at a time; the order of the bytes matters. Not a cryptographic hash: it is
    // for telling two frames' data apart (see Renderer::set_frame_checksum_enabled).
    auto checksum_bytes(std::uint64_t hash, void const *data, std::size_t size) noexcept -> std::uint64_t {
        constexpr auto prime = std::uint64_t{0x9E3779B97F4A7C15ULL};
        auto const *bytes = static_cast<unsigned char const *>(data);

        hash = (hash ^ size) * prime;
        auto offset = std::size_t{0};
        for (; offset + sizeof(std::uint64_t) <= size; offset += sizeof(std::uint64_t)) {
            auto word = std::uint64_t{0};
            std::memcpy(&word, bytes + offset, sizeof(word));
            hash = (hash ^ word) * prime;
            hash ^= hash >> 29U;
        }
        for (; offset < size; ++offset) {
            hash = (hash ^ bytes[offset]) * prime;
        }
        return hash;
    }

    // The LOD an instance at `instance_position` is drawn at, from its distance to the camera.
    auto lod_index_for(glm::vec3 const &camera_position, glm::vec3 const &instance_position) noexcept -> std::uint32_t {
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
    }

    constexpr std::uint32_t queries_per_overlay = 4;
    constexpr std::uint32_t full_frame_query_count = 2;
    constexpr std::uint32_t overlay_query_base = full_frame_query_count;
    constexpr std::uint32_t total_query_count =
            full_frame_query_count + (OverlayRegistry::max_overlays * queries_per_overlay);

    [[nodiscard]] constexpr auto overlay_query(std::uint32_t slot, std::uint32_t which) noexcept -> std::uint32_t {
        return overlay_query_base + (slot * queries_per_overlay) + which;
    }

    [[nodiscard]] constexpr auto model_source_key(ModelHandle handle) noexcept -> std::uint64_t {
        return (static_cast<std::uint64_t>(handle.generation) << 32U) | handle.index;
    }
}

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

    static_assert(sizeof(VkDrawMeshTasksIndirectCommandEXT) == 12);
    static_assert(offsetof(GpuDrawCommand, group_count_x) == offsetof(VkDrawMeshTasksIndirectCommandEXT, groupCountX));
    static_assert(offsetof(GpuDrawCommand, group_count_y) == offsetof(VkDrawMeshTasksIndirectCommandEXT, groupCountY));
    static_assert(offsetof(GpuDrawCommand, group_count_z) == offsetof(VkDrawMeshTasksIndirectCommandEXT, groupCountZ));
}

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
}

namespace {
    auto make_error(RendererErrorType type) -> RendererError {
        return RendererError{
                .type = type,
        };
    }

    auto buffers_shared_between_queues(VulkanContext const &context) -> bool {
        return context.queue_families.compute != context.queue_families.graphics;
    }

    auto create_shared_buffer(VulkanContext &context, BufferCreateInfo info)
            -> decltype(Buffer::create(context, info)) {
        if (buffers_shared_between_queues(context)) {
            info.concurrent_families = {context.queue_families.graphics, context.queue_families.compute};
            info.concurrent_family_count = 2;
        }
        return Buffer::create(context, info);
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

    constexpr std::uint32_t occlusion_view_disabled = 0U;
    constexpr std::uint32_t occlusion_view_enabled = 1U;
    constexpr std::uint32_t occlusion_view_force_occluded = 2U;

    struct OcclusionViewStates {
        std::uint32_t early = occlusion_view_disabled;
        std::uint32_t late = occlusion_view_disabled;
    };

    [[nodiscard]] constexpr auto occlusion_view_states(bool active, bool history_valid, OcclusionTestMode mode) noexcept
            -> OcclusionViewStates {
        if (!active) {
            return {};
        }

        switch (mode) {
            case OcclusionTestMode::hiz:
                return OcclusionViewStates{
                        .early = history_valid ? occlusion_view_enabled : occlusion_view_disabled,
                        .late = occlusion_view_enabled,
                };

            case OcclusionTestMode::never_occluded:
                return {};

            case OcclusionTestMode::always_defer:
                return OcclusionViewStates{
                        .early = occlusion_view_force_occluded,
                        .late = occlusion_view_disabled,
                };
        }

        return {};
    }
}

Renderer::Renderer(VulkanContext &context) noexcept :
    context_(context), screenshot_(std::make_unique<ScreenshotCapture>()) {}
Renderer::~Renderer() noexcept = default;

namespace {
    [[nodiscard]] auto make_pipeline_register_infos(RendererCreateInfo const &create_info)
            -> std::vector<PipelineRegisterInfo> {
    std::vector<PipelineRegisterInfo> pipeline_infos;
    pipeline_infos.reserve(13);

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/forward_geom.slang"),
                                    .entry_point = FlyString{"main_task"},
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/forward_geom.slang"),
                                    .entry_point = FlyString{"main_mesh"},
                                    .stage = renderer::ShaderStage::mesh,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/forward_geom.slang"),
                                    .entry_point = FlyString{"main_fs"},
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
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/forward_geom.slang"),
                                    .entry_point = FlyString{"main_task"},
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/forward_geom.slang"),
                                    .entry_point = FlyString{"main_mesh"},
                                    .stage = renderer::ShaderStage::mesh,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/forward_geom.slang"),
                                    .entry_point = FlyString{"main_fs"},
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
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/light_icons.slang"),
                                    .entry_point = FlyString{"main_task"},
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/light_icons.slang"),
                                    .entry_point = FlyString{"main_mesh"},
                                    .stage = renderer::ShaderStage::mesh,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/light_icons.slang"),
                                    .entry_point = FlyString{"main_fs"},
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
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/shadow_depth.slang"),
                                    .entry_point = FlyString{"main_task"},
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/shadow_depth.slang"),
                                    .entry_point = FlyString{"main_mesh"},
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
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/shadow_depth.slang"),
                                    .entry_point = FlyString{"main_task"},
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/shadow_depth.slang"),
                                    .entry_point = FlyString{"main_mesh"},
                                    .stage = renderer::ShaderStage::mesh,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/shadow_depth.slang"),
                                    .entry_point = FlyString{"main_fs"},
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
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/depth_prepass.slang"),
                                    .entry_point = FlyString{"main_task"},
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/depth_prepass.slang"),
                                    .entry_point = FlyString{"main_mesh"},
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
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/depth_prepass.slang"),
                                    .entry_point = FlyString{"main_task"},
                                    .stage = renderer::ShaderStage::task,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/depth_prepass.slang"),
                                    .entry_point = FlyString{"main_mesh"},
                                    .stage = renderer::ShaderStage::mesh,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/depth_prepass.slang"),
                                    .entry_point = FlyString{"main_fs"},
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
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/composite.slang"),
                                    .entry_point = FlyString{"main_vs"},
                                    .stage = renderer::ShaderStage::vertex,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/composite.slang"),
                                    .entry_point = FlyString{"main_fs"},
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
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/frustum_cull.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/bloom_downsample.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
                                    .source_path = data_path("assets/shaders/bloom_upsample.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
                                    .source_path = data_path("assets/shaders/gtao.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/gtao_denoise.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
    });

    for (std::size_t const meshlet_index: {0U, 1U, 3U, 4U, 5U, 6U}) {
        auto instanced = pipeline_infos[meshlet_index];

        std::erase_if(instanced.stages, [](renderer::ShaderCompileRequest const &stage) {
            return stage.stage == renderer::ShaderStage::task;
        });

        for (auto &stage: instanced.stages) {
            if (stage.stage == renderer::ShaderStage::mesh) {
                stage.stage = renderer::ShaderStage::vertex;
                stage.entry_point = FlyString{"main_vs"};
            }
        }

        instanced.debug_name += "_instanced";
        pipeline_infos.push_back(std::move(instanced));
    }

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/light_cluster.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.light_cluster_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/light_cull.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.light_cull_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/frustum_cull.slang"),
                                    .entry_point = FlyString{"late_cs"},
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
            .debug_name = "renderer.occlusion_cull_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/hiz_build.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.hiz_build_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/env_brdf_lut.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.env_brdf_lut_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/env_equirect_to_cube.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.env_equirect_to_cube_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/env_sky_to_cube.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.env_sky_to_cube_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/env_downsample.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.env_downsample_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/env_sh_project.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.env_sh_project_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/env_prefilter.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.env_prefilter_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/skybox.slang"),
                                    .entry_point = FlyString{"main_vs"},
                                    .stage = renderer::ShaderStage::vertex,
                                    .include_directories = {},
                                    .defines = {},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/skybox.slang"),
                                    .entry_point = FlyString{"main_fs"},
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
            .debug_name = "renderer.skybox_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/instance_lod.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.instance_lod_pipeline",
    });

    // The blended forward pipelines (1: meshlet, 14: instanced) keep only the faces each command asks for and shade
    // water; BLEND_PASS keeps that out of the opaque pipelines.
    for (auto const blend_index: {std::size_t{1}, std::size_t{14}}) {
        for (auto &stage: pipeline_infos[blend_index].stages) {
            stage.defines.push_back(renderer::ShaderDefine{.name = "BLEND_PASS", .value = "1"});
        }
    }

    for (auto const source_index: {std::size_t{0}, std::size_t{13}}) {
        auto outline_info = pipeline_infos[source_index];

        for (auto &stage: outline_info.stages) {
            if (stage.stage == renderer::ShaderStage::fragment) {
                stage.defines.push_back(renderer::ShaderDefine{.name = "OUTLINE_MASK", .value = "1"});
            }
        }

        outline_info.debug_name += ".outline";
        pipeline_infos.push_back(std::move(outline_info));
    }

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/skin.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.skin_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/decal_cluster.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.decal_cluster_pipeline",
    });

    pipeline_infos.push_back(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path("assets/shaders/shadow_cull.slang"),
                                    .entry_point = FlyString{"main_cs"},
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
            .debug_name = "renderer.shadow_cull_pipeline",
    });

        return pipeline_infos;
    }
}

namespace {
    // Creating the compiler costs ~300 ms, so a run served entirely by a shader pack never creates one.
    std::atomic<bool> compiler_created{false}; // NOLINT: process-wide, like the compiler itself.
}

auto Renderer::compiler() noexcept -> renderer::SlangCompiler & {
    static auto compiler_ = [] {
        compiler_created.store(true, std::memory_order_release);
        auto created = renderer::SlangCompiler::create();

        if (!created) {
            warn("Slang is unavailable ({}); shaders come from the shader pack only", created.error().diagnostics);
            return std::make_unique<renderer::SlangCompiler>();
        }

        return std::make_unique<renderer::SlangCompiler>(std::move(*created));
    }();

    return *compiler_;
}

namespace {
    std::jthread shader_prefetch_thread; // NOLINT: process-wide, joined by Renderer::destroy.
}

auto Renderer::prefetch_shaders() -> void {
    // Compiling needs no Vulkan state, only the shader requests, so it can overlap window and device creation.
    // The formats in the create info only shape the pipelines, never the compile requests.
    shader_prefetch_thread = std::jthread{[] {
        auto requests = std::vector<renderer::ShaderCompileRequest>{};

        for (auto const &info: make_pipeline_register_infos(RendererCreateInfo{})) {
            requests.insert(requests.end(), info.stages.begin(), info.stages.end());
        }

        if (auto const pack = renderer::installed_shader_pack()) {
            std::erase_if(requests, [&pack](auto const &request) {
                return pack->find(renderer::shader_request_key(request)).has_value();
            });
        }

        if (requests.empty()) {
            return;
        }

        auto const &shader_compiler = compiler();

        if (!shader_compiler.valid()) {
            return;
        }

        shader_compiler.prefetch(requests);
    }};
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
    pending_samples_ = create_info.samples;
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
                              .cache_file_path = cache_path("pipeline_cache.bin").absolute(),
                              .shader_binary_cache_directory = cache_path("shader_binaries").absolute(),
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

    auto pipeline_infos = make_pipeline_register_infos(create_info);

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
    light_cluster_pipeline_ = *registered_pipelines[19];
    light_cull_pipeline_ = *registered_pipelines[20];
    occlusion_cull_pipeline_ = *registered_pipelines[21];
    hiz_build_pipeline_ = *registered_pipelines[22];
    skybox_pipeline_ = *registered_pipelines[29];
    instance_lod_pipeline_ = *registered_pipelines[30];
    forward_outline_pipeline_ = *registered_pipelines[31];
    forward_outline_instanced_pipeline_ = *registered_pipelines[32];
    skin_pipeline_ = *registered_pipelines[33];
    decal_cluster_pipeline_ = *registered_pipelines[34];
    shadow_cull_pipeline_ = *registered_pipelines[35];

    {
        auto initialised = environment_.initialize(EnvironmentSystem::CreateInfo{
                .context = &context_,
                .images = &image_storage_,
                .samplers = &sampler_storage_,
                .pipelines = &pipeline_graph_,
                .handles =
                        EnvironmentPipelines{
                                .brdf_lut = *registered_pipelines[23],
                                .equirect_to_cube = *registered_pipelines[24],
                                .sky_to_cube = *registered_pipelines[25],
                                .downsample = *registered_pipelines[26],
                                .sh_project = *registered_pipelines[27],
                                .prefilter = *registered_pipelines[28],
                        },
                .frames_in_flight = frames_in_flight,
        });

        if (!initialised) {
            return std::unexpected(initialised.error());
        }
    }

    {
        auto const light_icon_encoded = read_resource(data_path("assets/textures/light_bulb.png"));
        auto light_icon_image = light_icon_encoded ? DecodedImage::load_from_memory(*light_icon_encoded) : std::nullopt;
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

    for (auto const lut: {DefaultColourLut::cinematic, DefaultColourLut::vintage, DefaultColourLut::noir}) {
        auto const added = add_colour_lut(std::string{default_colour_lut_name(lut)}, make_default_colour_lut(lut));
        if (!added) {
            return std::unexpected(added.error());
        }
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
    maximum_skin_palette_matrices_ = create_info.maximum_skin_palette_matrices;
    maximum_skin_jobs_ = create_info.maximum_skin_jobs;
    skin_scratch_capacity_ = create_info.skin_scratch_bytes;
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

    auto const cull_batch_capacity = std::min(maximum_draw_count_, maximum_cull_batch_count);
    auto const occlusion_indirect_size = static_cast<VkDeviceSize>(cull_batch_capacity) * sizeof(GpuDrawCommand);
    auto const cull_chunk_capacity =
            static_cast<VkDeviceSize>(cull_batch_capacity) +
            (static_cast<VkDeviceSize>(maximum_draw_count_) + cull_chunk_size - 1) / cull_chunk_size;
    auto const cull_chunks_size = cull_chunk_capacity * cull_chunk_bytes;
    cull_chunk_capacity_ = cull_chunk_capacity;
    auto const occlusion_candidates_size = static_cast<VkDeviceSize>(maximum_draw_count_) * sizeof(std::uint32_t);
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

    frame_layout_ = FrameBufferLayout{
            .draw_size = draw_size,
            .transform_size = transform_size,
            .indirect_size = indirect_size,
            .batch_bounds_size = batch_bounds_size,
            .occlusion_indirect_size = occlusion_indirect_size,
            .cull_chunks_size = cull_chunks_size,
            .occlusion_candidates_size = occlusion_candidates_size,
            .upload_size = upload_size,
            .transform_offset = transform_offset,
            .indirect_offset = indirect_offset,
            .batch_bounds_offset = batch_bounds_offset,
    };

    auto shadow_atlas = create_shadow_atlas();

    if (!shadow_atlas) {
        return std::unexpected(shadow_atlas.error());
    }

    shadow_atlas_ = std::move(*shadow_atlas);

    auto local_shadow_atlas = create_held_image(
            image_storage_,
            ImageCreateInfo{
                    .extent = VkExtent3D{.width = local_shadows::atlas_size, .height = local_shadows::atlas_size,
                                         .depth = 1},
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
                    .debug_name = "renderer.local_shadow_atlas",
            });

    if (!local_shadow_atlas) {
        return std::unexpected(make_image_error(local_shadow_atlas.error()));
    }

    local_shadow_atlas_ = std::move(*local_shadow_atlas);

    frames_.resize(frames_in_flight);
    transient_allocator_->initialize(context_, image_storage_, frames_in_flight);

    for (std::uint32_t frame_index = 0; frame_index < static_cast<std::uint32_t>(frames_.size()); ++frame_index) {
        if (auto created = create_frame_resources(frames_[frame_index], create_info.extent, frame_index); !created) {
            return std::unexpected(created.error());
        }
    }

    auto hiz = create_hiz_pyramid(create_info.extent);

    if (!hiz) {
        return std::unexpected(hiz.error());
    }

    hiz_ = std::move(*hiz);
    hiz_history_valid_ = false;

    VkFormatProperties hiz_format_properties{};
    vkGetPhysicalDeviceFormatProperties(context_.physical_device, VK_FORMAT_R32_SFLOAT, &hiz_format_properties);
    hiz_debug_view_supported_ =
            (hiz_format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;

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

    if (!pass_profiler_->initialize(frame_graph::PassProfilerCreateInfo{
                .physical_device = context_.physical_device,
                .device = context_.device,
                .queue_family = {context_.queue_families.graphics, context_.queue_families.compute},
                .timestamp_period = timestamp_period_,
                .slots = frames_in_flight,
                .max_passes = 64,
        })) {
        error("Failed to create the frame graph pass profiler");
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    auto ubos = create_ubos();

    if (!ubos) {
        return std::unexpected(ubos.error());
    }

    ubos_ = std::move(*ubos);

    mark_lights_dirty();

    if (auto light_icons = register_light_icon_overlay(); !light_icons) {
        return std::unexpected(light_icons.error());
    }

    rollback_on_failure = false;
    initialized_ = true;

    debug("[Renderer::initialize] exit: success");

    return {};
}

auto Renderer::create_shadow_atlas() -> std::expected<ImageHolder, RendererError> {
    auto shadow_atlas = create_held_image(
            image_storage_,
            ImageCreateInfo{
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

    return std::move(*shadow_atlas);
}

auto Renderer::create_frame_resources(RendererFrame &frame, VkExtent2D extent, std::uint32_t frame_index)
        -> std::expected<void, RendererError> {
    auto const &draw_size = frame_layout_.draw_size;
    auto const &transform_size = frame_layout_.transform_size;
    auto const &indirect_size = frame_layout_.indirect_size;
    auto const &culled_indirect_size = frame_layout_.indirect_size;
    auto const &batch_bounds_size = frame_layout_.batch_bounds_size;
    auto const &occlusion_indirect_size = frame_layout_.occlusion_indirect_size;
    auto const &cull_chunks_size = frame_layout_.cull_chunks_size;
    auto const &occlusion_candidates_size = frame_layout_.occlusion_candidates_size;
    auto const &upload_size = frame_layout_.upload_size;
    auto const &transform_offset = frame_layout_.transform_offset;
    auto const &indirect_offset = frame_layout_.indirect_offset;
    auto const &batch_bounds_offset = frame_layout_.batch_bounds_offset;
    constexpr VkDeviceSize visible_lights_size =
            (sizeof(glm::vec4) + sizeof(std::uint32_t)) * VkDeviceSize{maximum_light_count} + sizeof(std::uint32_t);


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

    auto draws = create_shared_buffer(context_, BufferCreateInfo{
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

    auto transforms = create_shared_buffer(context_, BufferCreateInfo{
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

    auto indirect = create_shared_buffer(
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

    auto batch_bounds = create_shared_buffer(context_, BufferCreateInfo{
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

    auto culled_indirect = create_shared_buffer(
            context_, BufferCreateInfo{
                              .size = culled_indirect_size,
                              .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              .memory = BufferMemory::device,
                              .debug_name = "renderer.frame_culled_indirect",
                      });

    if (!culled_indirect) {
        return std::unexpected(make_device_error(culled_indirect.error()));
    }

    frame.culled_indirect_buffer = std::move(*culled_indirect);

    auto visible_draws = create_shared_buffer(context_, BufferCreateInfo{
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

    auto visible_transforms =
            create_shared_buffer(context_, BufferCreateInfo{
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

    struct OcclusionBufferSpec {
        Buffer *buffer;
        VkDeviceSize size;
        VkBufferUsageFlags usage;
        BufferMemory memory;
        char const *debug_name;
    };

    constexpr VkBufferUsageFlags storage_usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    constexpr VkBufferUsageFlags indirect_usage = storage_usage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

    std::array const occlusion_buffers{
            OcclusionBufferSpec{.buffer = &frame.occlusion_views_buffer,
                                .size = 2 * sizeof(GpuOcclusionView),
                                .usage = storage_usage,
                                .memory = BufferMemory::upload,
                                .debug_name = "renderer.frame_occlusion_views"},
            OcclusionBufferSpec{.buffer = &frame.occlusion_candidates_buffer,
                                .size = occlusion_candidates_size,
                                .usage = storage_usage,
                                .memory = BufferMemory::device,
                                .debug_name = "renderer.frame_occlusion_candidates"},
            OcclusionBufferSpec{.buffer = &frame.cull_chunks_buffer,
                                .size = cull_chunks_size,
                                .usage = storage_usage,
                                .memory = BufferMemory::device,
                                .debug_name = "renderer.frame_cull_chunks"},
            OcclusionBufferSpec{.buffer = &frame.late_indirect_buffer,
                                .size = occlusion_indirect_size,
                                .usage = indirect_usage,
                                .memory = BufferMemory::device,
                                .debug_name = "renderer.frame_late_indirect"},
            OcclusionBufferSpec{.buffer = &frame.merged_indirect_buffer,
                                .size = occlusion_indirect_size,
                                .usage = indirect_usage,
                                .memory = BufferMemory::device,
                                .debug_name = "renderer.frame_merged_indirect"},
            OcclusionBufferSpec{.buffer = &frame.occlusion_stats_buffer,
                                .size = occlusion_stat_count * sizeof(std::uint32_t),
                                .usage = storage_usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .memory = BufferMemory::device,
                                .debug_name = "renderer.frame_occlusion_stats"},
            OcclusionBufferSpec{.buffer = &frame.occlusion_stats_readback_buffer,
                                .size = occlusion_stat_count * sizeof(std::uint32_t),
                                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .memory = BufferMemory::readback,
                                .debug_name = "renderer.frame_occlusion_stats_readback"},
    };

    for (auto const &spec: occlusion_buffers) {
        auto buffer = create_shared_buffer(context_, BufferCreateInfo{
                                                             .size = spec.size,
                                                             .usage = spec.usage,
                                                             .memory = spec.memory,
                                                             .debug_name = spec.debug_name,
                                                     });

        if (!buffer) {
            return std::unexpected(make_device_error(buffer.error()));
        }

        *spec.buffer = std::move(*buffer);
    }

    auto frustum_planes_buffer =
            create_shared_buffer(context_, BufferCreateInfo{
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

    auto lod_jobs_buffer =
            create_shared_buffer(context_, BufferCreateInfo{
                                                   .size = sizeof(GpuLodJob) * maximum_lod_job_count,
                                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                   .memory = BufferMemory::upload,
                                                   .debug_name = "renderer.frame_lod_jobs",
                                           });

    if (!lod_jobs_buffer) {
        return std::unexpected(make_device_error(lod_jobs_buffer.error()));
    }

    frame.lod_jobs_buffer = std::move(*lod_jobs_buffer);
    frame.lod_jobs.reserve(maximum_lod_job_count);

    {
        auto skin_upload = Buffer::create(context_, BufferCreateInfo{
                                                            .size = skin_input_size(),
                                                            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                            .memory = BufferMemory::upload,
                                                            .debug_name = "renderer.frame_skin_upload",
                                                    });
        if (!skin_upload) {
            return std::unexpected(make_device_error(skin_upload.error()));
        }
        frame.skin_upload_buffer = std::move(*skin_upload);

        auto skin_input = create_shared_buffer(context_, BufferCreateInfo{
                                                                 .size = skin_input_size(),
                                                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                                 .memory = BufferMemory::device,
                                                                 .debug_name = "renderer.frame_skin_input",
                                                         });
        if (!skin_input) {
            return std::unexpected(make_device_error(skin_input.error()));
        }
        frame.skin_input_buffer = std::move(*skin_input);

        auto skin_scratch = create_shared_buffer(context_, BufferCreateInfo{
                                                                   .size = std::max<VkDeviceSize>(
                                                                           skin_scratch_capacity_, 256),
                                                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                                   .memory = BufferMemory::device,
                                                                   .debug_name = "renderer.frame_skin_scratch",
                                                           });
        if (!skin_scratch) {
            return std::unexpected(make_device_error(skin_scratch.error()));
        }
        frame.skin_scratch_buffer = std::move(*skin_scratch);
        frame.skin_jobs.reserve(maximum_skin_jobs_);
    }

    auto lights_buffer = create_shared_buffer(context_, BufferCreateInfo{
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

    auto visible_lights = create_shared_buffer(context_, BufferCreateInfo{
                                                                 .size = visible_lights_size,
                                                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                                 .memory = BufferMemory::device,
                                                                 .debug_name = "renderer.frame_visible_lights",
                                                         });

    if (!visible_lights) {
        return std::unexpected(make_device_error(visible_lights.error()));
    }

    frame.visible_lights_buffer = std::move(*visible_lights);

    auto decals_buffer = create_shared_buffer(context_, BufferCreateInfo{
                                                                .size = sizeof(GpuDecal) * maximum_decal_count,
                                                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                                .memory = BufferMemory::upload,
                                                                .debug_name = "renderer.frame_decals",
                                                        });

    if (!decals_buffer) {
        return std::unexpected(make_device_error(decals_buffer.error()));
    }

    frame.decals_buffer = std::move(*decals_buffer);

    auto local_shadow_tiles = create_shared_buffer(
            context_, BufferCreateInfo{
                              .size = sizeof(local_shadows::GpuShadowTile) * local_shadows::maximum_tiles,
                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              .memory = BufferMemory::upload,
                              .debug_name = "renderer.frame_local_shadow_tiles",
                      });

    if (!local_shadow_tiles) {
        return std::unexpected(make_device_error(local_shadow_tiles.error()));
    }

    frame.local_shadow_tiles_buffer = std::move(*local_shadow_tiles);

    auto local_shadow_planes = create_shared_buffer(
            context_, BufferCreateInfo{
                              .size = sizeof(glm::vec4) * 6 * local_shadows::maximum_tiles,
                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              .memory = BufferMemory::upload,
                              .debug_name = "renderer.frame_local_shadow_planes",
                      });

    if (!local_shadow_planes) {
        return std::unexpected(make_device_error(local_shadow_planes.error()));
    }

    frame.local_shadow_planes_buffer = std::move(*local_shadow_planes);

    auto targets = create_frame_targets(frame_index, extent);

    if (!targets) {
        return std::unexpected(targets.error());
    }

    frame.viewport_target = std::move(targets->viewport_target);

    frame.draw_upload_offset = 0;
    frame.transform_upload_offset = transform_offset;
    frame.indirect_upload_offset = indirect_offset;
    frame.batch_bounds_upload_offset = batch_bounds_offset;
    frame.indirect_commands.reserve(maximum_draw_count_);
    frame.batch_bounds.reserve(maximum_draw_count_);

    return {};
}

auto Renderer::create_ubos() -> std::expected<std::vector<Buffer>, RendererError> {
    std::vector<Buffer> ubos(frames_in_flight);

    for (auto &ubo: ubos) {
        auto maybe_ubo = create_shared_buffer(context_, BufferCreateInfo{
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

    return ubos;
}

auto Renderer::destroy() noexcept -> void {
    debug("[Renderer::destroy] enter");

    // The views' buffers and images come out of the storages torn down below.
    if (!extra_views_.empty()) {
        static_cast<void>(wait_idle());
        extra_views_.clear();
    }

    // Its buffers come out of the allocator torn down below.
    game_gpu_.reset();

    light_icon_overlay_.reset();

    screenshot_->close();

    environment_.destroy();

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

    pass_profiler_->destroy();

    for (auto &ubo: ubos_) {
        ubo.destroy();
    }

    resident_instance_sets_.clear();
    pending_resident_uploads_.clear();

    for (auto &frame: frames_) {
        frame.cluster_stats_readback_buffer.destroy();
        frame.cluster_lights_buffer.destroy();
        frame.decal_clusters_buffer.destroy();
        frame.local_shadow_indirect_buffer.destroy();
        frame.local_shadow_counts_buffer.destroy();
        frame.local_shadow_survivors_buffer.destroy();
        frame.local_shadow_planes_buffer.destroy();
        frame.local_shadow_tiles_buffer.destroy();
        frame.decals_buffer.destroy();
        frame.visible_lights_buffer.destroy();
        frame.lights_buffer.destroy();
        frame.frustum_planes_buffer.destroy();
        frame.visible_transform_buffer.destroy();
        frame.visible_draw_buffer.destroy();
        frame.meshlet_visibility_buffer.destroy();
        frame.occlusion_stats_readback_buffer.destroy();
        frame.occlusion_stats_buffer.destroy();
        frame.merged_indirect_buffer.destroy();
        frame.late_indirect_buffer.destroy();
        frame.cull_chunks_buffer.destroy();
        frame.lod_jobs_buffer.destroy();
        frame.skin_scratch_buffer.destroy();
        frame.skin_input_buffer.destroy();
        frame.skin_upload_buffer.destroy();
        frame.retired_buffers.clear();
        frame.occlusion_candidates_buffer.destroy();
        frame.occlusion_views_buffer.destroy();
        frame.culled_indirect_buffer.destroy();
        frame.batch_bounds_buffer.destroy();
        frame.indirect_buffer.destroy();
        frame.transform_buffer.destroy();
        frame.draw_buffer.destroy();
        frame.upload_buffer.destroy();

        frame.draw_count = 0;
        frame.transform_count = 0;
        frame.indirect_commands.clear();
        frame.batch_bounds.clear();

        frame.indirect_command_count = 0;
        frame.occlusion_stats_pending = false;
        frame.occlusion_stats_active = false;
        frame.meshlet_occlusion_stats_active = false;
        frame.occlusion_active = false;
        frame.meshlet_occlusion_active = false;
        frame.meshlet_visibility_capacity_words = 0;
        frame.meshlet_visibility_words = 0;
        frame.cluster_stats_pending = false;
    }

    transient_allocator_->release_all();

    frames_.clear();

    local_shadow_atlas_.reset();
    local_shadow_allocator_.clear();
    shadow_atlas_.reset();

    hiz_ = HizPyramid{};
    hiz_history_valid_ = false;

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
    local_shadow_atlas_initialized_ = false;
    dynamic_shadow_casters_dirty_ = false;

    model_streamer_.wait_all();

    outline_variants_.clear();
    material_storage_.destroy();
    texture_streamer_.wait_all();
    image_storage_.destroy();
    geometry_arena_.destroy(context_);
    if (shader_prefetch_thread.joinable()) {
        shader_prefetch_thread.join();
    }
    if (compiler_created.load(std::memory_order_acquire)) {
        compiler().destroy();
    }

    clear_submissions();

    model_storage_.destroy();
    mesh_storage_.destroy();

    default_material_handle_ = {};

    forward_pipeline_ = {};
    composite_pipeline_ = {};

    maximum_draw_count_ = 0;
    maximum_submission_count_ = 0;
    maximum_skin_palette_matrices_ = 0;
    maximum_skin_jobs_ = 0;
    skin_scratch_capacity_ = 0;

    hdr_format_ = VK_FORMAT_UNDEFINED;
    depth_format_ = VK_FORMAT_UNDEFINED;

    samples_ = VK_SAMPLE_COUNT_1_BIT;
    pending_samples_ = VK_SAMPLE_COUNT_1_BIT;
    extent_ = {};

    initialized_ = false;
}

auto Renderer::load_model(AssetPath const &path) -> std::expected<ModelHandle, RendererError> {
    if (!initialized_) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    std::size_t const file_hash = std::hash<std::string>{}(path.key());

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
    model_sources_.insert_or_assign(model_source_key(*model_result), path);
    register_model_name(*model_result, path.absolute().filename().string());

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

    auto created = create_model(*imported_model, default_material_handle_);

    // Models loaded here, not through the streamer, need their triangles kept too when the editor asked for them. The
    // triangles are copied now, while the parsed model is at hand, and the hierarchy over them is built on another
    // thread and installed by a later frame.
    if (created && model_streamer_.collecting_pick_geometry()) {
        pending_pick_geometry_.push_back(PendingPickGeometry{
                .model = *created,
                .geometry = std::async(std::launch::async,
                                       [triangles = ModelPickGeometry::collect(cpu_data)]() mutable {
                                           return ModelPickGeometry::build(std::move(triangles.positions),
                                                                           std::move(triangles.indices));
                                       }),
        });
    }

    return created;
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

            // A skinned mesh ignores its node's transform (glTF 2.0): the joint matrices already carry every
            // ancestor, so applying it again would double-rotate e.g. a Z-up to Y-up parent node.
            auto const draw_transform = model.animation != nullptr ? glm::mat4{1.0F} : local_to_model;
            flattened_draws.push_back(ModelDraw{
                    .mesh = imported_meshes[node.mesh_index],
                    .local_transform = draw_transform,
                    .identity_local = draw_transform == glm::mat4{1.0F},
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
            .animation = model.animation,
            .skin_inflate = model.skin_inflate,
    });

    if (!handle) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

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

auto Renderer::model_animation(ModelHandle model) const -> std::shared_ptr<ModelAnimationData const> {
    auto const *slot = model_slot(model);
    return slot != nullptr ? slot->animation : nullptr;
}

auto Renderer::max_skinned_instances(ModelHandle model) const -> std::uint32_t {
    auto const *slot = model_slot(model);

    if (slot == nullptr) {
        return 0;
    }

    VkDeviceSize scratch_per_instance = 0;
    std::uint32_t jobs_per_instance = 0;

    for (auto const &draw: slot->draws) {
        auto const *mesh = mesh_slot(draw.mesh);

        if (mesh == nullptr) {
            continue;
        }

        for (auto const &submesh: mesh->submeshes) {
            auto const &geometry = submesh.lods[0];

            if (geometry.skin.valid()) {
                // Matches the per-job reservation in the draw loop.
                scratch_per_instance +=
                        (VkDeviceSize{geometry.vertices.vertex_count} * sizeof(CompressedModelVertex) + 15U) & ~VkDeviceSize{15U};
                ++jobs_per_instance;
            }
        }
    }

    if (jobs_per_instance == 0 || scratch_per_instance == 0) {
        return 0;
    }

    return static_cast<std::uint32_t>(std::min<VkDeviceSize>(skin_scratch_capacity_ / scratch_per_instance,
                                                              maximum_skin_jobs_ / jobs_per_instance));
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

auto Renderer::model_meshlet_cones(ModelHandle model) const -> std::vector<MeshletCone> {
    std::vector<MeshletCone> result;
    auto const *slot = model_slot(model);

    if (slot == nullptr) {
        return result;
    }

    for (auto const &draw: slot->draws) {
        auto const *mesh = mesh_slot(draw.mesh);

        if (mesh == nullptr) {
            continue;
        }

        auto const axes = glm::mat3{draw.local_transform};
        auto const scale = std::max({glm::length(axes[0]), glm::length(axes[1]), glm::length(axes[2])});

        for (auto const &submesh: mesh->submeshes) {
            auto const &cones = submesh.lods[0].meshlets.cones;

            if (!cones) {
                continue;
            }

            for (auto const &cone: *cones) {
                result.push_back(MeshletCone{
                        .centre = glm::vec3{draw.local_transform * glm::vec4{cone.centre, 1.0F}},
                        .radius = cone.radius * scale,
                        .axis = glm::normalize(axes * cone.axis),
                        .cutoff = cone.cutoff,
                });
            }
        }
    }

    return result;
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
                            std::span<MaterialSlotOverride const> slot_overrides, bool outlined)
        -> std::expected<void, RendererError> {
    if (model_slot(model) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (submitted_model_count() >= maximum_submission_count_) {
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
            .outlined = outlined,
    });
    outlined_submission_count_ += outlined ? 1U : 0U;

    return {};
}

auto Renderer::submit_model(ModelHandle model, glm::mat4 &&transform, MaterialHandle material_override,
                            std::span<MaterialSlotOverride const> slot_overrides, bool outlined)
        -> std::expected<void, RendererError> {
    if (model_slot(model) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (submitted_model_count() >= maximum_submission_count_) {
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
            .outlined = outlined,
    });
    outlined_submission_count_ += outlined ? 1U : 0U;

    return {};
}

auto Renderer::begin_model_submissions(std::size_t count) -> std::span<ModelSubmission> {
    auto const used = submitted_model_count();
    auto const room = used < maximum_submission_count_ ? static_cast<std::size_t>(maximum_submission_count_) - used
                                                       : std::size_t{0};
    count = std::min(count, room);

    opened_submissions_first_ = model_submissions_.size();
    model_submissions_.resize(opened_submissions_first_ + count);
    return std::span{model_submissions_}.subspan(opened_submissions_first_);
}

auto Renderer::set_model_submission_slot_overrides(ModelSubmission &submission,
                                                   std::span<MaterialSlotOverride const> overrides) -> void {
    submission.slot_override_first = static_cast<std::uint32_t>(slot_override_submissions_.size());
    submission.slot_override_count = static_cast<std::uint32_t>(overrides.size());
    slot_override_submissions_.insert(slot_override_submissions_.end(), overrides.begin(), overrides.end());
}

auto Renderer::end_model_submissions(std::size_t skipped, std::size_t outlined)
        -> std::expected<void, RendererError> {
    outlined_submission_count_ += outlined;

    if (skipped == 0) {
        return {};
    }

    auto first_error = std::optional<RendererError>{};
    auto kept = opened_submissions_first_;
    for (auto index = opened_submissions_first_; index < model_submissions_.size(); ++index) {
        auto &submission = model_submissions_[index];
        if (submission.skip) {
            continue;
        }
        if (model_slot(submission.model) == nullptr) {
            first_error = first_error.value_or(make_error(RendererErrorType::invalid_model));
            continue;
        }
        if (kept != index) {
            model_submissions_[kept] = submission;
        }
        ++kept;
    }
    model_submissions_.resize(kept);

    if (first_error) {
        return std::unexpected(*first_error);
    }
    return {};
}

auto Renderer::resident_lod_groups(Submesh const &submesh, MaterialHandle base_material) const noexcept
        -> ResidentLodGroups {
    ResidentLodGroups result;

    for (std::uint32_t lod = 0; lod < lod_count; ++lod) {
        auto const material = material_storage_.material_for_lod(base_material, lod);
        auto const &indices = submesh.lods[lod].indices;

        auto group = result.count;
        for (std::uint32_t existing = 0; existing < result.count; ++existing) {
            auto const representative = result.representative_lod[existing];
            auto const &existing_indices = submesh.lods[representative].indices;
            if (existing_indices.bytes.offset == indices.bytes.offset &&
                existing_indices.index_count == indices.index_count && result.material[existing] == material) {
                group = existing;
                break;
            }
        }

        if (group == result.count) {
            result.representative_lod[group] = lod;
            result.material[group] = material;
            ++result.count;
        }

        result.lod_groups |= group << (8U * lod);
    }

    return result;
}

auto Renderer::submit_resident_instances(ModelHandle model, std::span<glm::mat4 const> transforms,
                                         MaterialHandle material_override, std::uint64_t revision) -> bool {
    auto const *model_data = model_slot(model);

    if (model_data == nullptr || model_data->draws.size() != 1 ||
        !model_data->draws.front().identity_local) {
        return false;
    }

    auto const *mesh = mesh_slot(model_data->draws.front().mesh);
    if (mesh == nullptr || mesh->submeshes.empty()) {
        return false;
    }

    auto const instance_count = static_cast<std::uint64_t>(transforms.size());
    std::uint64_t slots = 0;

    for (auto const &submesh: mesh->submeshes) {
        auto const base = material_override.valid() ? material_override : submesh.material;
        auto const groups = resident_lod_groups(submesh, base);

        for (std::uint32_t group = 0; group < groups.count; ++group) {
            auto const *material = material_storage_.get(groups.material[group]);
            if (material != nullptr && material->alpha_mode == AlphaMode::blend) {
                return false;
            }
        }

        slots += instance_count * groups.count;
    }

    auto const capacity = std::min<std::uint64_t>(maximum_draw_count_, maximum_submission_count_);
    auto const jobs = static_cast<std::uint32_t>(mesh->submeshes.size());
    if (submitted_model_count() + resident_slots_this_frame_ + slots > capacity ||
        resident_jobs_this_frame_ + jobs > maximum_lod_job_count) {
        return false;
    }

    auto set = resident_instance_sets_.find(revision);

    if (set == resident_instance_sets_.end() || set->second.count != transforms.size()) {
        auto const size = static_cast<VkDeviceSize>(transforms.size_bytes());

        auto staging = Buffer::create(context_, BufferCreateInfo{
                                                        .size = size,
                                                        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                        .memory = BufferMemory::upload,
                                                        .debug_name = "renderer.resident_instances_staging",
                                                });
        auto resident = create_shared_buffer(context_, BufferCreateInfo{
                                                               .size = size,
                                                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                               .memory = BufferMemory::device,
                                                               .debug_name = "renderer.resident_instances",
                                                       });

        if (!staging || !resident || !staging->write(0, std::as_bytes(transforms))) {
            return false;
        }

        if (set != resident_instance_sets_.end()) {
            retired_resident_buffers_.push_back(std::move(set->second.transforms));
            resident_instance_sets_.erase(set);
        }

        pending_resident_uploads_.push_back(PendingResidentUpload{
                .staging = std::move(*staging),
                .destination = resident->buffer,
                .size = size,
        });

        set = resident_instance_sets_
                      .emplace(revision,
                               ResidentInstanceSet{
                                       .transforms = std::move(*resident),
                                       .count = static_cast<std::uint32_t>(transforms.size()),
                               })
                      .first;
    }

    set->second.last_used_frame = frame_counter_;

    resident_slots_this_frame_ += slots;
    resident_jobs_this_frame_ += jobs;

    instanced_submissions_.push_back(InstancedSubmission{
            .model = model,
            .material_override = material_override,
            .first_transform = 0,
            .transform_count = static_cast<std::uint32_t>(transforms.size()),
            .model_submission_position = model_submissions_.size(),
            .resident_revision = revision,
    });

    return true;
}

auto Renderer::set_skin_palette(std::span<glm::mat4 const> palette) -> std::expected<void, RendererError> {
    if (palette.size() > maximum_skin_palette_matrices_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    skin_palette_.assign(palette.begin(), palette.end());
    return {};
}

auto Renderer::append_skin_palette(std::span<glm::mat4 const> palette) -> std::expected<std::uint32_t, RendererError> {
    if (palette.size() > maximum_skin_palette_matrices_ - std::min<std::size_t>(skin_palette_.size(),
                                                                                  maximum_skin_palette_matrices_)) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    auto const offset = static_cast<std::uint32_t>(skin_palette_.size());
    skin_palette_.insert(skin_palette_.end(), palette.begin(), palette.end());
    return offset;
}

auto Renderer::submit_model_instances(ModelHandle model, std::span<glm::mat4 const> transforms,
                                      MaterialHandle material_override, std::uint64_t resident_revision,
                                      std::span<std::uint32_t const> palette_offsets)
        -> std::expected<void, RendererError> {
    auto const *model_data = model_slot(model);

    if (model_data == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_model));
    }

    if (!palette_offsets.empty() && palette_offsets.size() != transforms.size()) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    if (model_data->animation != nullptr) {
        resident_revision = 0;
    }

    if (resident_revision != 0 && !transforms.empty() &&
        submit_resident_instances(model, transforms, material_override, resident_revision)) {
        return {};
    }

    if (submitted_model_count() + transforms.size() > maximum_submission_count_) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    if (transforms.empty()) {
        return {};
    }

    instanced_submissions_.push_back(InstancedSubmission{
            .model = model,
            .material_override = material_override,
            .first_transform = static_cast<std::uint32_t>(instance_transforms_.size()),
            .transform_count = static_cast<std::uint32_t>(transforms.size()),
            .model_submission_position = model_submissions_.size(),
            .first_palette_offset = static_cast<std::uint32_t>(instance_palette_offsets_.size()),
            .palette_count = static_cast<std::uint32_t>(palette_offsets.size()),
    });
    instance_transforms_.insert(instance_transforms_.end(), transforms.begin(), transforms.end());
    instance_palette_offsets_.insert(instance_palette_offsets_.end(), palette_offsets.begin(), palette_offsets.end());

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

    retain_material(create_info.far_material);

    if (!debug_name.empty()) {
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

    auto const *previous_info = material_storage_.create_info(handle);
    auto const previous_far_material = previous_info != nullptr ? previous_info->far_material : MaterialHandle{};

    auto result = material_storage_.update_material(handle, create_info);

    if (!result) {
        return std::unexpected(make_material_error(result.error()));
    }

    retain_material(create_info.far_material);
    release_material(previous_far_material);

    mark_shadow_casters_dirty();
    return {};
}

auto Renderer::outline_variant(MaterialHandle source) -> MaterialHandle {
    auto const *source_info = material_storage_.create_info(source);

    if (source_info == nullptr) {
        return source;
    }

    auto info = *source_info;
    info.far_material = MaterialHandle{};
    info.outlined = true;

    auto const found = std::ranges::find(outline_variants_, source, &OutlineVariant::source);

    if (found != outline_variants_.end()) {
        if (found->refreshed_frame != frame_counter_) {
            static_cast<void>(material_storage_.update_material(found->variant, info));
            found->refreshed_frame = frame_counter_;
        }

        return found->variant;
    }

    auto created = create_material(info, {});

    if (!created) {
        warn("Renderer: could not make an outline variant of a material: {}", describe(created.error()));
        return source;
    }

    outline_variants_.push_back(
            OutlineVariant{.source = source, .variant = *created, .refreshed_frame = frame_counter_});

    return *created;
}

auto Renderer::prune_outline_variants() -> void {
    std::erase_if(outline_variants_, [this](OutlineVariant const &entry) {
        if (material_storage_.create_info(entry.source) != nullptr) {
            return false;
        }

        release_material(entry.variant);
        return true;
    });
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
    if (!initialized_ || !handle.valid() || handle == default_material_handle_) {
        return;
    }

    bool const last_reference = material_storage_.ref_count(handle) == 1;

    auto far_material = MaterialHandle{};
    if (auto const *info = material_storage_.create_info(handle); last_reference && info != nullptr) {
        far_material = info->far_material;
    }

    if (auto const result = material_storage_.destroy_material(handle); !result) {
        warn("Renderer::release_material: handle is not a live material");
        return;
    }

    if (last_reference) {
        assets_.materials().unregister(handle);
        mark_shadow_casters_dirty();

        release_material(far_material);
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

auto Renderer::request_texture(AssetPath source_path, TextureRole role, ImageHandle fallback, std::string debug_name)
        -> ImageHandle {
    auto const handle =
            texture_streamer_.request(image_storage_, std::move(source_path), role, fallback, FlyString{debug_name});

    static_cast<void>(assets_.textures().register_asset(std::move(debug_name), handle));

    return handle;
}

namespace {

    constexpr std::uint32_t cull_flag_late_union_meshlet_batches = 1U;
    constexpr std::uint32_t cull_stage_shift = 8U;

    constexpr std::uint32_t cull_dispatch_width = 65'535;

    auto dispatch_linear(VkCommandBuffer command_buffer, std::uint32_t group_count) -> void {
        if (group_count == 0) {
            return;
        }
        auto const width = std::min(group_count, cull_dispatch_width);
        auto const height = (group_count + width - 1) / width;
        vkCmdDispatch(command_buffer, width, height, 1);
    }

    auto record_compute_barrier(VkCommandBuffer command_buffer) -> void {
        VkMemoryBarrier2 const between_stages{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        };
        VkDependencyInfo const dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = nullptr,
                .dependencyFlags = 0,
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &between_stages,
                .bufferMemoryBarrierCount = 0,
                .pBufferMemoryBarriers = nullptr,
                .imageMemoryBarrierCount = 0,
                .pImageMemoryBarriers = nullptr,
        };
        vkCmdPipelineBarrier2(command_buffer, &dependency);
    }

    auto record_cull_stages(VkCommandBuffer command_buffer, VkPipelineLayout layout, CullPushConstants pc) -> void {
        auto const base_flags = pc.flags;
        constexpr std::array stages{0U, 1U, 2U};

        for (auto const stage: stages) {
            if (stage != 0U) {
                record_compute_barrier(command_buffer);
            }

            pc.flags = base_flags | (stage << cull_stage_shift);
            vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);

            dispatch_linear(command_buffer, stage == 1U ? pc.batch_count : pc.chunk_count);
        }
    }

    [[nodiscard]] auto validate_submesh_lods(std::array<MeshGeometry, lod_count> const &lods)
            -> std::expected<void, RendererError> {
        auto const &lod0 = lods[0];

        if (!lod0.vertices.bytes.valid() || !lod0.indices.bytes.valid() || lod0.vertices.vertex_count == 0 ||
            lod0.indices.index_count == 0) {
            return std::unexpected(make_error(RendererErrorType::invalid_argument));
        }

        if (!std::ranges::all_of(lods, [](MeshGeometry const &lod) { return lod.meshlets.valid(); })) {
            return std::unexpected(make_error(RendererErrorType::invalid_argument));
        }

        for (auto const &lod: lods) {
            if (lod.skin.valid() && lod.skin.size != VkDeviceSize{lod.vertices.vertex_count} * sizeof(GpuSkinVertex)) {
                return std::unexpected(make_error(RendererErrorType::invalid_argument));
            }
        }

        auto stride = index_stride(lod0.indices.index_type);

        if (!stride) {
            return std::unexpected(stride.error());
        }

        if (lod0.indices.bytes.offset % *stride != 0) {
            return std::unexpected(make_error(RendererErrorType::invalid_argument));
        }

        return {};
    }

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
        if (auto valid = validate_submesh_lods(submesh_info.lods); !valid) {
            return std::unexpected(valid.error());
        }

        if (material_storage_.get(submesh_info.material) == nullptr) {
            return std::unexpected(make_error(RendererErrorType::invalid_material));
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
            retire_once(lod.skin);
            retire_once(lod.indices.bytes);
            retire_once(lod.meshlets.descriptors);
            retire_once(lod.meshlets.data);
        }
    }

}

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

auto Renderer::update_submesh_geometry(MeshHandle mesh, std::uint32_t submesh_index, MeshGeometry const &geometry)
        -> std::expected<void, RendererError> {
    auto *slot = mesh_storage_.get(mesh);

    if (slot == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_mesh));
    }

    if (submesh_index >= slot->submeshes.size()) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    std::array<MeshGeometry, lod_count> lods{};
    lods.fill(geometry);

    if (auto valid = validate_submesh_lods(lods); !valid) {
        return std::unexpected(valid.error());
    }

    slot->submeshes[submesh_index].lods = lods;

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
    if (!initialized_) {
        return;
    }

    static_cast<void>(destroy_model(handle));
}

auto Renderer::register_model_name(ModelHandle handle, std::string_view name) -> void {
    static_cast<void>(assets_.models().register_asset(std::string{name}, handle));
}

auto Renderer::register_model_source(ModelHandle handle, AssetPath const &source) -> void {
    if (model_storage_.get(handle) == nullptr) {
        return;
    }

    model_cache_.try_emplace(std::hash<std::string>{}(source.key()), handle);
    model_sources_.insert_or_assign(model_source_key(handle), source);
}

auto Renderer::register_model_pick_geometry(ModelHandle handle, std::shared_ptr<ModelPickGeometry const> geometry)
        -> void {
    if (auto *slot = model_slot(handle)) {
        slot->pick_geometry = std::move(geometry);
    }
}

auto Renderer::model_pick_geometry(ModelHandle handle) const -> std::shared_ptr<ModelPickGeometry const> {
    auto const *slot = model_slot(handle);
    return slot != nullptr ? slot->pick_geometry : nullptr;
}

auto Renderer::cached_model(AssetPath const &source) const -> ModelHandle {
    auto const it = model_cache_.find(std::hash<std::string>{}(source.key()));
    return it != model_cache_.end() && model_storage_.get(it->second) != nullptr ? it->second : ModelHandle{};
}

auto Renderer::model_source(ModelHandle handle) const noexcept -> AssetPath const * {
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

auto Renderer::set_environment(SceneEnvironment const &environment) -> void {
    environment_.set_environment(environment);

    ambient_intensity_ = environment.ambient_intensity;

    fog_settings_ = FogSettings{
            .enabled = environment.fog.enabled,
            .colour = environment.fog.colour,
            .extinction = environment.fog.extinction,
            .inscattering = environment.fog.inscattering,
    };

    if (!environment.sun_drives_directional_light) {
        return;
    }

    auto const &sun = environment.sun;

    auto const elevation = glm::radians(std::clamp(sun.elevation_degrees, 5.0F, 89.0F));
    auto const azimuth = glm::radians(sun.azimuth_degrees);

    DirectionalLight light{
            .direction = glm::vec3{std::cos(elevation) * std::cos(azimuth), std::sin(elevation),
                                   std::cos(elevation) * std::sin(azimuth)},
            .colour = sun.colour,
            .intensity = sun.intensity,
    };

    if (environment.source == EnvironmentSource::procedural_sky) {
        auto const real_elevation = glm::radians(sun.elevation_degrees);

        if (sun.derive_colour_from_sky) {
            light.colour *= sun_transmittance(real_elevation, sun.turbidity);
        }

        auto const fade = std::clamp((sun.elevation_degrees + 2.0F) / 7.0F, 0.0F, 1.0F);
        light.intensity *= fade * fade * (3.0F - (2.0F * fade));
    }

    if (light.direction != light_.direction || light.colour != light_.colour || light.intensity != light_.intensity) {
        light_ = light;
    }
}

auto Renderer::submit_decal(Decal const &decal) -> std::expected<void, RendererError> {
    if (decal_submissions_.size() >= maximum_decal_count) {
        return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
    }

    if (material_storage_.get(decal.material) == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_material));
    }

    decal_submissions_.push_back(decal);

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

// What the stages that turn the frame's submissions into batches (build_batches, classify_batches, plan_shadow_batches,
// emit_batches) hand on to the rest of prepare_frame.
struct Renderer::BatchBuild {
    std::uint32_t submitted_triangle_count = 0;
    std::uint64_t resident_instance_count = 0;
    std::uint64_t resident_reserved_slots = 0;
    MeshletVisibilityLayout meshlet_layout;
    std::uint64_t shadow_scene_signature = 0;
    bool has_animated_shadow_casters = false;
    std::future<void> blend_sort_future;
    // A water material is among the blended batches (it needs the scene copy).
    bool has_water = false;
    // The transforms the chunks made for their instances; batches point into them until the frame is emitted.
    std::vector<std::deque<glm::mat4>> computed_chunks;
};

// A batch as one chunk of the frame's submissions saw it: the instances of the chunk that fall in the batch, in order.
struct Renderer::LocalBatch {
    BatchKey key{};
    MeshHandle mesh{};
    std::uint32_t submesh_index = 0;
    MaterialHandle material{};
    std::uint32_t lod_index = 0;
    bool skinned = false;

    std::vector<glm::mat4 const *> transforms;
    std::vector<std::uint32_t> palette_offsets;
};

// How the draws of a model resolve to batches, remembered for a chunk so the same model with the same material
// override and outline is not worked out again for every object that uses it. Per submesh it holds, for each LOD, the
// chunk's batch once it has been looked up. Objects with slot overrides do not use it.
class Renderer::BatchRecipes {
public:
    struct Submesh {
        std::array<LocalBatch *, lod_count> batch{};
    };

    struct Draw {
        MeshHandle mesh{};
        MeshSlotData const *data = nullptr;
        bool identity_local = true;
        glm::mat4 local{1.0F};
        std::vector<Submesh> submeshes;
    };

    struct Recipe {
        std::vector<Draw> draws;
    };

    // The recipe for the model, or nullptr when the model does not exist (nothing to draw), or an error when one of
    // its meshes does not.
    [[nodiscard]] auto find(Renderer &renderer, ModelHandle model, MaterialHandle material_override, bool outlined)
            -> std::expected<Recipe *, RendererError> {
        if (table_.empty()) {
            table_.assign(initial_capacity, 0U);
        }

        auto const key = Key{model.index, model.generation, material_override.index, material_override.generation,
                             outlined};
        auto const mask = table_.size() - 1;
        auto slot = hash(key) & mask;
        while (table_[slot] != 0U) {
            auto &entry = entries_[table_[slot] - 1U];
            if (entry.key == key) {
                return entry.recipe ? &*entry.recipe : nullptr;
            }
            slot = (slot + 1U) & mask;
        }

        auto recipe = std::optional<Recipe>{};
        if (auto const *model_data = renderer.model_slot(model); model_data != nullptr) {
            recipe.emplace();
            for (auto const &model_draw: model_data->draws) {
                auto const *mesh = renderer.mesh_slot(model_draw.mesh);
                if (mesh == nullptr) {
                    return std::unexpected(make_error(RendererErrorType::invalid_mesh));
                }
                recipe->draws.push_back(Draw{.mesh = model_draw.mesh,
                                             .data = mesh,
                                             .identity_local = model_draw.identity_local,
                                             .local = model_draw.local_transform,
                                             .submeshes = std::vector<Submesh>(mesh->submeshes.size())});
            }
        }

        entries_.push_back(Entry{key, std::move(recipe)});
        table_[slot] = static_cast<std::uint32_t>(entries_.size());
        if (entries_.size() * 2 > table_.size()) {
            grow();
        }
        return entries_.back().recipe ? &*entries_.back().recipe : nullptr;
    }

private:
    struct Key {
        std::uint32_t model_index;
        std::uint32_t model_generation;
        std::uint32_t override_index;
        std::uint32_t override_generation;
        bool outlined;

        auto operator==(Key const &) const noexcept -> bool = default;
    };

    struct Entry {
        Key key;
        std::optional<Recipe> recipe;
    };

    static constexpr std::size_t initial_capacity = 64;

    [[nodiscard]] static auto hash(Key const &key) noexcept -> std::size_t {
        auto value = (std::uint64_t{key.model_index} << 32U | key.model_generation) * 0x9E3779B97F4A7C15ULL;
        value ^= (std::uint64_t{key.override_index} << 32U | key.override_generation) * 0xC2B2AE3D27D4EB4FULL;
        value ^= key.outlined ? 0x165667B19E3779F9ULL : 0ULL;
        value ^= value >> 32U;
        return static_cast<std::size_t>(value);
    }

    auto grow() -> void {
        table_.assign(table_.size() * 2, 0U);
        auto const mask = table_.size() - 1;
        for (auto index = std::size_t{0}; index < entries_.size(); ++index) {
            auto slot = hash(entries_[index].key) & mask;
            while (table_[slot] != 0U) {
                slot = (slot + 1U) & mask;
            }
            table_[slot] = static_cast<std::uint32_t>(index + 1U);
        }
    }

    std::vector<std::uint32_t> table_;
    std::vector<Entry> entries_;
};

// The work of batching one range of the frame's model submissions, free of the shared batch table so that ranges can
// be batched on different threads and merged afterwards in order (see build_batches).
class Renderer::BatchChunk {
public:
    // The chunk's batch for `key`, made the first time the chunk needs it.
    auto local_batch(Renderer &renderer, BatchKey const &key, MeshHandle mesh, std::uint32_t submesh_index,
                     MaterialHandle material, std::uint32_t lod_index) -> LocalBatch & {
        if (auto const found = lookup.find(key); found != lookup.end()) {
            return *found->second;
        }

        auto &batch = batches.emplace_back();
        batch.key = key;
        batch.mesh = mesh;
        batch.submesh_index = submesh_index;
        batch.material = material;
        batch.lod_index = lod_index;
        batch.transforms.reserve(instance_reserve);
        auto const *batch_mesh = renderer.mesh_slot(mesh);
        batch.skinned = batch_mesh != nullptr && submesh_index < batch_mesh->submeshes.size() &&
                        batch_mesh->submeshes[submesh_index].lods[lod_index].skinned();
        lookup.emplace(key, &batch);
        return batch;
    }

    // How many instances each batch of the chunk gets room for when it is made, so that the lists do not grow by
    // repeated doubling (which costs more than the batching in a big chunk). A guess from the size of the chunk: a
    // batch holds a fraction of it, and an over-estimate only costs memory that is released with the chunk.
    std::size_t instance_reserve = 0;

    // In the order the chunk first needed them.
    std::deque<LocalBatch> batches;
    std::unordered_map<BatchKey, LocalBatch *, BatchKeyHash> lookup;
    // Transforms made for the chunk's instances (a model's local transform times the object's); the batches point
    // at them, so they live as long as the frame's batches do.
    std::deque<glm::mat4> computed;
    BatchRecipes recipes;
    std::optional<RendererError> error;
};

// Batches model submissions [first, last) into `chunk`.
auto Renderer::build_chunk(BatchChunk &chunk, std::size_t first, std::size_t last, glm::vec3 const &camera_position)
        -> void {
    constexpr auto max_instance_reserve = std::size_t{4096};
    chunk.instance_reserve = std::min((last - first) / 8, max_instance_reserve);

    auto const push = [](LocalBatch &batch, glm::mat4 const *transform) {
        batch.transforms.push_back(transform);
        if (batch.skinned) {
            batch.palette_offsets.push_back(BatchEntry::no_palette);
        }
    };

    for (auto position = first; position < last; ++position) {
        auto const &model_submission = model_submissions_[position];

        if (model_submission.slot_override_count == 0) {
            auto const recipe = chunk.recipes.find(*this, model_submission.model, model_submission.material_override,
                                                   model_submission.outlined);
            if (!recipe) {
                chunk.error = recipe.error();
                return;
            }
            if (*recipe == nullptr) {
                continue;
            }

            for (auto &draw: (*recipe)->draws) {
                auto const *instance_transform = &model_submission.transform;
                if (!draw.identity_local) {
                    instance_transform = &chunk.computed.emplace_back(model_submission.transform * draw.local);
                }
                auto const lod_index = lod_index_for(camera_position, glm::vec3((*instance_transform)[3]));

                for (std::uint32_t submesh_index = 0; submesh_index < draw.submeshes.size(); ++submesh_index) {
                    auto *&batch = draw.submeshes[submesh_index].batch[lod_index];
                    if (batch == nullptr) {
                        auto const &submesh = draw.data->submeshes[submesh_index];
                        auto material = model_submission.material_override.valid() ? model_submission.material_override
                                                                                   : submesh.material;
                        material = material_storage_.material_for_lod(material, lod_index);

                        if (model_submission.outlined) {
                            material = outline_variant(material);
                        }

                        batch = &chunk.local_batch(*this,
                                                   BatchKey{
                                                           .mesh_index = draw.mesh.index,
                                                           .submesh_index = submesh_index,
                                                           .material_index = material_storage_.gpu_index(material),
                                                           .lod_index = lod_index,
                                                   },
                                                   draw.mesh, submesh_index, material, lod_index);
                    }
                    push(*batch, instance_transform);
                }
            }
            continue;
        }

        auto const *model = model_slot(model_submission.model);
        if (model == nullptr) {
            continue;
        }

        auto const slot_overrides = std::span{slot_override_submissions_}.subspan(model_submission.slot_override_first,
                                                                                  model_submission.slot_override_count);

        for (auto const &model_draw: model->draws) {
            auto const *mesh = mesh_slot(model_draw.mesh);
            if (mesh == nullptr) {
                chunk.error = make_error(RendererErrorType::invalid_mesh);
                return;
            }
            auto const *instance_transform = &model_submission.transform;
            if (!model_draw.identity_local) {
                instance_transform = &chunk.computed.emplace_back(model_submission.transform * model_draw.local_transform);
            }
            auto const lod_index = lod_index_for(camera_position, glm::vec3((*instance_transform)[3]));
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
                material = material_storage_.material_for_lod(material, lod_index);

                if (model_submission.outlined) {
                    material = outline_variant(material);
                }

                auto &batch = chunk.local_batch(*this,
                                                BatchKey{
                                                        .mesh_index = model_draw.mesh.index,
                                                        .submesh_index = submesh_index,
                                                        .material_index = material_storage_.gpu_index(material),
                                                        .lod_index = lod_index,
                                                },
                                                model_draw.mesh, submesh_index, material, lod_index);
                push(batch, instance_transform);
            }
        }
    }
}

// Groups the frame's model, instanced and mesh submissions into batches (one per mesh, submesh, material and LOD), each
// with the transforms of its instances in submission order, and opens the batches for the frame.
auto Renderer::build_batches(RendererFrame &frame, glm::vec3 const &camera_position, BatchBuild &build)
        -> std::expected<void, RendererError> {
    auto const select_lod_index = [&camera_position](glm::vec3 const &instance_position) -> std::uint32_t {
        return lod_index_for(camera_position, instance_position);
    };

    auto const batch_for = [this](BatchKey const &key, MeshHandle mesh, std::uint32_t submesh_index,
                                  MaterialHandle material, std::uint32_t lod_index) -> BatchEntry & {
        auto iterator = batches_.try_emplace(key).first;
        auto &batch = iterator->second;

        if (batch.frame_stamp != batch_frame_) {
            batch.mesh = mesh;
            batch.submesh_index = submesh_index;
            batch.material = material;
            batch.lod_index = lod_index;
            batch.lod_job = BatchEntry::no_lod_job;
            batch.lod_group = 0;
            batch.resident_capacity = 0;
            batch.transforms.clear();
            batch.palette_offsets.clear();
            auto const *batch_mesh = mesh_slot(mesh);
            batch.skinned = batch_mesh != nullptr && submesh_index < batch_mesh->submeshes.size() &&
                            batch_mesh->submeshes[submesh_index].lods[lod_index].skinned();
            batch.frame_stamp = batch_frame_;
            active_batches_.push_back(&batch);
        }
        return batch;
    };

    auto const append_batch_transform = [&batch_for](BatchKey const &key, MeshHandle mesh, std::uint32_t submesh_index,
                                                     MaterialHandle material, std::uint32_t lod_index,
                                                     glm::mat4 const *transform) {
        auto &batch = batch_for(key, mesh, submesh_index, material, lod_index);
        batch.transforms.push_back(transform);
        if (batch.skinned) {
            batch.palette_offsets.push_back(BatchEntry::no_palette);
        }
    };

    auto const lod_slot_count = lod_distances.size() + 1;
    std::vector<BatchEntry *> instanced_batch_cache;
    std::vector<MaterialHandle> instanced_materials;

    std::uint64_t resident_instance_count = 0;
    std::uint64_t resident_reserved_slots = 0;

    auto const append_resident = [&](InstancedSubmission const &instanced) -> bool {
        auto const *model = model_slot(instanced.model);
        auto const set = resident_instance_sets_.find(instanced.resident_revision);
        if (model == nullptr || set == resident_instance_sets_.end()) {
            return true;
        }

        auto const &model_draw = model->draws.front();
        auto const *mesh = mesh_slot(model_draw.mesh);
        if (mesh == nullptr) {
            return false;
        }

        auto const instance_count = set->second.count;
        resident_instance_count += instance_count;

        for (std::uint32_t submesh_index = 0; submesh_index < mesh->submeshes.size(); ++submesh_index) {
            auto const &submesh = mesh->submeshes[submesh_index];
            auto const base = instanced.material_override.valid() ? instanced.material_override : submesh.material;
            auto const groups = resident_lod_groups(submesh, base);

            auto const job_index = static_cast<std::uint32_t>(frame.lod_jobs.size());
            frame.lod_jobs.push_back(GpuLodJob{
                    .transforms_address = set->second.transforms.device_address,
                    .instance_count = instance_count,
                    .first_chunk = frame.lod_chunk_count,
                    .group_count = groups.count,
                    .lod_groups = groups.lod_groups,
            });
            frame.lod_chunk_count += (instance_count + cull_chunk_size - 1) / cull_chunk_size;

            for (std::uint32_t group = 0; group < groups.count; ++group) {
                auto const lod_index = groups.representative_lod[group];
                auto const material = groups.material[group];
                auto &batch = batch_for(
                        BatchKey{
                                .mesh_index = model_draw.mesh.index,
                                .submesh_index = submesh_index,
                                .material_index = material_storage_.gpu_index(material),
                                .lod_index = lod_index,
                                .lod_job_key = job_index + 1,
                        },
                        model_draw.mesh, submesh_index, material, lod_index);
                batch.lod_job = job_index;
                batch.lod_group = group;
                batch.resident_capacity = instance_count;
                resident_reserved_slots += instance_count;
            }
        }

        return true;
    };

    auto const append_instanced = [&](InstancedSubmission const &instanced) -> bool {
        if (instanced.resident_revision != 0) {
            return append_resident(instanced);
        }

        auto const *model = model_slot(instanced.model);
        if (model == nullptr) {
            return true;
        }

        auto const transforms =
                std::span{instance_transforms_}.subspan(instanced.first_transform, instanced.transform_count);
        auto const palette_offsets =
                std::span{instance_palette_offsets_}.subspan(instanced.first_palette_offset, instanced.palette_count);

        auto const joint_count =
                model->animation != nullptr ? static_cast<std::uint64_t>(model->animation->skeleton.joint_count()) : 0U;

        for (auto const &model_draw: model->draws) {
            auto const *mesh = mesh_slot(model_draw.mesh);
            if (mesh == nullptr) {
                return false;
            }

            auto const submesh_count = mesh->submeshes.size();
            instanced_materials.clear();
            for (auto const &submesh: mesh->submeshes) {
                instanced_materials.push_back(instanced.material_override.valid() ? instanced.material_override
                                                                                  : submesh.material);
            }
            instanced_batch_cache.assign(submesh_count * lod_slot_count, nullptr);

            auto const identity_local = model_draw.identity_local;

            for (std::size_t instance_index = 0; instance_index < transforms.size(); ++instance_index) {
                auto const &transform = transforms[instance_index];
                auto palette_offset = BatchEntry::no_palette;
                if (instance_index < palette_offsets.size()) {
                    palette_offset = palette_offsets[instance_index];
                    if (palette_offset == BatchEntry::no_palette ||
                        std::uint64_t{palette_offset} + joint_count > skin_palette_.size() || joint_count == 0) {
                        palette_offset = BatchEntry::no_palette;
                    }
                }
                auto const *instance_transform = &transform;
                if (!identity_local) {
                    instance_transform = &computed_transforms_.emplace_back(transform * model_draw.local_transform);
                }
                auto const lod_index = select_lod_index(glm::vec3((*instance_transform)[3]));

                for (std::uint32_t submesh_index = 0; submesh_index < submesh_count; ++submesh_index) {
                    auto *&batch = instanced_batch_cache[submesh_index * lod_slot_count + lod_index];
                    if (batch == nullptr) {
                        auto const material =
                                material_storage_.material_for_lod(instanced_materials[submesh_index], lod_index);
                        batch = &batch_for(
                                BatchKey{
                                        .mesh_index = model_draw.mesh.index,
                                        .submesh_index = submesh_index,
                                        .material_index = material_storage_.gpu_index(material),
                                        .lod_index = lod_index,
                                },
                                model_draw.mesh, submesh_index, material, lod_index);
                    }
                    batch->transforms.push_back(instance_transform);
                    if (batch->skinned) {
                        batch->palette_offsets.push_back(palette_offset);
                    }
                }
            }
        }
        return true;
    };

    auto next_instanced = std::size_t{0};
    auto const append_instanced_before = [&](std::size_t model_submission_position) -> bool {
        while (next_instanced < instanced_submissions_.size() &&
               instanced_submissions_[next_instanced].model_submission_position <= model_submission_position) {
            if (!append_instanced(instanced_submissions_[next_instanced])) {
                return false;
            }
            ++next_instanced;
        }
        return true;
    };

    // The model submissions are batched in chunks (on several threads when there are many) and the chunks merged in
    // order: each chunk's batches in the order it first needed them, so the batches come out in the order the serial
    // loop would have made them, with their instances in submission order. An instanced submission sits between two
    // runs of model submissions, so a run ends at the next one. Outline variants are made on demand, which only a
    // single thread may do: with any submission outlined, the whole frame goes through one chunk.
    constexpr auto parallel_batching_threshold = std::size_t{4096};
    auto const model_count = model_submissions_.size();
    auto const any_outlined = outlined_submission_count_ != 0;

    auto const batch_models = [&](std::size_t first, std::size_t last) -> std::expected<void, RendererError> {
        auto chunks = std::vector<std::unique_ptr<BatchChunk>>{};

        if (last - first < parallel_batching_threshold || any_outlined) {
            auto &chunk = chunks.emplace_back(std::make_unique<BatchChunk>());
            build_chunk(*chunk, first, last, camera_position);
        } else {
            struct Finished {
                std::size_t first;
                std::unique_ptr<BatchChunk> chunk;
            };
            auto finished = std::vector<Finished>{};
            auto finished_mutex = std::mutex{};

            thread_pool()
                    .submit_blocks(first, last,
                                   [&](std::size_t block_first, std::size_t block_last) {
                                       auto chunk = std::make_unique<BatchChunk>();
                                       build_chunk(*chunk, block_first, block_last, camera_position);
                                       auto const lock = std::scoped_lock{finished_mutex};
                                       finished.push_back(Finished{block_first, std::move(chunk)});
                                   })
                    .wait();

            std::ranges::sort(finished, {}, &Finished::first);
            for (auto &done: finished) {
                chunks.push_back(std::move(done.chunk));
            }
        }

        for (auto &chunk: chunks) {
            if (chunk->error) {
                return std::unexpected(*chunk->error);
            }

            for (auto const &local: chunk->batches) {
                auto &batch = batch_for(local.key, local.mesh, local.submesh_index, local.material, local.lod_index);
                batch.transforms.insert(batch.transforms.end(), local.transforms.begin(), local.transforms.end());
                batch.palette_offsets.insert(batch.palette_offsets.end(), local.palette_offsets.begin(),
                                             local.palette_offsets.end());
            }
            build.computed_chunks.push_back(std::move(chunk->computed));
        }
        return {};
    };

    for (auto position = std::size_t{0};;) {
        if (!append_instanced_before(position)) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }
        if (position == model_count) {
            break;
        }

        auto end = model_count;
        if (next_instanced < instanced_submissions_.size()) {
            end = std::min(end, instanced_submissions_[next_instanced].model_submission_position);
        }

        if (auto batched = batch_models(position, end); !batched) {
            clear_submissions();
            return std::unexpected(batched.error());
        }
        position = end;
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
            auto const material = material_storage_.material_for_lod(
                    submission.material_override.valid() ? submission.material_override : submesh.material, lod_index);
            auto const key = BatchKey{
                    .mesh_index = submission.mesh.index,
                    .submesh_index = submesh_index,
                    .material_index = material_storage_.gpu_index(material),
                    .lod_index = lod_index,
            };

            append_batch_transform(key, submission.mesh, submesh_index, material, lod_index, &submission.transform);
        }
    }


    build.resident_instance_count = resident_instance_count;
    build.resident_reserved_slots = resident_reserved_slots;
    return {};
}

// Sorts the frame's batches by how they are drawn (opaque, double sided, alpha-masked, blended), and starts ordering
// the blended ones back to front.
auto Renderer::classify_batches(glm::vec3 const &camera_position, BatchBuild &build)
        -> std::expected<void, RendererError> {
    opaque_batches_.reserve(active_batches_.size());
    double_sided_batches_.reserve(active_batches_.size());
    mask_batches_.reserve(active_batches_.size());
    blend_batches_.reserve(active_batches_.size());

    for (auto const *batch: active_batches_) {
        auto const *material = material_storage_.get(batch->material);
        auto const alpha_mode = material != nullptr ? material->alpha_mode : AlphaMode::opaque;

        switch (alpha_mode) {
            case AlphaMode::opaque:
                if (material != nullptr && (material->flags & GpuMaterial::flag_double_sided) != 0) {
                    double_sided_batches_.push_back(batch);
                } else {
                    opaque_batches_.push_back(batch);
                }
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
                auto const local_radius = glm::length(submesh.bounds_max - submesh.bounds_min) * 0.5F;
                auto world_centre = glm::vec3{0.0F};
                auto world_radius = local_radius;
                if (!batch->transforms.empty()) {
                    auto const &model = *batch->transforms.front();
                    world_centre = glm::vec3(model * glm::vec4(local_centre, 1.0F));
                    auto const max_scale = std::sqrt(std::max({glm::dot(glm::vec3(model[0]), glm::vec3(model[0])),
                                                               glm::dot(glm::vec3(model[1]), glm::vec3(model[1])),
                                                               glm::dot(glm::vec3(model[2]), glm::vec3(model[2]))}));
                    world_radius = local_radius * max_scale;
                }

                auto const flags = material != nullptr ? material->flags : 0U;
                if ((flags & GpuMaterial::flag_water) != 0) {
                    build.has_water = true;
                }

                blend_batches_.push_back(PendingBlendBatch{
                        .entry = batch,
                        .order =
                                blend_order::Entry{
                                        .camera_distance = glm::length(world_centre - camera_position),
                                        .bounds_radius = world_radius,
                                        .stable_id = {material_storage_.gpu_index(batch->material), batch->mesh.index,
                                                      batch->submesh_index, batch->lod_index},
                                        .water = (flags & GpuMaterial::flag_water) != 0,
                                },
                        .double_sided = (flags & GpuMaterial::flag_double_sided) != 0,
                });

                break;
            }
        }
    }

    constexpr std::size_t parallel_blend_sort_threshold = 4'096;

    auto sort_blend_batches = [this] {
        std::ranges::sort(blend_batches_, [](PendingBlendBatch const &lhs, PendingBlendBatch const &rhs) {
            return blend_order::draws_before(lhs.order, rhs.order);
        });
    };

    if (blend_batches_.size() >= parallel_blend_sort_threshold) {
        build.blend_sort_future = thread_pool().submit_task(sort_blend_batches);
    } else {
        sort_blend_batches();
    }


    return {};
}

// The furthest shadow cascade a batch's material casts into, or -1 if it casts none.
auto Renderer::batch_max_shadow_cascade(BatchEntry const *batch) const noexcept -> std::int32_t {
    auto const *material = material_storage_.get(batch->material);
    auto const cascade = material != nullptr ? material->max_shadow_cascade : shadow_cascade_count - 1;
    return cascade == GpuMaterial::no_shadow_cascade ? -1 : static_cast<std::int32_t>(cascade);
}

// The shadow side of the batches: a signature of what casts shadows (so a static scene can reuse its shadow maps),
// whether anything casting is animated, and the batches ordered by the furthest cascade they reach.
auto Renderer::plan_shadow_batches(RendererFrame &frame, BatchBuild &build) -> void {
    std::uint64_t current_shadow_scene_signature = 0;
    bool has_animated_shadow_casters = false;
    for (auto const *batch: active_batches_) {
        auto const *material = material_storage_.get(batch->material);
        auto const alpha_mode = material != nullptr ? material->alpha_mode : AlphaMode::opaque;
        auto const max_cascade = batch_max_shadow_cascade(batch);

        if (alpha_mode == AlphaMode::blend || max_cascade < 0) {
            continue;
        }

        // The LOD is left out, and so is the batch count: an instance that crosses an LOD threshold as the camera moves
        // moves between batches of the same mesh, submesh and material, which would otherwise redraw every cascade for
        // a change the shadow maps barely show. Weighting each key by its instance count makes the sum the same however
        // the instances are split between LODs.
        std::uint64_t batch_signature = 0xcbf29ce484222325ULL;
        batch_signature = shadow_signature_combine(batch_signature, batch->mesh.index);
        batch_signature = shadow_signature_combine(batch_signature, batch->submesh_index);
        batch_signature = shadow_signature_combine(batch_signature, material_storage_.gpu_index(batch->material));
        batch_signature = shadow_signature_combine(batch_signature, static_cast<std::uint64_t>(max_cascade));

        current_shadow_scene_signature += shadow_signature_mix(batch_signature) * batch->transforms.size();

        has_animated_shadow_casters = has_animated_shadow_casters ||
                                      (material != nullptr && std::abs(material->wind_strength) > 1e-6F) ||
                                      batch->lod_job != BatchEntry::no_lod_job ||
                                      batch->skinned;
    }

    auto const order_shadow_batches = [this](auto &batches) {
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
    frame.shadow_double_sided_indirect_count = order_shadow_batches(double_sided_batches_);
    frame.shadow_mask_indirect_count = order_shadow_batches(mask_batches_);

    build.shadow_scene_signature = current_shadow_scene_signature;
    build.has_animated_shadow_casters = has_animated_shadow_casters;
}

// Writes the batches out for the GPU: each instance's transform and draw record into the frame's upload buffer, and
// one indirect command and cull bounds per batch, in the order opaque, double sided, masked, blended.
auto Renderer::emit_batches(RendererFrame &frame, BatchBuild &build) -> std::expected<void, RendererError> {
    auto &meshlet_layout = build.meshlet_layout;
    auto &submitted_triangle_count = build.submitted_triangle_count;

    // The per-instance copies of a batch (its transforms and its draw records) are not made while the batch is planned:
    // each batch's slice of the upload buffer is already known then, so all the copies are made together afterwards,
    // on several threads when there are many. Batches that need skinning are the exception, as they fill the shared
    // skin job list in order.
    struct CopyTask {
        BatchEntry const *batch = nullptr;
        std::byte *transform_out = nullptr;
        std::byte *draw_out = nullptr;
        std::uint32_t instance_count = 0;
        GpuDraw draw{};
        bool visibility = false;
        std::uint64_t first_meshlet_bit = 0;
        std::uint32_t meshlet_count = 0;
    };
    auto copy_tasks = std::vector<CopyTask>{};

    auto const emit_batch = [this, &frame, &submitted_triangle_count, &meshlet_layout,
                             &copy_tasks](BatchEntry const &batch, bool const allocate_meshlet_bits,
                                          blend_order::Faces const faces =
                                                  blend_order::Faces::both) -> std::expected<void, RendererError> {
        auto const *mesh = mesh_slot(batch.mesh);

        if (mesh == nullptr) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }

        auto const &submesh = mesh->submeshes[batch.submesh_index];
        auto const &geometry = submesh.lods[batch.lod_index];
        auto const resident = batch.lod_job != BatchEntry::no_lod_job;

        auto const instance_count =
                resident ? batch.resident_capacity : static_cast<std::uint32_t>(batch.transforms.size());

        if (!resident) {
            submitted_triangle_count += (geometry.indices.index_count / 3) * instance_count;
        }
        if (frame.transform_count + instance_count > maximum_submission_count_ ||
            frame.draw_count + instance_count > maximum_draw_count_) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
        }

        if (!geometry.meshlets.valid()) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::invalid_mesh));
        }

        if (!frame.upload_buffer.mapped()) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::device_error));
        }

        auto const skin_stream = !resident && batch.skinned && batch.palette_offsets.size() == instance_count;

        auto *const transform_out = frame.upload_buffer.mapped_data() + frame.transform_upload_offset +
                                    static_cast<std::size_t>(frame.transform_count) * sizeof(glm::mat4);
        for (std::uint32_t instance = 0; skin_stream && instance < instance_count; ++instance) {
            std::memcpy(transform_out + static_cast<std::size_t>(instance) * sizeof(glm::mat4),
                        batch.transforms[instance], sizeof(glm::mat4));
        }
        frame.transform_count += instance_count;

        if (!resident) {
            auto &ranges = frame.cpu_instance_ranges;
            auto const first = frame.transform_count - instance_count;
            if (!ranges.empty() && ranges.back().first + ranges.back().second == first) {
                ranges.back().second += instance_count;
            } else {
                ranges.emplace_back(first, instance_count);
            }
        }

        auto const first_instance = frame.draw_count;
        auto const vertex_address = geometry_arena_.vertex_address(geometry.vertices);
        auto const meshlet_address = geometry_arena_.device_address(geometry.meshlets.descriptors);
        auto const meshlet_data_address = geometry_arena_.device_address(geometry.meshlets.data);
        auto const material_index = material_storage_.gpu_index(batch.material);

        auto const meshlet_path = uses_meshlet_path(geometry.meshlets.meshlet_count);
        auto const first_meshlet_bit = meshlet_path && allocate_meshlet_bits
                                               ? meshlet_layout.reserve(instance_count, geometry.meshlets.meshlet_count)
                                               : std::uint64_t{0};

        auto const skin_vertex_count = geometry.vertices.vertex_count;
        auto const skin_bytes = (VkDeviceSize{skin_vertex_count} * sizeof(CompressedModelVertex) + 15U) & ~VkDeviceSize{15U};

        auto *const draw_out = frame.upload_buffer.mapped_data() + frame.draw_upload_offset +
                               static_cast<std::size_t>(first_instance) * sizeof(GpuDraw);
        if (!resident && !skin_stream) {
            copy_tasks.push_back(CopyTask{
                    .batch = &batch,
                    .transform_out = transform_out,
                    .draw_out = draw_out,
                    .instance_count = instance_count,
                    .draw =
                            GpuDraw{
                                    .vertex_address = vertex_address,
                                    .meshlet_address = meshlet_address,
                                    .meshlet_data_address = meshlet_data_address,
                                    .material_index = material_index,
                                    .meshlet_visibility_offset = 0,
                            },
                    .visibility = meshlet_path && allocate_meshlet_bits,
                    .first_meshlet_bit = first_meshlet_bit,
                    .meshlet_count = geometry.meshlets.meshlet_count,
            });
        }

        for (std::uint32_t instance = 0; skin_stream && instance < instance_count; ++instance) {
            auto instance_vertex_address = vertex_address;

            if (skin_stream) {
                auto const palette_offset = batch.palette_offsets[instance];

                if (palette_offset != BatchEntry::no_palette) {
                    if (frame.skin_jobs.size() < maximum_skin_jobs_ &&
                        frame.skin_scratch_used + skin_bytes <= skin_scratch_capacity_) {
                        instance_vertex_address = frame.skin_scratch_buffer.device_address + frame.skin_scratch_used;
                        frame.skin_jobs.push_back(GpuSkinJob{
                                .rest_vertex_addr = vertex_address,
                                .skin_addr = geometry_arena_.device_address(geometry.skin),
                                .out_addr = instance_vertex_address,
                                .vertex_count = skin_vertex_count,
                                .palette_offset = palette_offset,
                        });
                        frame.skin_scratch_used += skin_bytes;
                        frame.skinned_vertices += skin_vertex_count;
                    } else {
                        ++frame.skin_fallback_instances;
                    }
                } else {
                    ++frame.skin_fallback_instances;
                }
            }

            auto const draw = GpuDraw{
                    .vertex_address = instance_vertex_address,
                    .meshlet_address = meshlet_address,
                    .meshlet_data_address = meshlet_data_address,
                    .material_index = material_index,
                    .meshlet_visibility_offset = meshlet_path && allocate_meshlet_bits
                                                         ? meshlet_visibility_offset(first_meshlet_bit, instance,
                                                                                     geometry.meshlets.meshlet_count)
                                                         : 0U,
            };
            std::memcpy(draw_out + static_cast<std::size_t>(instance) * sizeof(GpuDraw), &draw, sizeof(GpuDraw));
        }
        frame.draw_count += instance_count;

        GpuDrawCommand command{
                .instance_count = resident ? 0U : instance_count,
                .first_instance = first_instance,
                .blend_faces = std::to_underlying(faces),
        };

        if (meshlet_path) {
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

        if (resident) {
            auto const first_bit = meshlet_path && allocate_meshlet_bits ? first_meshlet_bit : std::uint64_t{0};
            frame.lod_jobs[batch.lod_job].groups[batch.lod_group] = GpuLodGroup{
                    .batch = static_cast<std::uint32_t>(frame.indirect_commands.size()),
                    .first_instance = first_instance,
                    .meshlet_count = meshlet_path && allocate_meshlet_bits ? geometry.meshlets.meshlet_count : 0U,
                    .first_meshlet_bit = static_cast<std::uint32_t>(
                            std::min<std::uint64_t>(first_bit, std::numeric_limits<std::uint32_t>::max())),
                    .draw =
                            GpuDraw{
                                    .vertex_address = vertex_address,
                                    .meshlet_address = meshlet_address,
                                    .meshlet_data_address = meshlet_data_address,
                                    .material_index = material_index,
                                    .meshlet_visibility_offset = 0,
                            },
            };
        }

        frame.indirect_commands.push_back(command);

        auto const *material = material_storage_.get(batch.material);

        auto wind_padding = material != nullptr ? material->wind_strength : 0.0F;
        if (material != nullptr && (material->flags & GpuMaterial::flag_water) != 0) {
            // The vertex stage lifts and lowers a water surface by up to its wave height.
            wind_padding = std::max(std::abs(wind_padding), std::abs(material->water_shape.x));
        }

        frame.batch_bounds.push_back(GpuCullBounds{
                .bounds_min = submesh.bounds_min,
                .wind_padding = wind_padding,
                .bounds_max = submesh.bounds_max,
                .first_chunk = frame.cull_chunk_count,
        });
        frame.cull_chunk_count += (instance_count + cull_chunk_size - 1) / cull_chunk_size;

        return {};
    };

    for (auto const *batch: opaque_batches_) {
        if (auto result = emit_batch(*batch, true); !result) {
            return std::unexpected(result.error());
        }
    }

    frame.opaque_indirect_count = static_cast<std::uint32_t>(frame.indirect_commands.size());

    for (auto const *batch: double_sided_batches_) {
        if (auto result = emit_batch(*batch, true); !result) {
            return std::unexpected(result.error());
        }
    }

    frame.double_sided_indirect_count =
            static_cast<std::uint32_t>(frame.indirect_commands.size()) - frame.opaque_indirect_count;

    for (auto const *batch: mask_batches_) {
        if (auto result = emit_batch(*batch, true); !result) {
            return std::unexpected(result.error());
        }
    }

    frame.mask_indirect_count = static_cast<std::uint32_t>(frame.indirect_commands.size()) -
                                frame.opaque_indirect_count - frame.double_sided_indirect_count;

    if (build.blend_sort_future.valid()) {
        build.blend_sort_future.wait();
    }

    frame.blend_runs.clear();
    frame.scene_copy_needed = build.has_water;
    auto const blend_first = static_cast<std::uint32_t>(frame.indirect_commands.size());

    for (auto const &pending: blend_batches_) {
        // A double-sided batch is emitted twice (back faces, then front faces), except when its single command is
        // filled in on the GPU (LOD-driven) or it streams skinned vertices, which would be skinned twice.
        auto const &entry = *pending.entry;
        auto const split_allowed = entry.lod_job == BatchEntry::no_lod_job && !entry.skinned;
        auto const passes = blend_order::passes_for(pending.double_sided, split_allowed);

        for (std::uint32_t pass = 0; pass < passes.count; ++pass) {
            if (auto result = emit_batch(entry, false, passes.faces[pass]); !result) {
                return std::unexpected(result.error());
            }

            auto const meshlet = frame.indirect_commands.back().meshlet_count != 0;
            auto const index = static_cast<std::uint32_t>(frame.indirect_commands.size()) - 1U - blend_first;
            if (!frame.blend_runs.empty() && frame.blend_runs.back().meshlet == meshlet) {
                ++frame.blend_runs.back().count;
            } else {
                frame.blend_runs.push_back(render_pass::BlendRun{.first = index, .count = 1, .meshlet = meshlet});
            }
        }
    }

    frame.blend_indirect_count = static_cast<std::uint32_t>(frame.indirect_commands.size()) -
                                 frame.opaque_indirect_count - frame.double_sided_indirect_count -
                                 frame.mask_indirect_count;

    {
        auto starts = std::vector<std::size_t>(copy_tasks.size() + 1, 0);
        for (auto index = std::size_t{0}; index < copy_tasks.size(); ++index) {
            starts[index + 1] = starts[index] + copy_tasks[index].instance_count;
        }
        auto const total = starts.back();

        // Instances [first, last) counted across all the tasks in order.
        auto const copy_range = [&](std::size_t first, std::size_t last) {
            auto task = static_cast<std::size_t>(std::ranges::upper_bound(starts, first) - starts.begin()) - 1;
            for (auto at = first; at < last; ++task) {
                auto const &copy = copy_tasks[task];
                auto const begin = at - starts[task];
                auto const end = std::min<std::size_t>(copy.instance_count, last - starts[task]);

                for (auto instance = begin; instance < end; ++instance) {
                    std::memcpy(copy.transform_out + instance * sizeof(glm::mat4), copy.batch->transforms[instance],
                                sizeof(glm::mat4));

                    auto draw = copy.draw;
                    if (copy.visibility) {
                        draw.meshlet_visibility_offset = meshlet_visibility_offset(
                                copy.first_meshlet_bit, static_cast<std::uint32_t>(instance), copy.meshlet_count);
                    }
                    std::memcpy(copy.draw_out + instance * sizeof(GpuDraw), &draw, sizeof(GpuDraw));
                }
                at = starts[task] + end;
            }
        };

        constexpr auto parallel_copy_threshold = std::size_t{16'384};
        if (total < parallel_copy_threshold) {
            copy_range(0, total);
        } else {
            thread_pool().submit_blocks(std::size_t{0}, total, copy_range).wait();
        }
    }

    frame.indirect_command_count = static_cast<std::uint32_t>(frame.indirect_commands.size());

    return {};
}

auto Renderer::prepare_frame_view(VkCommandBuffer command_buffer, CameraMatrices const &matrices,
                                  std::uint32_t frame_index, bool secondary) -> std::expected<void, RendererError> {
    // A secondary view takes the submissions the primary view already collected, and leaves everything that happens
    // once a frame (streaming, pipeline and material updates, the environment, the timing queries) to the primary.
    frame_time_ = matrices.time;

    // Replaced transients are destroyed here, with nothing recorded for the frame yet that could have them bound.
    transient_allocator_->release_retired();

    ZoneScopedNC("PrepareFrame", tracy::Color::RoyalBlue);

    if (!initialized_ || command_buffer == VK_NULL_HANDLE || frame_index >= frames_.size()) {
        clear_submissions();
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    if (!secondary && pending_samples_ != samples_) {
        // Frame-graph targets are keyed on their sample count, so they reallocate by themselves; the idle wait only
        // guarantees the previous count is no longer in flight.
        if (auto waited = wait_idle(); !waited) {
            clear_submissions();
            return std::unexpected(waited.error());
        }

        info("Renderer: MSAA {}x -> {}x", static_cast<std::uint32_t>(samples_),
             static_cast<std::uint32_t>(pending_samples_));

        samples_ = pending_samples_;
        hiz_history_valid_ = false;
    }

    if (history_samples_ != samples_) {
        hiz_history_valid_ = false;
        history_samples_ = samples_;
    }

    auto &frame_query = timestamp_queries_[frame_index];
    auto &frame_pipeline_query = pipeline_stat_queries_[frame_index];

    if (!secondary) {
        vkCmdResetQueryPool(command_buffer, frame_query.query_pool, 0, total_query_count);

        vkCmdResetQueryPool(command_buffer, frame_pipeline_query.query_pool, 0, 1);

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame_query.query_pool, 0);

        pipeline_graph_.tick_retirement();
        geometry_arena_.tick_retirement();

        prune_outline_variants();

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

        std::erase_if(pending_pick_geometry_, [&](PendingPickGeometry &pending) {
            if (pending.geometry.wait_for(std::chrono::seconds{0}) != std::future_status::ready) {
                return false;
            }

            // A model destroyed in the meantime has no slot, and the geometry is dropped.
            register_model_pick_geometry(pending.model, pending.geometry.get());
            return true;
        });

        if (auto environment = environment_.prepare(command_buffer, ++frame_counter_); !environment) {
            clear_submissions();
            return std::unexpected(environment.error());
        }

        auto resource_result = gpu_resource_table_.prepare_frame(frame_index, image_storage_, sampler_storage_);

        if (!resource_result) {
            clear_submissions();
            return std::unexpected(make_resource_table_error(resource_result.error()));
        }
    }

    auto &frame = frames_[frame_index];

    if (auto cluster_buffers = prepare_cluster_buffers(frame); !cluster_buffers) {
        clear_submissions();
        return std::unexpected(cluster_buffers.error());
    }

    frame.view_projection = matrices.projection * matrices.view;

    frame.retired_buffers.clear();
    record_resident_instance_uploads(command_buffer, frame);

    frame.draw_count = 0;
    frame.transform_count = 0;
    computed_transforms_.clear();
    frame.indirect_commands.clear();
    frame.batch_bounds.clear();

    frame.indirect_command_count = 0;
    frame.cull_chunk_count = 0;
    frame.lod_chunk_count = 0;
    frame.skin_jobs.clear();
    frame.skin_scratch_used = 0;
    frame.skin_palette_count = 0;
    frame.skin_chunk_total = 0;
    frame.skin_fallback_instances = 0;
    frame.skinned_vertices = 0;
    frame.lod_jobs.clear();
    frame.cpu_instance_ranges.clear();
    frame.opaque_indirect_count = 0;
    frame.double_sided_indirect_count = 0;
    frame.mask_indirect_count = 0;
    frame.blend_indirect_count = 0;
    frame.blend_runs.clear();
    frame.scene_copy_needed = false;
    frame.shadow_update_mask = 0;

    auto const camera_position = glm::vec3(glm::inverse(matrices.view)[3]);
    frame.lod_camera_position = camera_position;

    ++batch_frame_;

    if (batch_frame_ == 0) {
        for (auto &batch: batches_ | std::views::values) {
            batch.frame_stamp = 0;
        }
        batch_frame_ = 1;
    }

    active_batches_.clear();

    opaque_batches_.clear();
    double_sided_batches_.clear();
    mask_batches_.clear();
    blend_batches_.clear();

    BatchBuild build_state;
    if (auto built = build_batches(frame, camera_position, build_state); !built) {
        return std::unexpected(built.error());
    }
    if (auto classified = classify_batches(camera_position, build_state); !classified) {
        return std::unexpected(classified.error());
    }
    plan_shadow_batches(frame, build_state);
    if (auto emitted = emit_batches(frame, build_state); !emitted) {
        return std::unexpected(emitted.error());
    }

    auto const resident_instance_count = build_state.resident_instance_count;
    auto const resident_reserved_slots = build_state.resident_reserved_slots;
    auto const current_shadow_scene_signature = build_state.shadow_scene_signature;
    auto const has_animated_shadow_casters = build_state.has_animated_shadow_casters;
    auto &meshlet_layout = build_state.meshlet_layout;
    auto &submitted_triangle_count = build_state.submitted_triangle_count;

    auto frame_checksum = std::uint64_t{0};
    if (frame_checksum_enabled_ && frame.upload_buffer.mapped()) {
        auto const *const upload = frame.upload_buffer.mapped_data();
        frame_checksum = checksum_bytes(0, upload + frame.transform_upload_offset,
                                        static_cast<std::size_t>(frame.transform_count) * sizeof(glm::mat4));
        frame_checksum = checksum_bytes(frame_checksum, upload + frame.draw_upload_offset,
                                        static_cast<std::size_t>(frame.draw_count) * sizeof(GpuDraw));
        frame_checksum = checksum_bytes(frame_checksum, frame.indirect_commands.data(),
                                        frame.indirect_commands.size() * sizeof(GpuDrawCommand));
        frame_checksum = checksum_bytes(frame_checksum, frame.batch_bounds.data(),
                                        frame.batch_bounds.size() * sizeof(GpuCullBounds));
    }

    if (!frame.lod_jobs.empty()) {
        if (auto written = frame.lod_jobs_buffer.write(0, std::span<GpuLodJob const>{frame.lod_jobs}); !written) {
            clear_submissions();
            return std::unexpected(make_device_error(written.error()));
        }
    }

    if (!secondary) {
        auto material_result = material_storage_.prepare_frame(command_buffer, frame_index);
        if (!material_result) {
            clear_submissions();
            return std::unexpected(make_material_error(material_result.error()));
        }
    }

    if (auto skin_result = prepare_skin_upload(frame); !skin_result) {
        clear_submissions();
        return std::unexpected(skin_result.error());
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

    // The primary view consumes the flag; the other views' cascade caches see the value it had.
    auto const dynamic_casters_dirty = secondary ? dynamic_shadow_casters_dirty_for_views_ : dynamic_shadow_casters_dirty_;
    if (!secondary) {
        dynamic_shadow_casters_dirty_for_views_ = dynamic_shadow_casters_dirty_;
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

        auto const dynamic_due = (dynamic_casters_dirty || has_animated_shadow_casters) && period_due;
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
            resolved_view_projection[cascade] = cached.view_projection;
            resolved_split_far[cascade] = cached.split_far;
            resolved_texel_world[cascade] = cached.texel_world;
            resolved_depth_scale[cascade] = cached.depth_scale;
        }
    }

    frame.pending_shadow_light_direction = light_direction;
    frame.pending_shadow_depth_bias_constant = shadow_settings_.depth_bias_constant;
    frame.pending_shadow_depth_bias_slope = shadow_settings_.depth_bias_slope;
    frame.pending_shadow_caster_revision = shadow_caster_revision_;
    frame.pending_shadow_scene_signature = current_shadow_scene_signature;
    if (!secondary) {
        dynamic_shadow_casters_dirty_ = false;
    }

    auto const view_projection = projection * view;

    auto const frustum_planes = extract_frustum_planes(view_projection);

    auto const cluster_near = std::max(matrices.near_clip, 1e-4F);
    auto const cluster_far = std::max(matrices.far_clip, cluster_near * 2.0F);
    auto const cluster_z_scale =
            static_cast<float>(frame.cluster_grid.depth_slices) / std::log(cluster_far / cluster_near);
    auto const cluster_z_bias = -std::log(cluster_near) * cluster_z_scale;

    auto const light_lod_pixel_scale = std::abs(projection[1][1]) * static_cast<float>(extent_.height) * 0.5F;
    auto const light_lod_cull = std::max(light_lod_settings_.cull_radius_pixels, 0.0F);
    auto const light_lod_fade = std::max(light_lod_settings_.fade_radius_pixels, light_lod_cull + 1e-3F);

    // The local shadow atlas is shared, and the primary view picks and draws its tiles. A secondary view reuses the
    // tiles the primary just chose, and draws none of its own.
    frame.local_shadow_render_tiles.clear();
    if (!secondary) {
        local_shadow_tile_staging_.clear();
        local_shadow_plane_staging_.clear();
        point_light_shadow_tiles_.assign(point_light_submissions_.size(), local_shadows::no_tile);
        spot_light_shadow_tiles_.assign(spot_light_submissions_.size(), local_shadows::no_tile);
        local_shadow_stats_ = {};
        local_shadow_allocator_.begin_frame();
    }

    if (!secondary && local_shadow_settings_.enabled) {
        struct ShadowCandidate {
            bool point = false;
            std::size_t index = 0;
            std::uint64_t id = 0;
            float radius_pixels = 0.0F;
            // radius_pixels with a bonus for lights that already hold tiles, so near-ties do not swap places.
            float priority = 0.0F;
        };

        std::vector<ShadowCandidate> candidates;
        auto const view_planes = extract_frustum_planes(view_projection);

        auto const consider = [&](bool point, std::size_t index, std::uint64_t id, glm::vec3 position, float range) {
            if (id == 0) {
                return;
            }

            for (auto const &plane: view_planes) {
                if (glm::dot(glm::vec3{plane}, position) + plane.w < -range) {
                    return;
                }
            }

            auto const distance = std::max(glm::length(position - camera_position), 1e-3F);
            auto const radius_pixels = range * light_lod_pixel_scale / distance;

            if (radius_pixels < local_shadow_settings_.minimum_screen_radius_pixels) {
                return;
            }

            constexpr float incumbent_bonus = 1.3F;
            auto const incumbent = local_shadow_allocator_.find(id) != nullptr;

            candidates.push_back(ShadowCandidate{.point = point,
                                                 .index = index,
                                                 .id = id,
                                                 .radius_pixels = radius_pixels,
                                                 .priority = radius_pixels * (incumbent ? incumbent_bonus : 1.0F)});
        };

        for (std::size_t i = 0; i < point_light_submissions_.size(); ++i) {
            auto const &light = point_light_submissions_[i];
            if (light.casts_shadows) {
                consider(true, i, light.id, light.position, light.range);
            }
        }
        for (std::size_t i = 0; i < spot_light_submissions_.size(); ++i) {
            auto const &light = spot_light_submissions_[i];
            if (light.casts_shadows) {
                consider(false, i, light.id, light.position, light.range);
            }
        }

        std::ranges::sort(candidates, [](ShadowCandidate const &lhs, ShadowCandidate const &rhs) {
            return lhs.priority != rhs.priority ? lhs.priority > rhs.priority : lhs.id < rhs.id;
        });

        local_shadow_stats_.requested_lights = static_cast<std::uint32_t>(candidates.size());

        struct SelectedShadow {
            ShadowCandidate candidate;
            local_shadows::Allocation allocation;
            std::uint64_t hash = 0;
            bool dirty = true;
        };

        std::vector<SelectedShadow> selected;

        for (auto const &candidate: candidates) {
            if (selected.size() >= local_shadow_settings_.maximum_lights) {
                break;
            }

            auto resolution = local_shadows::resolution_for_screen_radius(candidate.radius_pixels);
            auto const tile_count = candidate.point ? local_shadows::point_face_count : 1U;

            // A light keeps the tile size it has unless the size it wants is two tiers away: resizing redraws the
            // tile, so a light hovering at a tier boundary would otherwise redraw every few frames.
            if (auto const *existing = local_shadow_allocator_.find(candidate.id);
                existing != nullptr && existing->tile_count == tile_count && existing->resolution * 2 >= resolution &&
                existing->resolution <= resolution * 2) {
                resolution = existing->resolution;
            }
            auto const *allocation = local_shadow_allocator_.request_best_fit(candidate.id, tile_count, resolution);

            if (allocation != nullptr) {
                selected.push_back(SelectedShadow{.candidate = candidate, .allocation = *allocation});
            }
        }

        auto const light_position = [&](ShadowCandidate const &candidate) {
            return candidate.point ? point_light_submissions_[candidate.index].position
                                   : spot_light_submissions_[candidate.index].position;
        };
        auto const light_range = [&](ShadowCandidate const &candidate) {
            return candidate.point ? point_light_submissions_[candidate.index].range
                                   : spot_light_submissions_[candidate.index].range;
        };

        if (!local_shadow_atlas_initialized_ || !local_shadow_settings_.caching) {
            local_shadow_cache_.clear();
        }

        if (local_shadow_settings_.caching && !selected.empty()) {
            auto const float_bits = [](float value) { return std::uint64_t{std::bit_cast<std::uint32_t>(value)}; };
            auto const vec3_hash = [&](std::uint64_t seed, glm::vec3 const &value) {
                seed = shadow_signature_combine(seed, float_bits(value.x));
                seed = shadow_signature_combine(seed, float_bits(value.y));
                return shadow_signature_combine(seed, float_bits(value.z));
            };

            // What is drawn into a tile depends on the light and on every shadow caster inside its range. Both are
            // folded into one hash per light; a tile whose hash matches last frame's keeps its depth.
            std::vector<std::uint64_t> caster_hash(selected.size(), 0);
            // A point light is covered by its range sphere. A spot light is only affected inside its own view
            // frustum, which is far smaller than the sphere of the same range.
            std::vector<glm::vec4> volumes;
            std::vector<std::array<glm::vec4, 6>> spot_planes(selected.size());
            volumes.reserve(selected.size());
            for (std::size_t light = 0; light < selected.size(); ++light) {
                auto const &entry = selected[light];
                volumes.emplace_back(light_position(entry.candidate), light_range(entry.candidate));

                if (!entry.candidate.point) {
                    auto const &spot = spot_light_submissions_[entry.candidate.index];
                    auto const outer = glm::radians(std::max(spot.inner_cone_degrees, spot.outer_cone_degrees));
                    spot_planes[light] = extract_frustum_planes(local_shadows::spot_view_projection(
                            spot.position, spot.direction, outer, spot.range, entry.allocation.resolution));
                }
            }

            // First one cheap hash over every caster in the scene. While it matches last frame's, nothing anywhere
            // moved, so each light's casters are what they were and the per-light pass below is skipped.
            bool lod_driven_casters = false;
            std::uint64_t scene_hash = 0;

            for (auto const *batch: active_batches_) {
                auto const *material = material_storage_.get(batch->material);
                auto const alpha_mode = material != nullptr ? material->alpha_mode : AlphaMode::opaque;

                if (alpha_mode == AlphaMode::blend || batch_max_shadow_cascade(batch) < 0) {
                    continue;
                }

                if (batch->lod_job != BatchEntry::no_lod_job) {
                    // Instances picked on the GPU by distance: their positions are not known here, so every light is
                    // redrawn and the hashes below would go unused.
                    lod_driven_casters = true;
                    break;
                }

                auto const wind = material != nullptr ? std::abs(material->wind_strength) : 0.0F;
                auto batch_seed = std::uint64_t{0xcbf29ce484222325ULL};
                batch_seed = shadow_signature_combine(batch_seed, batch->mesh.index);
                batch_seed = shadow_signature_combine(batch_seed, batch->submesh_index);
                batch_seed = shadow_signature_combine(batch_seed, material_storage_.gpu_index(batch->material));
                // No LOD in the seed: an instance crossing an LOD threshold as the camera moves is not a change that
                // is worth redrawing a light's tile for.
                if (wind > 1e-6F || batch->skinned) {
                    batch_seed = shadow_signature_combine(batch_seed, shadow_frame_);
                }

                for (auto const *transform: batch->transforms) {
                    std::array<std::uint64_t, 8> words{};
                    std::memcpy(words.data(), transform, sizeof(words));

                    auto word_hash = batch_seed;
                    for (auto const word: words) {
                        word_hash = (word_hash ^ word) * 0x9e3779b97f4a7c15ULL;
                        word_hash ^= word_hash >> 29U;
                    }
                    scene_hash += word_hash;
                }
            }

            auto const casters_unchanged =
                    !lod_driven_casters && local_shadow_scene_hash_valid_ && scene_hash == local_shadow_scene_hash_ &&
                    std::ranges::all_of(selected, [this](SelectedShadow const &entry) {
                        return local_shadow_cache_.contains(entry.candidate.id);
                    });

            local_shadow_scene_hash_ = scene_hash;
            local_shadow_scene_hash_valid_ = !lod_driven_casters;

            if (casters_unchanged) {
                for (std::size_t light = 0; light < selected.size(); ++light) {
                    caster_hash[light] = local_shadow_cache_.at(selected[light].candidate.id).casters;
                }
            }

            if (!casters_unchanged && !lod_driven_casters) {
                // Which casters fall inside which light's volume, summed per light. The sum is order independent, so
                // the instances are split into blocks that run on the thread pool and add into atomics.
                struct CasterBatch {
                    BatchEntry const *batch = nullptr;
                    glm::vec3 local_centre{0.0F};
                    glm::vec3 local_extent{0.0F};
                    float wind_padding = 0.0F;
                    std::uint64_t key = 0;
                };

                struct CasterBlock {
                    std::uint32_t batch = 0;
                    std::uint32_t begin = 0;
                    std::uint32_t end = 0;
                };

                constexpr std::uint32_t instances_per_block = 512;

                std::vector<CasterBatch> caster_batches;
                std::vector<CasterBlock> blocks;

                for (auto const *batch: active_batches_) {
                    auto const *material = material_storage_.get(batch->material);
                    auto const alpha_mode = material != nullptr ? material->alpha_mode : AlphaMode::opaque;

                    if (alpha_mode == AlphaMode::blend || batch_max_shadow_cascade(batch) < 0 ||
                        batch->lod_job != BatchEntry::no_lod_job) {
                        continue;
                    }

                    auto const *mesh = mesh_slot(batch->mesh);
                    if (mesh == nullptr) {
                        continue;
                    }

                    auto const &submesh = mesh->submeshes[batch->submesh_index];
                    auto const wind_padding = material != nullptr ? material->wind_strength : 0.0F;
                    auto const animated = std::abs(wind_padding) > 1e-6F || batch->skinned;

                    auto batch_key = std::uint64_t{0xcbf29ce484222325ULL};
                    batch_key = shadow_signature_combine(batch_key, batch->mesh.index);
                    batch_key = shadow_signature_combine(batch_key, batch->submesh_index);
                    batch_key = shadow_signature_combine(batch_key, material_storage_.gpu_index(batch->material));
                    if (animated) {
                        batch_key = shadow_signature_combine(batch_key, shadow_frame_);
                    }

                    auto const batch_index = static_cast<std::uint32_t>(caster_batches.size());
                    caster_batches.push_back(CasterBatch{
                            .batch = batch,
                            .local_centre = (submesh.bounds_min + submesh.bounds_max) * 0.5F,
                            .local_extent = (submesh.bounds_max - submesh.bounds_min) * 0.5F,
                            .wind_padding = wind_padding,
                            .key = batch_key,
                    });

                    auto const count = static_cast<std::uint32_t>(batch->transforms.size());
                    for (std::uint32_t begin = 0; begin < count; begin += instances_per_block) {
                        blocks.push_back(CasterBlock{.batch = batch_index,
                                                     .begin = begin,
                                                     .end = std::min(begin + instances_per_block, count)});
                    }
                }

                std::vector<std::atomic<std::uint64_t>> totals(selected.size());
                for (auto &total: totals) {
                    total.store(0, std::memory_order_relaxed);
                }

                // The light volumes binned into a coarse grid on XZ, so an instance only tests the lights whose volume
                // reaches its cells instead of every shadowed light.
                constexpr int light_grid_size = 32;
                auto grid_min = glm::vec2{std::numeric_limits<float>::max()};
                auto grid_max = glm::vec2{std::numeric_limits<float>::lowest()};
                for (auto const &volume: volumes) {
                    grid_min = glm::min(grid_min, glm::vec2{volume.x, volume.z} - volume.w);
                    grid_max = glm::max(grid_max, glm::vec2{volume.x, volume.z} + volume.w);
                }

                auto const grid_scale = static_cast<float>(light_grid_size) /
                                        std::max(std::max(grid_max.x - grid_min.x, grid_max.y - grid_min.y), 1e-3F);
                auto const grid_cell = [&](float value, float low) {
                    return std::clamp(static_cast<int>((value - low) * grid_scale), 0, light_grid_size - 1);
                };

                // Compressed rows: the lights of cell `i` are cell_light_indices[cell_first[i] .. cell_first[i + 1]).
                std::array<std::uint32_t, light_grid_size * light_grid_size + 1> cell_first{};
                auto const for_each_light_cell = [&](glm::vec4 const &volume, auto &&visit) {
                    for (auto row = grid_cell(volume.z - volume.w, grid_min.y);
                         row <= grid_cell(volume.z + volume.w, grid_min.y); ++row) {
                        for (auto column = grid_cell(volume.x - volume.w, grid_min.x);
                             column <= grid_cell(volume.x + volume.w, grid_min.x); ++column) {
                            visit(static_cast<std::size_t>(row * light_grid_size + column));
                        }
                    }
                };

                for (auto const &volume: volumes) {
                    for_each_light_cell(volume, [&](std::size_t cell) { ++cell_first[cell + 1]; });
                }
                std::partial_sum(cell_first.begin(), cell_first.end(), cell_first.begin());

                std::vector<std::uint32_t> cell_light_indices(cell_first.back());
                auto cell_fill = cell_first;
                for (std::size_t light = 0; light < volumes.size(); ++light) {
                    for_each_light_cell(volumes[light], [&](std::size_t cell) {
                        cell_light_indices[cell_fill[cell]++] = static_cast<std::uint32_t>(light);
                    });
                }

                auto const cell_lights = [&](int row, int column) {
                    auto const cell = static_cast<std::size_t>(row * light_grid_size + column);
                    return std::span<std::uint32_t const>{cell_light_indices.data() + cell_first[cell],
                                                          cell_light_indices.data() + cell_first[cell + 1]};
                };

                auto const accumulate = [&](std::size_t first_block, std::size_t last_block) {
                    std::vector<std::uint64_t> local(selected.size(), 0);
                    std::vector<std::uint32_t> visited(selected.size(), 0);
                    std::uint32_t visit_stamp = 0;

                    for (auto block_index = first_block; block_index < last_block; ++block_index) {
                        auto const &block = blocks[block_index];
                        auto const &caster = caster_batches[block.batch];

                        for (auto instance = block.begin; instance < block.end; ++instance) {
                            auto const &model = *caster.batch->transforms[instance];
                            auto const centre = glm::vec3{model * glm::vec4{caster.local_centre, 1.0F}};
                            auto extent = glm::abs(glm::vec3{model[0]}) * caster.local_extent.x +
                                          glm::abs(glm::vec3{model[1]}) * caster.local_extent.y +
                                          glm::abs(glm::vec3{model[2]}) * caster.local_extent.z;
                            extent.x += std::abs(caster.wind_padding);
                            extent.z += std::abs(caster.wind_padding);

                            std::uint64_t instance_hash = 0;
                            bool hashed = false;

                            auto const visit_light = [&](std::size_t light) {
                                auto const outside =
                                        glm::max(glm::abs(glm::vec3{volumes[light]} - centre) - extent, glm::vec3{0.0F});
                                if (glm::dot(outside, outside) > volumes[light].w * volumes[light].w) {
                                    return;
                                }

                                if (!selected[light].candidate.point) {
                                    bool inside = true;
                                    for (auto const &plane: spot_planes[light]) {
                                        if (glm::dot(glm::vec3{plane}, centre) + plane.w +
                                                    glm::dot(glm::abs(glm::vec3{plane}), extent) < 0.0F) {
                                            inside = false;
                                            break;
                                        }
                                    }

                                    if (!inside) {
                                        return;
                                    }
                                }

                                if (!hashed) {
                                    instance_hash = caster.key;
                                    for (int column = 0; column < 4; ++column) {
                                        instance_hash = vec3_hash(instance_hash, glm::vec3{model[column]});
                                    }
                                    instance_hash = shadow_signature_mix(instance_hash);
                                    hashed = true;
                                }

                                local[light] += instance_hash;
                            };

                            auto const low_x = centre.x - extent.x;
                            auto const high_x = centre.x + extent.x;
                            auto const low_z = centre.z - extent.z;
                            auto const high_z = centre.z + extent.z;
                            if (high_x < grid_min.x || low_x > grid_max.x || high_z < grid_min.y ||
                                low_z > grid_max.y) {
                                continue;
                            }

                            auto const first_column = grid_cell(low_x, grid_min.x);
                            auto const last_column = grid_cell(high_x, grid_min.x);
                            auto const first_row = grid_cell(low_z, grid_min.y);
                            auto const last_row = grid_cell(high_z, grid_min.y);

                            if (first_column == last_column && first_row == last_row) {
                                for (auto const light: cell_lights(first_row, first_column)) {
                                    visit_light(light);
                                }
                                continue;
                            }

                            // A light listed in several of the cells must still count once.
                            ++visit_stamp;
                            for (auto row = first_row; row <= last_row; ++row) {
                                for (auto column = first_column; column <= last_column; ++column) {
                                    for (auto const light: cell_lights(row, column)) {
                                        if (visited[light] != visit_stamp) {
                                            visited[light] = visit_stamp;
                                            visit_light(light);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (std::size_t light = 0; light < local.size(); ++light) {
                        totals[light].fetch_add(local[light], std::memory_order_relaxed);
                    }
                };

                constexpr std::size_t serial_block_limit = 4;

                if (blocks.size() <= serial_block_limit) {
                    accumulate(0, blocks.size());
                } else {
                    thread_pool().submit_blocks(std::size_t{0}, blocks.size(), accumulate).wait();
                }

                for (std::size_t light = 0; light < selected.size(); ++light) {
                    caster_hash[light] = totals[light].load(std::memory_order_relaxed);
                }
            }

            for (std::size_t light = 0; light < selected.size(); ++light) {
                auto &entry = selected[light];
                auto const &candidate = entry.candidate;

                auto state = std::uint64_t{0x84222325cbf29ce4ULL};
                state = shadow_signature_combine(state, entry.allocation.generation);
                state = shadow_signature_combine(state, shadow_caster_revision_);
                state = shadow_signature_combine(state, float_bits(local_shadow_settings_.depth_bias_constant));
                state = shadow_signature_combine(state, float_bits(local_shadow_settings_.depth_bias_slope));
                state = shadow_signature_combine(state, float_bits(light_range(candidate)));
                state = vec3_hash(state, light_position(candidate));

                if (!candidate.point) {
                    auto const &spot = spot_light_submissions_[candidate.index];
                    state = vec3_hash(state, spot.direction);
                    state = shadow_signature_combine(
                            state, float_bits(std::max(spot.inner_cone_degrees, spot.outer_cone_degrees)));
                }

                entry.hash = shadow_signature_combine(state, caster_hash[light]);

                if (lod_driven_casters) {
                    // No caster hash was computed; a miss makes the next frame without LOD casters redraw the tile.
                    entry.dirty = true;
                    local_shadow_cache_.erase(candidate.id);
                    continue;
                }

                auto const cached = local_shadow_cache_.find(candidate.id);
                entry.dirty = cached == local_shadow_cache_.end() || cached->second.hash != entry.hash;
                local_shadow_cache_[candidate.id] = LocalShadowCacheEntry{.hash = entry.hash,
                                                                          .casters = caster_hash[light]};
            }
        }

        // Tiles that must be redrawn come first, so the pass draws a prefix of the tile list.
        std::ranges::stable_partition(selected, [](SelectedShadow const &entry) { return entry.dirty; });

        for (auto const &entry: selected) {
            auto const &candidate = entry.candidate;
            auto const &allocation = entry.allocation;
            auto const first_tile = static_cast<std::uint32_t>(local_shadow_tile_staging_.size());

            for (std::uint32_t face = 0; face < allocation.tile_count; ++face) {
                glm::mat4 face_view_projection{1.0F};
                float texel_world = 0.0F;

                if (candidate.point) {
                    auto const &light = point_light_submissions_[candidate.index];
                    face_view_projection = local_shadows::point_face_view_projection(light.position, face, light.range,
                                                                                     allocation.resolution);
                    texel_world = local_shadows::point_texel_world(allocation.resolution);
                } else {
                    auto const &light = spot_light_submissions_[candidate.index];
                    auto const outer = glm::radians(std::max(light.inner_cone_degrees, light.outer_cone_degrees));
                    face_view_projection = local_shadows::spot_view_projection(light.position, light.direction, outer,
                                                                               light.range, allocation.resolution);
                    texel_world = local_shadows::spot_texel_world(outer, allocation.resolution);
                }

                local_shadow_tile_staging_.push_back(
                        local_shadows::make_gpu_tile(face_view_projection, allocation.tiles[face], texel_world));

                for (auto const &plane: extract_frustum_planes(face_view_projection)) {
                    local_shadow_plane_staging_.push_back(plane);
                }

                if (entry.dirty) {
                    frame.local_shadow_render_tiles.push_back(allocation.tiles[face]);
                }
            }

            (candidate.point ? point_light_shadow_tiles_ : spot_light_shadow_tiles_)[candidate.index] = first_tile;
            ++local_shadow_stats_.shadowed_lights;
            local_shadow_stats_.cached_lights += entry.dirty ? 0U : 1U;
        }
    }

    frame.local_shadow_tile_count = static_cast<std::uint32_t>(local_shadow_tile_staging_.size());
    if (!secondary) {
        local_shadow_stats_.rendered_tiles = static_cast<std::uint32_t>(frame.local_shadow_render_tiles.size());
        local_shadow_stats_.atlas_cells_used = local_shadow_allocator_.used_cells();
    }

    if (frame.local_shadow_tile_count != 0 &&
        (!frame.local_shadow_tiles_buffer.write(0, std::as_bytes(std::span{local_shadow_tile_staging_})) ||
         !frame.local_shadow_planes_buffer.write(0, std::as_bytes(std::span{local_shadow_plane_staging_})))) {
        clear_submissions();
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    if (auto cull_buffers = secondary ? std::expected<void, RendererError>{} : prepare_local_shadow_cull_buffers(frame);
        !cull_buffers) {
        clear_submissions();
        return std::unexpected(cull_buffers.error());
    }

    decal_staging_.clear();
    {
        auto sorted = decals_enabled_ ? decal_submissions_ : std::vector<Decal>{};
        std::ranges::stable_sort(sorted, {}, &Decal::sort_order);

        for (auto const &decal: sorted) {
            auto const inverse = glm::inverse(decal.transform);
            auto const axis = [&decal](int column) { return glm::vec3{decal.transform[column]}; };

            // A scaled box's bounding sphere reaches its corners: half the diagonal of the scaled unit cube.
            auto const half_diagonal =
                    0.5F * glm::length(glm::vec3{glm::length(axis(0)), glm::length(axis(1)), glm::length(axis(2))});

            auto const cos_start = std::cos(glm::radians(std::min(decal.angle_fade_start_degrees, decal.angle_fade_end_degrees)));
            auto const cos_end = std::cos(glm::radians(std::max(decal.angle_fade_start_degrees, decal.angle_fade_end_degrees)));
            auto const fade_scale = 1.0F / std::max(cos_start - cos_end, 1e-4F);
            auto const fade_offset = -cos_end * fade_scale;

            decal_staging_.push_back(GpuDecal{
                    .world_to_decal = inverse,
                    .bounding_sphere = glm::vec4{glm::vec3{decal.transform[3]}, half_diagonal},
                    .tangent_fade_scale = glm::vec4{glm::normalize(axis(0)), fade_scale},
                    .bitangent_fade_offset = glm::vec4{glm::normalize(axis(2)), fade_offset},
                    .normal_opacity = glm::vec4{glm::normalize(axis(1)), decal.opacity},
                    .material_index = material_storage_.gpu_index(decal.material),
            });
        }
    }

    frame.decal_count = static_cast<std::uint32_t>(decal_staging_.size());

    if (!decal_staging_.empty() && !frame.decals_buffer.write(0, std::as_bytes(std::span{decal_staging_}))) {
        clear_submissions();
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

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
            .cluster_z_scale = cluster_z_scale,
            .cluster_z_bias = cluster_z_bias,
            .clustered_lighting = clustered_lighting_ ? 1U : 0U,
            .cluster_debug_heatmap = clustered_lighting_ && cluster_debug_heatmap_ ? 1U : 0U,
            .cluster_grid_x = frame.cluster_grid.tiles_x,
            .cluster_grid_y = frame.cluster_grid.tiles_y,
            .cluster_grid_z = frame.cluster_grid.depth_slices,
            .cluster_light_capacity = frame.cluster_grid.light_capacity,
            .light_lod_pixel_scale = light_lod_pixel_scale,
            .light_lod_cull_radius_pixels = light_lod_settings_.enabled ? light_lod_cull : 0.0F,
            .light_lod_fade_radius_pixels = light_lod_settings_.enabled ? light_lod_fade : 0.0F,
            .view_mode = static_cast<std::uint32_t>(view_mode_),
            .decals_address = frame.decals_buffer.device_address,
            .decal_clusters_address = frame.decal_clusters_buffer.device_address,
            .decal_count = frame.decal_count,
            .local_shadow_tiles_address = frame.local_shadow_tiles_buffer.device_address,
            .local_shadow_atlas_texture = local_shadow_atlas_.handle().index,
            .local_shadow_tile_count = frame.local_shadow_tile_count,
            .local_shadow_normal_offset_texels = local_shadow_settings_.normal_offset_texels,
            .environment = environment_.ubo_block(),
    };

    if (!ubos_[frame_index].write(0, std::span{&ubo, 1})) {
        clear_submissions();
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

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

    auto const forward_extent = extent_;
    frame.occlusion_active = occlusion_culling_ && occlusion_culling_supported() && static_cast<bool>(hiz_.image) &&
                             hiz_.depth_extent.width == forward_extent.width &&
                             hiz_.depth_extent.height == forward_extent.height;

    if (!meshlet_layout.fits() && !meshlet_visibility_cap_warned_) {
        meshlet_visibility_cap_warned_ = true;
        warn("Renderer: {} meshlet visibility bits exceed the cap of {}; meshlet occlusion culling is off for such "
             "frames",
             meshlet_layout.total_bits(), maximum_meshlet_visibility_bits);
    }

    frame.meshlet_occlusion_active = frame.occlusion_active && meshlet_occlusion_culling_ && meshlet_culling_ &&
                                     meshlet_layout.fits() && meshlet_layout.total_bits() != 0;
    frame.meshlet_visibility_words =
            frame.meshlet_occlusion_active ? static_cast<std::uint32_t>(meshlet_layout.word_count()) : 0U;

    if (frame.meshlet_visibility_words > frame.meshlet_visibility_capacity_words) {
        auto const capacity_words = static_cast<std::uint32_t>(std::bit_ceil(frame.meshlet_visibility_words));
        auto visibility = create_shared_buffer(
                context_, BufferCreateInfo{
                                  .size = VkDeviceSize{capacity_words} * sizeof(std::uint32_t),
                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  .memory = BufferMemory::device,
                                  .debug_name = "renderer.frame_meshlet_visibility",
                          });

        if (!visibility) {
            clear_submissions();
            return std::unexpected(make_device_error(visibility.error()));
        }

        frame.meshlet_visibility_buffer = std::move(*visibility);
        frame.meshlet_visibility_capacity_words = capacity_words;
    }

    {
        auto const states = occlusion_view_states(frame.occlusion_active, hiz_history_valid_, occlusion_test_mode_);

        auto const meshlet_visibility_address =
                frame.meshlet_occlusion_active ? frame.meshlet_visibility_buffer.device_address : VkDeviceAddress{0};

        auto const stats_slot_address = [&frame](std::uint32_t slot) -> VkDeviceAddress {
            return frame.occlusion_stats_buffer.device_address + VkDeviceAddress{slot} * sizeof(std::uint32_t);
        };

        std::array<GpuOcclusionView, 2> const occlusion_views{
                GpuOcclusionView{
                        .view_projection = hiz_history_view_projection_,
                        .meshlet_visibility_address = meshlet_visibility_address,
                        .stats_address = stats_slot_address(occlusion_stat_deferred_meshlets),
                        .hiz_texture_index = hiz_.image.handle().index,
                        .hiz_mip_count = hiz_.mip_count,
                        .depth_width = hiz_.depth_extent.width,
                        .depth_height = hiz_.depth_extent.height,
                        .enabled = states.early,
                },
                GpuOcclusionView{
                        .view_projection = view_projection,
                        .meshlet_visibility_address = meshlet_visibility_address,
                        .stats_address = stats_slot_address(occlusion_stat_occluded_meshlets),
                        .hiz_texture_index = hiz_.image.handle().index,
                        .hiz_mip_count = hiz_.mip_count,
                        .depth_width = hiz_.depth_extent.width,
                        .depth_height = hiz_.depth_extent.height,
                        .enabled = states.late,
                },
        };

        if (!frame.occlusion_views_buffer.write(0, std::span{occlusion_views})) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::device_error));
        }
    }

    auto const shadow_tiles_in_use = local_shadow_stats_.shadowed_lights != 0;

    if ((lights_dirty_mask_ & (1u << frame_index)) != 0 || shadow_tiles_in_use || frame.lights_have_shadow_tiles) {
        light_staging_.clear();
        light_staging_.reserve(point_light_submissions_.size() + spot_light_submissions_.size());

        for (std::size_t point_index = 0; point_index < point_light_submissions_.size(); ++point_index) {
            auto const &point = point_light_submissions_[point_index];
            light_staging_.push_back(GpuLight{
                    .position = point.position,
                    .range = point.range,
                    .colour = point.colour,
                    .intensity = point.intensity,
                    .type = GpuLightType::point,
                    .shadow_tile = point_light_shadow_tiles_[point_index],
            });
        }

        for (std::size_t spot_index = 0; spot_index < spot_light_submissions_.size(); ++spot_index) {
            auto const &spot = spot_light_submissions_[spot_index];
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
                    .shadow_tile = spot_light_shadow_tiles_[spot_index],
            });
        }

        light_count_ = static_cast<std::uint32_t>(light_staging_.size());
        frame.light_count = light_count_;

        if (!light_staging_.empty() && !frame.lights_buffer.write(0, std::as_bytes(std::span{light_staging_}))) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::device_error));
        }

        frame.lights_have_shadow_tiles = shadow_tiles_in_use;
        lights_dirty_mask_ &= ~(1u << frame_index);
    } else {
        frame.light_count = light_count_;
    }

#pragma region Culling
    if (frame.indirect_command_count != 0) {
        if (frame.lod_chunk_count > cull_chunk_capacity_ * (cull_chunk_bytes / lod_chunk_bytes)) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
        }

        if (frame.indirect_command_count > maximum_cull_batch_count) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::capacity_exceeded));
        }

        if (resolve_layout(pipeline_graph_, frustum_cull_pipeline_) == VK_NULL_HANDLE) {
            clear_submissions();
            return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
        }
    }
#pragma endregion

#pragma region LightClustering
    if (clustered_lighting_ && (resolve_layout(pipeline_graph_, light_cull_pipeline_) == VK_NULL_HANDLE ||
                                resolve_layout(pipeline_graph_, light_cluster_pipeline_) == VK_NULL_HANDLE)) {
        clear_submissions();
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }
#pragma endregion

    last_frame_stats_ = FrameStats{
            .frame_checksum = frame_checksum,
            .submitted_triangle_count = submitted_triangle_count,
            .submitted_instance_count = static_cast<std::uint32_t>(frame.transform_count - resident_reserved_slots +
                                                                   resident_instance_count),
            .skin_job_count = static_cast<std::uint32_t>(frame.skin_jobs.size()),
            .skinned_vertex_count = frame.skinned_vertices,
            .skin_scratch_bytes_used = frame.skin_scratch_used,
            .skin_fallback_instance_count = frame.skin_fallback_instances,
            .indirect_command_count = frame.indirect_command_count,
            .opaque_indirect_count = frame.opaque_indirect_count,
            .double_sided_indirect_count = frame.double_sided_indirect_count,
            .mask_indirect_count = frame.mask_indirect_count,
            .blend_indirect_count = frame.blend_indirect_count,
            .model_submission_count = static_cast<std::uint32_t>(submitted_model_count()),
            .mesh_submission_count = static_cast<std::uint32_t>(submissions_.size()),
            .point_light_count = static_cast<std::uint32_t>(point_light_submissions_.size()),
            .spot_light_count = static_cast<std::uint32_t>(spot_light_submissions_.size()),
    };

    clear_submissions();

    if (!secondary && frame_query.has_results) {
        last_frame_timings_.valid = false;
        std::array<std::uint64_t, full_frame_query_count> results{};
        auto const query_result =
                vkGetQueryPoolResults(context_.device, frame_query.query_pool, 0, full_frame_query_count,
                                      sizeof(results), results.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);

        if (query_result == VK_NOT_READY) {
            warn("Timestamp queries not ready for frame {}", frame_index);
        }

        if (query_result == VK_SUCCESS) {
            last_frame_timings_.full_frame_ms =
                    static_cast<float>(results[1] - results[0]) * timestamp_period_ / 1'000'000.0F;

            read_overlay_timings(frame_query);
            last_frame_timings_.frame_serial = frame_query.serial;
            last_frame_timings_.valid = true;
        }

        frame_query.has_results = false;
    }

    if (!secondary && frame_pipeline_query.has_results) {
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

auto Renderer::record_occlusion_stats_clear(VkCommandBuffer command_buffer, RendererFrame const &frame) -> void {
    vkCmdFillBuffer(command_buffer, frame.occlusion_stats_buffer.buffer, 0,
                    VkDeviceSize{occlusion_stat_count} * sizeof(std::uint32_t), 0);
}

auto Renderer::record_meshlet_visibility_clear(VkCommandBuffer command_buffer, RendererFrame const &frame) -> void {
    vkCmdFillBuffer(command_buffer, frame.meshlet_visibility_buffer.buffer, 0,
                    VkDeviceSize{frame.meshlet_visibility_words} * sizeof(std::uint32_t), 0);
}

auto Renderer::record_resident_instance_uploads(VkCommandBuffer command_buffer, RendererFrame &frame) -> void {
    for (auto &upload: pending_resident_uploads_) {
        VkBufferCopy const region{.srcOffset = 0, .dstOffset = 0, .size = upload.size};
        vkCmdCopyBuffer(command_buffer, upload.staging.buffer, upload.destination, 1, &region);
        frame.retired_buffers.push_back(std::move(upload.staging));
    }

    if (!pending_resident_uploads_.empty()) {
        VkMemoryBarrier2 const uploaded{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        };
        VkDependencyInfo const dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = nullptr,
                .dependencyFlags = 0,
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &uploaded,
                .bufferMemoryBarrierCount = 0,
                .pBufferMemoryBarriers = nullptr,
                .imageMemoryBarrierCount = 0,
                .pImageMemoryBarriers = nullptr,
        };
        vkCmdPipelineBarrier2(command_buffer, &dependency);
    }

    pending_resident_uploads_.clear();

    for (auto &retired: retired_resident_buffers_) {
        frame.retired_buffers.push_back(std::move(retired));
    }
    retired_resident_buffers_.clear();

    for (auto set = resident_instance_sets_.begin(); set != resident_instance_sets_.end();) {
        if (frame_counter_ > set->second.last_used_frame + resident_set_idle_frames) {
            frame.retired_buffers.push_back(std::move(set->second.transforms));
            set = resident_instance_sets_.erase(set);
        } else {
            ++set;
        }
    }
}

auto Renderer::prepare_skin_upload(RendererFrame &frame) -> std::expected<void, RendererError> {
    if (frame.skin_jobs.empty()) {
        return {};
    }

    if (!frame.skin_upload_buffer.mapped()) {
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    frame.skin_palette_count = static_cast<std::uint32_t>(skin_palette_.size());
    auto const table = skin_chunk_table(frame.skin_jobs);
    frame.skin_chunk_total = table.back();

    auto const palette_bytes = std::as_bytes(std::span{skin_palette_});
    auto const jobs_bytes = std::as_bytes(std::span{frame.skin_jobs});
    auto const table_bytes = std::as_bytes(std::span{table});

    if (!palette_bytes.empty() && !frame.skin_upload_buffer.write(0, palette_bytes)) {
        return std::unexpected(make_error(RendererErrorType::device_error));
    }
    if (!frame.skin_upload_buffer.write(skin_jobs_offset(), jobs_bytes) ||
        !frame.skin_upload_buffer.write(skin_chunks_offset(), table_bytes)) {
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    return {};
}

auto Renderer::record_skin_upload(VkCommandBuffer command_buffer, RendererFrame const &frame) -> void {
    std::array<VkBufferCopy, 3> regions{};
    std::uint32_t count = 0;

    auto const add = [&](VkDeviceSize offset, VkDeviceSize size) {
        if (size != 0) {
            regions[count++] = VkBufferCopy{.srcOffset = offset, .dstOffset = offset, .size = size};
        }
    };

    add(0, VkDeviceSize{frame.skin_palette_count} * sizeof(glm::mat4));
    add(skin_jobs_offset(), frame.skin_jobs.size() * sizeof(GpuSkinJob));
    add(skin_chunks_offset(), (frame.skin_jobs.size() + 1) * sizeof(std::uint32_t));

    vkCmdCopyBuffer(command_buffer, frame.skin_upload_buffer.buffer, frame.skin_input_buffer.buffer, count,
                    regions.data());
}

auto Renderer::record_skin(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> std::expected<void, RendererError> {
    auto const command_buffer = pass_context.command_buffer;

    if (frame.skin_jobs.empty()) {
        return {};
    }

    auto const layout = resolve_layout(pipeline_graph_, skin_pipeline_);
    if (layout == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    bind_compute_node(pipeline_graph_, skin_pipeline_, command_buffer);
    gpu_resource_table_.bind(command_buffer, pass_context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, layout);

    auto const base = frame.skin_input_buffer.device_address;
    SkinPushConstants const pc{
            .jobs_address = base + skin_jobs_offset(),
            .job_first_chunk_address = base + skin_chunks_offset(),
            .palette_address = base,
            .job_count = static_cast<std::uint32_t>(frame.skin_jobs.size()),
            .chunk_count = frame.skin_chunk_total,
    };

    vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    dispatch_linear(command_buffer, pc.chunk_count);
    return {};
}

auto Renderer::record_instance_lods(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> std::expected<void, RendererError> {
    auto const command_buffer = pass_context.command_buffer;

    if (frame.lod_jobs.empty()) {
        return {};
    }

    auto const layout = resolve_layout(pipeline_graph_, instance_lod_pipeline_);
    if (layout == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    bind_compute_node(pipeline_graph_, instance_lod_pipeline_, command_buffer);
    gpu_resource_table_.bind(command_buffer, pass_context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, layout);

    InstanceLodPushConstants pc{
            .jobs_address = frame.lod_jobs_buffer.device_address,
            .chunks_address = frame.cull_chunks_buffer.device_address,
            .draws_address = frame.draw_buffer.device_address,
            .transforms_address = frame.transform_buffer.device_address,
            .indirect_address = frame.indirect_buffer.device_address,
            .camera_x = frame.lod_camera_position.x,
            .camera_y = frame.lod_camera_position.y,
            .camera_z = frame.lod_camera_position.z,
            .job_count = static_cast<std::uint32_t>(frame.lod_jobs.size()),
            .lod_distance_sq0 = lod_distances[0] * lod_distances[0],
            .lod_distance_sq1 = lod_distances[1] * lod_distances[1],
            .lod_distance_sq2 = lod_distances[2] * lod_distances[2],
            .chunk_count = frame.lod_chunk_count,
            .stage = 0,
            ._padding = 0,
    };

    constexpr std::array stages{0U, 1U, 2U};
    for (auto const stage: stages) {
        pc.stage = stage;
        vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
        dispatch_linear(command_buffer, stage == 1U ? pc.job_count : pc.chunk_count);
        record_compute_barrier(command_buffer);
    }

    return {};
}

auto Renderer::record_gpu_culling(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> std::expected<void, RendererError> {
    auto const command_buffer = pass_context.command_buffer;

    if (auto lods = record_instance_lods(pass_context, frame); !lods) {
        return lods;
    }

    if (frame.indirect_command_count != 0) {
        auto const layout = resolve_layout(pipeline_graph_, frustum_cull_pipeline_);

        if (layout == VK_NULL_HANDLE) {
            return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
        }

        bind_compute_node(pipeline_graph_, frustum_cull_pipeline_, command_buffer);
        gpu_resource_table_.bind(command_buffer, pass_context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, layout);

        CullPushConstants const cull_pc{
                .src_draws_address = frame.draw_buffer.device_address,
                .src_transforms_address = frame.transform_buffer.device_address,
                .batch_bounds_address = frame.batch_bounds_buffer.device_address,
                .src_indirect_address = frame.indirect_buffer.device_address,
                .dst_indirect_address = frame.culled_indirect_buffer.device_address,
                .dst_draws_address = frame.visible_draw_buffer.device_address,
                .dst_transforms_address = frame.visible_transform_buffer.device_address,
                .frustum_planes_address = frame.frustum_planes_buffer.device_address,
                .occlusion_address = frame.occlusion_views_buffer.device_address,
                .occlusion_candidates_address = frame.occlusion_candidates_buffer.device_address,
                .chunks_address = frame.cull_chunks_buffer.device_address,
                .merged_indirect_address = frame.merged_indirect_buffer.device_address,
                .late_indirect_address = frame.late_indirect_buffer.device_address,
                .occlusion_stats_address = frame.occlusion_stats_buffer.device_address,
                .batch_count = frame.indirect_command_count,
                .occludable_batch_count = batch_counts(frame).blend_first(),
                .flags = 0,
                .chunk_count = frame.cull_chunk_count,
        };

        record_cull_stages(command_buffer, layout, cull_pc);
    }

    return {};
}

auto Renderer::record_cluster_stats_clear(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> void {

    vkCmdFillBuffer(pass_context.command_buffer, frame.cluster_lights_buffer.buffer, 0, cluster_stats_bytes, 0);
}

auto Renderer::record_decal_cluster(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> std::expected<void, RendererError> {
    if (frame.decal_count == 0) {
        return {};
    }

    auto const command_buffer = pass_context.command_buffer;
    auto const layout = resolve_layout(pipeline_graph_, decal_cluster_pipeline_);

    if (layout == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    bind_compute_node(pipeline_graph_, decal_cluster_pipeline_, command_buffer);
    gpu_resource_table_.bind(command_buffer, pass_context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, layout);

    DecalClusterPushConstants const pc{
            .ubo_address = ubos_[pass_context.frame_index].device_address,
            .decals_address = frame.decals_buffer.device_address,
            .decal_clusters_address = frame.decal_clusters_buffer.device_address,
            .decal_count = frame.decal_count,
    };

    vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);

    vkCmdDispatch(command_buffer, cluster_tile_count(frame.cluster_grid), 1, 1);
    return {};
}

auto Renderer::record_light_cull(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> std::expected<void, RendererError> {
    auto const command_buffer = pass_context.command_buffer;
    auto const layout = resolve_layout(pipeline_graph_, light_cull_pipeline_);

    if (layout == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const visible_spheres_address = frame.visible_lights_buffer.device_address;
    auto const visible_indices_address =
            visible_spheres_address + (sizeof(glm::vec4) * VkDeviceSize{maximum_light_count});
    auto const visible_count_address =
            visible_indices_address + (sizeof(std::uint32_t) * VkDeviceSize{maximum_light_count});

    bind_compute_node(pipeline_graph_, light_cull_pipeline_, command_buffer);
    gpu_resource_table_.bind(command_buffer, pass_context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, layout);

    LightCullPushConstants const cull_pc{
            .ubo_address = ubos_[pass_context.frame_index].device_address,
            .lights_address = frame.lights_buffer.device_address,
            .frustum_planes_address = frame.frustum_planes_buffer.device_address,
            .visible_spheres_address = visible_spheres_address,
            .visible_light_indices_address = visible_indices_address,
            .visible_light_count_address = visible_count_address,
            .light_count = frame.light_count,
    };

    vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(cull_pc), &cull_pc);

    vkCmdDispatch(command_buffer, 1, 1, 1);
    return {};
}

auto Renderer::record_light_cluster(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> std::expected<void, RendererError> {
    auto const command_buffer = pass_context.command_buffer;
    auto const layout = resolve_layout(pipeline_graph_, light_cluster_pipeline_);

    if (layout == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    auto const visible_spheres_address = frame.visible_lights_buffer.device_address;
    auto const visible_indices_address =
            visible_spheres_address + (sizeof(glm::vec4) * VkDeviceSize{maximum_light_count});
    auto const visible_count_address =
            visible_indices_address + (sizeof(std::uint32_t) * VkDeviceSize{maximum_light_count});

    auto const cluster_stats_address = frame.cluster_lights_buffer.device_address;
    auto const cluster_lists_address = cluster_stats_address + cluster_stats_bytes;

    bind_compute_node(pipeline_graph_, light_cluster_pipeline_, command_buffer);
    gpu_resource_table_.bind(command_buffer, pass_context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, layout);

    LightClusterPushConstants const cluster_pc{
            .ubo_address = ubos_[pass_context.frame_index].device_address,
            .visible_spheres_address = visible_spheres_address,
            .visible_light_indices_address = visible_indices_address,
            .visible_light_count_address = visible_count_address,
            .cluster_lights_address = cluster_lists_address,
            .cluster_stats_address = cluster_stats_address,
    };

    vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(cluster_pc), &cluster_pc);

    vkCmdDispatch(command_buffer, cluster_tile_count(frame.cluster_grid), 1, 1);
    return {};
}

auto Renderer::record_cluster_stats_readback(render_pass::Context const &pass_context, RendererFrame &frame) -> void {
    auto const command_buffer = pass_context.command_buffer;

    VkBufferCopy2 const stats_region{
            .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            .srcOffset = 0,
            .dstOffset = 0,
            .size = cluster_stats_bytes,
    };

    VkCopyBufferInfo2 const stats_copy{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            .srcBuffer = frame.cluster_lights_buffer.buffer,
            .dstBuffer = frame.cluster_stats_readback_buffer.buffer,
            .regionCount = 1,
            .pRegions = &stats_region,
    };

    vkCmdCopyBuffer2(command_buffer, &stats_copy);

    frame.cluster_stats_grid = frame.cluster_grid;
    frame.cluster_stats_pending = true;
}

auto Renderer::consume_culled_readback(RendererFrame &frame) -> void {
    if (!frame.occlusion_stats_pending) {
        return;
    }

    frame.occlusion_stats_pending = false;

    if (auto invalidated = frame.occlusion_stats_readback_buffer.invalidate(0, VkDeviceSize{occlusion_stat_count} *
                                                                                       sizeof(std::uint32_t));
        !invalidated) {
        error("[Renderer] Failed to invalidate the occlusion statistics readback buffer");
        return;
    }

    auto const *stats = frame.occlusion_stats_readback_buffer.mapped_data_as<std::uint32_t const>();
    if (stats == nullptr) {
        return;
    }

    auto const candidates = stats[occlusion_stat_candidates];
    auto const late = stats[occlusion_stat_late];

    last_frame_stats_.visible_instance_count = stats[occlusion_stat_early] + late;
    last_frame_stats_.frustum_visible_instance_count = stats[occlusion_stat_frustum_visible];
    last_frame_stats_.early_instance_count = stats[occlusion_stat_early];
    last_frame_stats_.occlusion_candidate_count = candidates;
    last_frame_stats_.late_instance_count = late;
    last_frame_stats_.occluded_instance_count = candidates >= late ? candidates - late : 0U;
    last_frame_stats_.occlusion_stats_valid = frame.occlusion_stats_active;

    last_frame_stats_.deferred_meshlet_count = stats[occlusion_stat_deferred_meshlets];
    last_frame_stats_.occluded_meshlet_count = stats[occlusion_stat_occluded_meshlets];
    last_frame_stats_.meshlet_occlusion_stats_valid = frame.meshlet_occlusion_stats_active;
}

auto Renderer::add_colour_lut(std::string name, ColourLut const &lut) -> std::expected<std::int32_t, RendererError> {
    auto const pixels = lut.to_volume_rgba8();
    auto const debug_name = std::string{"renderer.colour_lut."} + name;

    // Graded colour is data, not a picture: UNORM so the sampler returns the stored values untouched.
    auto image = image_storage_.create_image(
            ImageCreateInfo{
                    .extent = VkExtent3D{.width = lut.size, .height = lut.size, .depth = lut.size},
                    .format = VK_FORMAT_R8G8B8A8_UNORM,
                    .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                    .image_type = VK_IMAGE_TYPE_3D,
                    .view_type = VK_IMAGE_VIEW_TYPE_3D,
                    .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_3d),
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .tiling = VK_IMAGE_TILING_OPTIMAL,
                    .mip_levels = 1,
                    .array_layers = 1,
                    .debug_name = debug_name,
            },
            pixels);

    if (!image) {
        return std::unexpected(make_image_error(image.error()));
    }

    colour_luts_.push_back(ColourLutEntry{.name = std::move(name), .image = *image, .size = lut.size});

    return static_cast<std::int32_t>(colour_luts_.size() - 1);
}

auto Renderer::colour_lut_name(std::int32_t lut) const noexcept -> std::string_view {
    if (lut < 0 || static_cast<std::size_t>(lut) >= colour_luts_.size()) {
        return {};
    }

    return colour_luts_[static_cast<std::size_t>(lut)].name;
}

auto Renderer::find_colour_lut(std::string_view name) const noexcept -> std::optional<std::int32_t> {
    auto const found = std::ranges::find(colour_luts_, name, &ColourLutEntry::name);
    if (found == colour_luts_.end()) {
        return std::nullopt;
    }

    return static_cast<std::int32_t>(found - colour_luts_.begin());
}

auto Renderer::load_colour_lut(std::string_view path) -> std::expected<std::int32_t, RendererError> {
    if (auto const existing = find_colour_lut(path)) {
        return *existing;
    }

    auto const bytes = read_resource(data_path(path));
    if (!bytes) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    auto const lut = parse_cube_lut(std::string_view{reinterpret_cast<char const *>(bytes->data()), bytes->size()});
    if (!lut) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    return add_colour_lut(std::string{path}, *lut);
}

auto Renderer::set_occlusion_culling(bool enabled) noexcept -> void {
    if (enabled != occlusion_culling_) {
        hiz_history_valid_ = false;
    }

    occlusion_culling_ = enabled;
}

auto Renderer::set_occlusion_test_mode(OcclusionTestMode mode) noexcept -> void {
    if (mode != occlusion_test_mode_) {
        hiz_history_valid_ = false;
    }

    occlusion_test_mode_ = mode;
}

auto Renderer::hiz_debug_view(std::uint32_t mip) const noexcept -> ImageHandle {
    if (!hiz_debug_view_supported_ || !hiz_.layout_initialised || hiz_.mip_count == 0) {
        return {};
    }

    return hiz_.mip_slots[std::min(mip, hiz_.mip_count - 1)].handle();
}

auto Renderer::create_hiz_pyramid(VkExtent2D depth_extent) -> std::expected<HizPyramid, RendererError> {
    HizExtent const depth{.width = depth_extent.width, .height = depth_extent.height};
    auto const image_extent = hiz_image_extent(depth);

    HizPyramid pyramid;
    pyramid.depth_extent = depth_extent;
    pyramid.mip_count = hiz_mip_count(depth);

    auto image = create_held_image(
            image_storage_,
            ImageCreateInfo{
                    .extent = VkExtent3D{.width = image_extent.width, .height = image_extent.height, .depth = 1},
                    .format = VK_FORMAT_R32_SFLOAT,
                    .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                    .image_type = VK_IMAGE_TYPE_2D,
                    .view_type = VK_IMAGE_VIEW_TYPE_2D,
                    .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                    .flags = 0,
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .tiling = VK_IMAGE_TILING_OPTIMAL,
                    .mip_levels = pyramid.mip_count,
                    .array_layers = 1,
                    .create_mip_layer_views = true,
                    .debug_name = "renderer.hiz",
            });

    if (!image) {
        return std::unexpected(make_image_error(image.error()));
    }

    pyramid.image = std::move(*image);

    auto const *hiz_image = pyramid.image.get();

    for (std::uint32_t mip = 0; mip < pyramid.mip_count; ++mip) {
        auto const view = hiz_image->mip_layer_view(mip, 0);

        auto mip_slot = register_held_view(image_storage_, ImageViewRegistration{
                                                                   .sampled_2d = view,
                                                                   .storage_2d = view,
                                                           });

        if (!mip_slot) {
            return std::unexpected(make_image_error(mip_slot.error()));
        }

        pyramid.mip_slots[mip] = std::move(*mip_slot);
    }

    return pyramid;
}

auto Renderer::max_samples() const noexcept -> VkSampleCountFlagBits {
    auto const supported = context_.supported_sample_counts;

    for (auto const candidate: {VK_SAMPLE_COUNT_64_BIT, VK_SAMPLE_COUNT_32_BIT, VK_SAMPLE_COUNT_16_BIT,
                                VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_2_BIT}) {
        if ((supported & candidate) != 0) {
            return candidate;
        }
    }

    return VK_SAMPLE_COUNT_1_BIT;
}

auto Renderer::set_samples(VkSampleCountFlagBits samples) noexcept -> void {
    auto const max = max_samples();

    pending_samples_ = samples > max ? max : samples;
}

auto Renderer::occlusion_culling_supported() const noexcept -> bool {
    return samples_ == VK_SAMPLE_COUNT_1_BIT || context_.depth_resolve_min_supported;
}

auto Renderer::set_cluster_grid(ClusterGridSettings const &grid) -> std::expected<void, std::string> {
    if (auto valid = validate_cluster_grid(grid); !valid) {
        return valid;
    }

    cluster_grid_ = grid;
    return {};
}

auto Renderer::prepare_cluster_buffers(RendererFrame &frame) -> std::expected<void, RendererError> {
    if (frame.cluster_stats_pending) {
        frame.cluster_stats_pending = false;

        if (auto invalidated = frame.cluster_stats_readback_buffer.invalidate(0, cluster_stats_bytes); !invalidated) {
            error("[Renderer] Failed to invalidate the cluster statistics readback buffer");
        } else if (auto const *stats = frame.cluster_stats_readback_buffer.mapped_data_as<std::uint32_t const>();
                   stats != nullptr) {
            last_cluster_stats_ = ClusterStats{
                    .grid = frame.cluster_stats_grid,
                    .occupied_clusters = stats[0],
                    .overflowing_clusters = stats[1],
                    .maximum_lights = stats[2],
                    .stored_lights = stats[3],
                    .valid = true,
            };
        }
    }

    if (!clustered_lighting_) {
        last_cluster_stats_ = {};
    }

    if (!frame.cluster_stats_readback_buffer.valid()) {
        auto readback = Buffer::create(context_, BufferCreateInfo{
                                                         .size = cluster_stats_bytes,
                                                         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                         .memory = BufferMemory::readback,
                                                         .debug_name = "renderer.cluster_stats_readback",
                                                 });

        if (!readback) {
            return std::unexpected(make_device_error(readback.error()));
        }

        frame.cluster_stats_readback_buffer = std::move(*readback);
    }

    if (auto const cells = cluster_count(cluster_grid_); !frame.decal_clusters_buffer.valid() || frame.decal_cluster_cells != cells) {
        auto decal_clusters = create_shared_buffer(
                context_, BufferCreateInfo{
                                  .size = sizeof(std::uint32_t) * VkDeviceSize{cells} * (1 + decal_cluster_capacity),
                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  .memory = BufferMemory::device,
                                  .debug_name = "renderer.frame_decal_clusters",
                          });

        if (!decal_clusters) {
            return std::unexpected(make_device_error(decal_clusters.error()));
        }

        frame.decal_clusters_buffer = std::move(*decal_clusters);
        frame.decal_cluster_cells = cells;
    }

    if (frame.cluster_lights_buffer.valid() && frame.cluster_grid == cluster_grid_) {
        return {};
    }

    auto cluster_lights = create_shared_buffer(
            context_, BufferCreateInfo{
                              .size = cluster_buffer_bytes(cluster_grid_),
                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              .memory = BufferMemory::device,
                              .debug_name = "renderer.frame_cluster_lights",
                      });

    if (!cluster_lights) {
        return std::unexpected(make_device_error(cluster_lights.error()));
    }

    frame.cluster_lights_buffer = std::move(*cluster_lights);
    frame.cluster_grid = cluster_grid_;

    return {};
}

auto Renderer::resolve_frame_targets(RendererFrame const &frame) const -> std::expected<FrameTargets, RendererError> {
    FrameTargets const targets{
            .shadow_atlas = shadow_atlas_.get(),
            .local_shadow_atlas = local_shadow_atlas_.get(),
            .viewport = frame.viewport_target.get(),
            .extent = extent_,
            .multisampled = samples_ != VK_SAMPLE_COUNT_1_BIT,
    };

    auto const usable = [](Image const *image) { return image != nullptr && image->valid(); };

    if (!usable(targets.shadow_atlas) || !usable(targets.local_shadow_atlas) || !usable(targets.viewport)) {
        return std::unexpected(make_error(RendererErrorType::image_error));
    }

    if (targets.extent.width == 0 || targets.extent.height == 0) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    return targets;
}

auto Renderer::early_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers {
    return render_pass::DrawBuffers{
            .draws = frame.visible_draw_buffer,
            .transforms = frame.visible_transform_buffer,
            .indirect = frame.culled_indirect_buffer,
            .index_buffer = geometry_arena_.bindable_buffer(),
    };
}

auto Renderer::late_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers {
    return render_pass::DrawBuffers{
            .draws = frame.visible_draw_buffer,
            .transforms = frame.visible_transform_buffer,
            .indirect = frame.late_indirect_buffer,
            .index_buffer = geometry_arena_.bindable_buffer(),
    };
}

auto Renderer::forward_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers {
    return render_pass::DrawBuffers{
            .draws = frame.visible_draw_buffer,
            .transforms = frame.visible_transform_buffer,
            .indirect = frame.occlusion_active ? frame.merged_indirect_buffer : frame.culled_indirect_buffer,
            .index_buffer = geometry_arena_.bindable_buffer(),
    };
}

auto Renderer::batch_counts(RendererFrame const &frame) noexcept -> render_pass::DrawCounts {
    return render_pass::DrawCounts{
            .opaque = frame.opaque_indirect_count,
            .double_sided = frame.double_sided_indirect_count,
            .mask = frame.mask_indirect_count,
            .blend = frame.blend_indirect_count,
    };
}

auto Renderer::record_shadow_pass(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> std::expected<void, RendererError> {
    auto const frame_index = pass_context.frame_index;

    auto const result = render_pass::shadow(
            pass_context,
            render_pass::ShadowPassInfo{
                    .draws =
                            {
                                    .draws = frame.draw_buffer,
                                    .transforms = frame.transform_buffer,
                                    .indirect = frame.indirect_buffer,
                                    .index_buffer = geometry_arena_.bindable_buffer(),
                            },
                    .counts = batch_counts(frame),
                    .opaque_cascade_counts = frame.shadow_opaque_indirect_count,
                    .double_sided_cascade_counts = frame.shadow_double_sided_indirect_count,
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

auto Renderer::prepare_local_shadow_cull_buffers(RendererFrame &frame) -> std::expected<void, RendererError> {
    frame.local_shadow_culled = false;

    auto const render_tile_count = static_cast<std::uint32_t>(frame.local_shadow_render_tiles.size());

    if (!local_shadow_settings_.gpu_culling || render_tile_count == 0 ||
        resolve_layout(pipeline_graph_, shadow_cull_pipeline_) == VK_NULL_HANDLE) {
        return {};
    }

    auto const batches = batch_counts(frame).blend_first();

    if (batches == 0 || frame.transform_count == 0) {
        return {};
    }

    constexpr std::uint64_t maximum_entries = std::uint64_t{1} << 27;
    auto const survivors_needed = std::uint64_t{render_tile_count} * frame.transform_count;
    auto const batches_needed = std::uint64_t{render_tile_count} * batches;

    if (survivors_needed > maximum_entries || batches_needed > maximum_entries) {
        return {};
    }

    auto const grow = [&](Buffer &buffer, std::uint64_t &capacity, std::uint64_t needed, std::uint64_t entry_bytes,
                          VkBufferUsageFlags extra_usage, char const *name) -> std::expected<void, RendererError> {
        if (buffer.valid() && capacity >= needed) {
            return {};
        }

        auto const entries = std::bit_ceil(std::max<std::uint64_t>(needed, 1024));
        auto created = create_shared_buffer(context_, BufferCreateInfo{
                                                              .size = entries * entry_bytes,
                                                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                                       extra_usage,
                                                              .memory = BufferMemory::device,
                                                              .debug_name = name,
                                                      });

        if (!created) {
            return std::unexpected(make_device_error(created.error()));
        }

        if (buffer.valid()) {
            frame.retired_buffers.push_back(std::move(buffer));
        }

        buffer = std::move(*created);
        capacity = entries;
        return {};
    };

    std::uint64_t counts_capacity = frame.local_shadow_batch_capacity;

    if (auto grown = grow(frame.local_shadow_survivors_buffer, frame.local_shadow_survivors_capacity,
                          survivors_needed, sizeof(std::uint32_t), 0, "renderer.frame_local_shadow_survivors");
        !grown) {
        return grown;
    }

    if (auto grown = grow(frame.local_shadow_counts_buffer, counts_capacity, batches_needed, sizeof(std::uint32_t),
                          0, "renderer.frame_local_shadow_counts");
        !grown) {
        return grown;
    }

    std::uint64_t indirect_capacity = frame.local_shadow_batch_capacity;

    if (auto grown = grow(frame.local_shadow_indirect_buffer, indirect_capacity, batches_needed,
                          sizeof(GpuDrawCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                          "renderer.frame_local_shadow_indirect");
        !grown) {
        return grown;
    }

    frame.local_shadow_batch_capacity = std::min(counts_capacity, indirect_capacity);
    frame.local_shadow_cull_batches = batches;
    frame.local_shadow_instance_stride = frame.transform_count;
    frame.local_shadow_culled = true;
    local_shadow_stats_.culled = true;

    return {};
}

auto Renderer::record_local_shadow_cull(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> std::expected<void, RendererError> {
    if (!frame.local_shadow_culled || frame.local_shadow_render_tiles.empty()) {
        return {};
    }

    auto const command_buffer = pass_context.command_buffer;
    auto const layout = resolve_layout(pipeline_graph_, shadow_cull_pipeline_);

    if (layout == VK_NULL_HANDLE) {
        return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
    }

    bind_compute_node(pipeline_graph_, shadow_cull_pipeline_, command_buffer);
    gpu_resource_table_.bind(command_buffer, pass_context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, layout);

    ShadowCullPushConstants pc{
            .src_transforms_address = frame.transform_buffer.device_address,
            .batch_bounds_address = frame.batch_bounds_buffer.device_address,
            .src_indirect_address = frame.indirect_buffer.device_address,
            .tile_planes_address = frame.local_shadow_planes_buffer.device_address,
            .survivors_address = frame.local_shadow_survivors_buffer.device_address,
            .counts_address = frame.local_shadow_counts_buffer.device_address,
            .dst_indirect_address = frame.local_shadow_indirect_buffer.device_address,
            .source_batch_count = frame.indirect_command_count,
            .batch_count = frame.local_shadow_cull_batches,
            .chunk_count = frame.cull_chunk_count,
            .instance_capacity = frame.local_shadow_instance_stride,
            .stage = 0,
            ._padding = 0,
    };

    auto const tiles = static_cast<std::uint32_t>(frame.local_shadow_render_tiles.size());
    auto const batch_groups = (pc.batch_count + 255U) / 256U;
    auto const chunk_width = std::min(std::max(pc.chunk_count, 1U), cull_dispatch_width);
    auto const chunk_height = (std::max(pc.chunk_count, 1U) + chunk_width - 1) / chunk_width;

    constexpr std::uint32_t clear_stage = 0;
    constexpr std::uint32_t cull_stage = 1;
    constexpr std::uint32_t finalize_stage = 2;

    pc.stage = clear_stage;
    vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    vkCmdDispatch(command_buffer, batch_groups, 1, tiles);
    record_compute_barrier(command_buffer);

    pc.stage = cull_stage;
    vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    vkCmdDispatch(command_buffer, chunk_width, chunk_height, tiles);
    record_compute_barrier(command_buffer);

    pc.stage = finalize_stage;
    vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    vkCmdDispatch(command_buffer, batch_groups, 1, tiles);

    return {};
}

auto Renderer::record_local_shadow_pass(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> std::expected<void, RendererError> {
    if (frame.local_shadow_render_tiles.empty()) {
        return {};
    }

    auto const frame_index = pass_context.frame_index;

    auto const result = render_pass::local_shadow(
            pass_context,
            render_pass::LocalShadowPassInfo{
                    .draws =
                            {
                                    .draws = frame.draw_buffer,
                                    .transforms = frame.transform_buffer,
                                    .indirect = frame.indirect_buffer,
                                    .index_buffer = geometry_arena_.bindable_buffer(),
                            },
                    .counts = batch_counts(frame),
                    .opaque_count = frame.shadow_opaque_indirect_count[0],
                    .double_sided_count = frame.shadow_double_sided_indirect_count[0],
                    .mask_count = frame.shadow_mask_indirect_count[0],
                    .tiles = frame.local_shadow_render_tiles,
                    .tile_indirect = frame.local_shadow_culled ? &frame.local_shadow_indirect_buffer : nullptr,
                    .survivors_address = frame.local_shadow_survivors_buffer.device_address,
                    .cull_batches = frame.local_shadow_cull_batches,
                    .instance_stride = frame.local_shadow_instance_stride,
                    .meshlet_culling = meshlet_culling_,
                    .tiles_address = frame.local_shadow_tiles_buffer.device_address,
                    .cull_planes_address = frame.local_shadow_planes_buffer.device_address,
                    .materials_address = material_storage_.device_address(),
                    .ubo_address = ubos_[frame_index].device_address,
                    .lights_address = frame.lights_buffer.device_address,
                    .opaque_pipeline = shadow_pipeline_,
                    .mask_pipeline = shadow_mask_pipeline_,
                    .opaque_instanced_pipeline = shadow_instanced_pipeline_,
                    .mask_instanced_pipeline = shadow_mask_instanced_pipeline_,
                    .depth_bias_constant = local_shadow_settings_.depth_bias_constant,
                    .depth_bias_slope = local_shadow_settings_.depth_bias_slope,
            });

    if (!result) {
        return std::unexpected(result.error());
    }

    local_shadow_atlas_initialized_ = true;
    return {};
}

auto Renderer::record_depth_prepass(render_pass::Context const &pass_context, RendererFrame const &frame,
                                    FrameTargets const &targets, render_pass::DepthPrepassPhase phase)
        -> std::expected<void, RendererError> {
    bool const late = phase == render_pass::DepthPrepassPhase::late;

    auto const frame_index = pass_context.frame_index;

    auto const meshlet_view_address = frame.meshlet_occlusion_active ? frame.occlusion_views_buffer.device_address +
                                                                               (late ? sizeof(GpuOcclusionView) : 0U)
                                                                     : VkDeviceAddress{0};
    auto const meshlet_flags =
            !frame.meshlet_occlusion_active
                    ? 0U
                    : (late ? render_pass::cull_occlusion | render_pass::cull_skip_recorded | render_pass::cull_record |
                                       render_pass::cull_stats
                            : render_pass::cull_occlusion | render_pass::cull_record | render_pass::cull_stats);

    return render_pass::depth_prepass(pass_context,
                                      render_pass::DepthPrepassInfo{
                                              .extent = targets.extent,
                                              .samples = samples_,
                                              .phase = phase,
                                              .draws = late ? late_view_draws(frame) : early_view_draws(frame),
                                              .counts = batch_counts(frame),
                                              .cull_planes_address = frame.frustum_planes_buffer.device_address,
                                              .materials_address = material_storage_.device_address(),
                                              .ubo_address = ubos_[frame_index].device_address,
                                              .lights_address = frame.lights_buffer.device_address,
                                              .occlusion_view_address = meshlet_view_address,
                                              .extra_cull_flags = meshlet_flags,
                                              .opaque_pipeline = depth_prepass_pipeline_,
                                              .mask_pipeline = depth_prepass_mask_pipeline_,
                                              .opaque_instanced_pipeline = depth_prepass_instanced_pipeline_,
                                              .mask_instanced_pipeline = depth_prepass_mask_instanced_pipeline_,
                                              .meshlet_culling = meshlet_culling_,
                                      });
}

auto Renderer::record_environment_pass(render_pass::Context const &pass_context, RendererFrame const &frame) -> void {
    auto const command_buffer = pass_context.command_buffer;

    environment_.record(command_buffer, gpu_resource_table_, pass_context.frame_index,
                        ubos_[pass_context.frame_index].device_address);

    static_cast<void>(frame);
}

auto Renderer::record_hiz_build(render_pass::Context const &pass_context, FrameTargets const &targets,
                                std::uint32_t depth_texture_index) -> std::expected<void, RendererError> {
    auto const *hiz_image = hiz_.image.get();

    if (hiz_image == nullptr || hiz_.mip_count == 0 || hiz_.mip_count > hiz_max_mip_count) {
        return std::unexpected(make_error(RendererErrorType::image_error));
    }

    std::array<std::uint32_t, hiz_max_mip_count> mip_texture_indices{};

    for (std::uint32_t mip = 0; mip < hiz_.mip_count; ++mip) {
        mip_texture_indices[mip] = hiz_.mip_slots[mip].handle().index;
    }

    auto const built = render_pass::build_hiz(
            pass_context, render_pass::HizBuildInfo{
                                  .source_texture_index = depth_texture_index,
                                  .depth_extent = targets.extent,
                                  .hiz = *hiz_image,
                                  .mip_texture_indices = std::span{mip_texture_indices}.first(hiz_.mip_count),
                                  .pipeline = hiz_build_pipeline_,
                          });

    if (!built) {
        return std::unexpected(built.error());
    }

    hiz_.layout_initialised = true;
    return {};
}

auto Renderer::record_occlusion_cull_pass(render_pass::Context const &pass_context, RendererFrame const &frame)
        -> std::expected<void, RendererError> {
    auto const command_buffer = pass_context.command_buffer;

    if (frame.indirect_command_count != 0) {
        auto const layout = resolve_layout(pipeline_graph_, occlusion_cull_pipeline_);

        if (layout == VK_NULL_HANDLE) {
            return std::unexpected(make_error(RendererErrorType::invalid_pipeline));
        }

        bind_compute_node(pipeline_graph_, occlusion_cull_pipeline_, command_buffer);
        gpu_resource_table_.bind(command_buffer, pass_context.frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, layout);

        CullPushConstants const cull_pc{
                .src_draws_address = frame.draw_buffer.device_address,
                .src_transforms_address = frame.transform_buffer.device_address,
                .batch_bounds_address = frame.batch_bounds_buffer.device_address,
                .src_indirect_address = frame.indirect_buffer.device_address,
                .dst_indirect_address = frame.culled_indirect_buffer.device_address,
                .dst_draws_address = frame.visible_draw_buffer.device_address,
                .dst_transforms_address = frame.visible_transform_buffer.device_address,
                .frustum_planes_address = frame.frustum_planes_buffer.device_address,
                .occlusion_address = frame.occlusion_views_buffer.device_address + sizeof(GpuOcclusionView),
                .occlusion_candidates_address = frame.occlusion_candidates_buffer.device_address,
                .chunks_address = frame.cull_chunks_buffer.device_address,
                .merged_indirect_address = frame.merged_indirect_buffer.device_address,
                .late_indirect_address = frame.late_indirect_buffer.device_address,
                .occlusion_stats_address = frame.occlusion_stats_buffer.device_address,
                .batch_count = frame.indirect_command_count,
                .occludable_batch_count = batch_counts(frame).blend_first(),
                .flags = frame.meshlet_occlusion_active ? cull_flag_late_union_meshlet_batches : 0U,
                .chunk_count = frame.cull_chunk_count,
        };

        record_cull_stages(command_buffer, layout, cull_pc);
    }

    return {};
}

auto Renderer::record_occlusion_stats_readback(VkCommandBuffer command_buffer, RendererFrame &frame) -> void {
    auto const stats_size = VkDeviceSize{occlusion_stat_count} * sizeof(std::uint32_t);

    VkBufferCopy2 const region{
            .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            .srcOffset = 0,
            .dstOffset = 0,
            .size = stats_size,
    };

    VkCopyBufferInfo2 const copy{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            .srcBuffer = frame.occlusion_stats_buffer.buffer,
            .dstBuffer = frame.occlusion_stats_readback_buffer.buffer,
            .regionCount = 1,
            .pRegions = &region,
    };

    vkCmdCopyBuffer2(command_buffer, &copy);

    frame.occlusion_stats_pending = true;
    frame.occlusion_stats_active = frame.occlusion_active;
    frame.meshlet_occlusion_stats_active = frame.meshlet_occlusion_active;
}

auto Renderer::ambient_occlusion_info(FrameTargets const &targets, std::uint32_t frame_index,
                                      std::uint32_t depth_texture_index, std::uint32_t raw_texture_index,
                                      std::uint32_t denoised_texture_index) const -> render_pass::AmbientOcclusionInfo {
    return render_pass::AmbientOcclusionInfo{
            .extent = targets.extent,
            .depth_texture_index = depth_texture_index,
            .raw_ao_texture_index = raw_texture_index,
            .denoised_ao_texture_index = denoised_texture_index,
            .point_sampler_index = sampler_storage_.nearest_clamp().index,
            .ubo_address = ubos_[frame_index].device_address,
            .gtao_pipeline = gtao_pipeline_,
            .denoise_pipeline = gtao_denoise_pipeline_,
            .radius_view = ao_settings_.radius,
            .falloff_range = ao_settings_.falloff_range,
            .slice_count = ao_settings_.slice_count,
            .step_count = ao_settings_.step_count,
            .denoise_depth_sigma = ao_settings_.denoise_depth_sigma,
    };
}

auto Renderer::record_forward_pass(render_pass::Context const &pass_context, RendererFrame const &frame,
                                   FrameTargets const &targets, std::uint32_t ao_texture_index,
                                   std::uint32_t hdr_texture_index, render_pass::Callback scene_overlays,
                                   render_pass::ForwardPhase phase, std::uint32_t scene_colour_index,
                                   std::uint32_t scene_depth_index)
        -> std::expected<render_pass::HdrTextureIndex, RendererError> {
    auto const frame_index = pass_context.frame_index;

    return render_pass::forward_geometry(
            pass_context,
            render_pass::ForwardGeometryInfo{
                    .output_hdr = {.index = hdr_texture_index},
                    .phase = phase,
                    .blend_runs = frame.blend_runs,
                    .scene_colour_index = scene_colour_index,
                    .scene_depth_index = scene_depth_index,
                    .extent = targets.extent,
                    .samples = samples_,
                    .draws = forward_view_draws(frame),
                    .counts = batch_counts(frame),
                    .cull_planes_address = frame.frustum_planes_buffer.device_address,
                    .materials_address = material_storage_.device_address(),
                    .ubo_address = ubos_[frame_index].device_address,
                    .lights_address = frame.lights_buffer.device_address,
                    .light_count = frame.light_count,
                    .cluster_lights_address = frame.cluster_lights_buffer.device_address + cluster_stats_bytes,
                    .occlusion_view_address =
                            frame.meshlet_occlusion_active
                                    ? frame.occlusion_views_buffer.device_address + sizeof(GpuOcclusionView)
                                    : VkDeviceAddress{0},
                    .extra_cull_flags = frame.meshlet_occlusion_active ? render_pass::cull_replay : 0U,
                    .pipeline_statistics_query_pool = pipeline_stats_hold_frames_ != 0 && !secondary_view_active_
                                                              ? pipeline_stat_queries_[frame_index].query_pool
                                                              : VkQueryPool{VK_NULL_HANDLE},
                    .meshlet_culling = meshlet_culling_,
                    .opaque_pipeline = forward_outline_pipeline_,
                    .blend_pipeline = forward_blend_pipeline_,
                    .opaque_instanced_pipeline = forward_outline_instanced_pipeline_,
                    .blend_instanced_pipeline = forward_blend_instanced_pipeline_,
                    .skybox_pipeline = skybox_pipeline_,
                    .draw_skybox = (environment_.ubo_block().flags & environment_flag::skybox) != 0U,
                    .ao_texture_index = ao_texture_index,
                    .ao_sampler_index = sampler_storage_.linear_clamp().index,
                    .outline_mask = true,
            },
            scene_overlays);
}

auto Renderer::record_bloom_pass(render_pass::Context const &pass_context, FrameTargets const &targets,
                                 render_pass::HdrTextureIndex hdr, Image const &bloom_image,
                                 std::array<std::uint32_t, render_pass::bloom_mip_count> const &mip_texture_indices)
        -> std::expected<std::optional<render_pass::BloomTextureIndex>, RendererError> {
    return render_pass::bloom(pass_context, render_pass::BloomPassInfo{
                                                    .enabled = bloom_settings_.enabled,
                                                    .input_hdr = hdr,
                                                    .target = bloom_settings_.enabled ? &bloom_image : nullptr,
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

auto Renderer::record_frame_end(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void {
    auto &frame_query = timestamp_queries_[frame_index];

    vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, frame_query.query_pool, 1);

    frame_query.has_results = true;
    frame_query.serial = ++recorded_frame_count_;
    pipeline_stat_queries_[frame_index].has_results = pipeline_stats_hold_frames_ != 0;
    if (pipeline_stats_hold_frames_ != 0) {
        --pipeline_stats_hold_frames_;
    }
}

auto Renderer::make_pass_context(VkCommandBuffer command_buffer, std::uint32_t frame_index, bool compute_only)
        -> render_pass::Context {
    return render_pass::Context{
            .command_buffer = command_buffer,
            .frame_index = frame_index,
            .pipeline_graph = pipeline_graph_,
            .resource_table = gpu_resource_table_,
            .timestamp_query_pool = timestamp_queries_[frame_index].query_pool,
            .compute_only = compute_only,
    };
}

auto Renderer::record_overlay_prepares(render_pass::Context const &pass_context) -> void {
    auto const command_buffer = pass_context.command_buffer;
    auto &frame_query = timestamp_queries_[pass_context.frame_index];

    frame_query.overlays.clear();

    for (auto &entry: overlays_.all()) {
        frame_query.overlays.push_back(RecordedOverlay{
                .name = entry.desc.name,
                .stage = entry.desc.stage,
                .slot = entry.slot,
        });

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pass_context.timestamp_query_pool,
                             overlay_query(entry.slot, 0));

        if (entry.desc.prepare) {
            ZoneTransientN(cpu_zone, entry.desc.name.c_str(), true);
            TracyVkZoneTransient(context_.host_query_context.context, gpu_zone, command_buffer, entry.desc.name.c_str(),
                                 true);

            static_cast<void>(entry.desc.prepare(OverlayPrepareContext{
                    .command_buffer = command_buffer,
                    .frame_index = pass_context.frame_index,
            }));
        }

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pass_context.timestamp_query_pool,
                             overlay_query(entry.slot, 1));
    }
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

auto Renderer::create_frame_targets(std::uint32_t frame_index, VkExtent2D extent)
        -> std::expected<OwnedFrameTargets, RendererError> {
    OwnedFrameTargets targets;

    auto const viewport_target_name = std::format("renderer.viewport_target_{}", frame_index);
    auto viewport_target = create_held_image(
            image_storage_, ImageCreateInfo{
                                    .extent = VkExtent3D{.width = extent.width, .height = extent.height, .depth = 1},
                                    .format = swapchain_format_,
                                    .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
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

    return targets;
}

auto Renderer::resize(VkExtent2D extent) -> std::expected<void, RendererError> {
    if (extent.width == 0 || extent.height == 0) {
        return {};
    }

    if (extent.width == extent_.width && extent.height == extent_.height) {
        return {};
    }

    perf_events::record(PerfEvent::render_resize);

    if (auto waited = wait_idle(); !waited) {
        return std::unexpected(waited.error());
    }

    std::vector<OwnedFrameTargets> replacements;
    replacements.reserve(frames_.size());

    for (std::size_t index = 0; index < frames_.size(); ++index) {
        auto targets = create_frame_targets(static_cast<std::uint32_t>(index), extent);

        if (!targets) {
            return std::unexpected(targets.error());
        }

        replacements.push_back(std::move(*targets));
    }

    auto hiz = create_hiz_pyramid(extent);

    if (!hiz) {
        return std::unexpected(hiz.error());
    }

    for (std::size_t index = 0; index < frames_.size(); ++index) {
        auto &frame = frames_[index];
        auto &targets = replacements[index];

        frame.viewport_target = std::move(targets.viewport_target);
    }

    hiz_ = std::move(*hiz);
    hiz_history_valid_ = false;

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
    perf_events::record(PerfEvent::device_wait_idle);
    auto const result = wait_idle_bounded(context_.device, "Renderer::wait_idle");

    if (is_device_failure(result)) {
        context_.device_lost.store(true, std::memory_order_release);
    }

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

    auto const draw_size = static_cast<VkDeviceSize>(frame.draw_count) * sizeof(GpuDraw);
    auto const transform_size = static_cast<VkDeviceSize>(frame.transform_count) * sizeof(glm::mat4);
    auto const indirect_size = static_cast<VkDeviceSize>(frame.indirect_commands.size()) * sizeof(GpuDrawCommand);
    auto const batch_bounds_size = static_cast<VkDeviceSize>(frame.batch_bounds.size()) * sizeof(GpuCullBounds);

    std::vector<VkBufferCopy2> draw_regions;
    std::vector<VkBufferCopy2> transform_regions;
    draw_regions.reserve(frame.cpu_instance_ranges.size());
    transform_regions.reserve(frame.cpu_instance_ranges.size());

    for (auto const &[first, count]: frame.cpu_instance_ranges) {
        auto const draw_offset = static_cast<VkDeviceSize>(first) * sizeof(GpuDraw);
        auto const draw_bytes = static_cast<VkDeviceSize>(count) * sizeof(GpuDraw);
        auto const transform_offset = static_cast<VkDeviceSize>(first) * sizeof(glm::mat4);
        auto const transform_bytes = static_cast<VkDeviceSize>(count) * sizeof(glm::mat4);

        if (!frame.upload_buffer.flush(frame.draw_upload_offset + draw_offset, draw_bytes) ||
            !frame.upload_buffer.flush(frame.transform_upload_offset + transform_offset, transform_bytes)) {
            return std::unexpected(make_error(RendererErrorType::device_error));
        }

        draw_regions.push_back(VkBufferCopy2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                .pNext = nullptr,
                .srcOffset = frame.draw_upload_offset + draw_offset,
                .dstOffset = draw_offset,
                .size = draw_bytes,
        });
        transform_regions.push_back(VkBufferCopy2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                .pNext = nullptr,
                .srcOffset = frame.transform_upload_offset + transform_offset,
                .dstOffset = transform_offset,
                .size = transform_bytes,
        });
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

    auto const copy_regions = [&](VkBuffer destination, std::vector<VkBufferCopy2> const &regions) {
        if (regions.empty()) {
            return;
        }
        VkCopyBufferInfo2 const copy_info{
                .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                .pNext = nullptr,
                .srcBuffer = frame.upload_buffer.buffer,
                .dstBuffer = destination,
                .regionCount = static_cast<std::uint32_t>(regions.size()),
                .pRegions = regions.data(),
        };
        vkCmdCopyBuffer2(command_buffer, &copy_info);
    };

    copy_regions(frame.draw_buffer.buffer, draw_regions);
    copy_regions(frame.transform_buffer.buffer, transform_regions);

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

    if (copy_count == 0 && draw_regions.empty()) {
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
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
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
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
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
                .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = frame.indirect_buffer.buffer,
                .offset = 0,
                .size = indirect_size,
        };
    }

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
    if (hold_submissions_) {
        return;
    }

    submissions_.clear();
    model_submissions_.clear();
    outlined_submission_count_ = 0;
    slot_override_submissions_.clear();
    instanced_submissions_.clear();
    instance_transforms_.clear();
    instance_palette_offsets_.clear();
    skin_palette_.clear();
    resident_slots_this_frame_ = 0;
    resident_jobs_this_frame_ = 0;
    decal_submissions_.clear();
    point_light_submissions_.clear();
    spot_light_submissions_.clear();
}
