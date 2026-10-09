#pragma once

#include <atomic>
#include <queue>
#include <volk.h>

#include <glm/glm.hpp>

#include <BS_thread_pool.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "assets/asset_registry.hxx"
#include "core/paths.hxx"
#include "assets/geometry_arena.hxx"
#include "assets/load_model.hxx"
#include "assets/material_storage.hxx"
#include "assets/mesh_create_info.hxx"
#include "assets/mesh_sink.hxx"
#include "assets/mesh_storage.hxx"
#include "assets/meshlet.hxx"
#include "assets/model.hxx"
#include "assets/model_sink.hxx"
#include "assets/model_storage.hxx"
#include "assets/model_streamer.hxx"
#include "assets/shader_change_queue.hxx"
#include "assets/slang_compiler.hxx"
#include "assets/texture_streamer.hxx"
#include "core/config.hxx"
#include "core/error_context.hxx"
#include "core/forward.hxx"
#include "core/renderer_error.hxx"
#include "gpu/buffer.hxx"
#include "gpu/gpu_resource_table.hxx"
#include "gpu/image_storage.hxx"
#include "gpu/sampler_storage.hxx"
#include "gpu/skinning.hxx"
#include "gpu/submission_plan.hxx"
#include "rendering/cluster_grid.hxx"
#include "rendering/environment.hxx"
#include "rendering/forward_target.hxx"
#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/compiler.hxx"
#include "rendering/frame_graph/frame_graph.hxx"
#include "rendering/frame_graph/pass_profiler.hxx"
#include "rendering/frame_graph/transient_allocator.hxx"
#include "rendering/frame_graph/view.hxx"
#include "rendering/hiz_occlusion.hxx"
#include "rendering/meshlet_visibility.hxx"
#include "rendering/pipeline_graph_repository.hxx"
#include "rendering/render_passes.hxx"
#include "rendering/script_storage.hxx"
#include "rendering/shadow_cascades.hxx"
#include "scene/environment.hxx"

struct BloomSettings {
    bool enabled = true;
    float threshold = 1.0F;
    float knee = 0.5F;

    float filter_radius = 1.0F;

    float intensity = 0.1F;
};

struct OutlineSettings {
    glm::vec3 colour{1.0F, 0.78F, 0.15F};

    float thickness_pixels = 3.0F;
};

struct AoSettings {
    bool enabled = true;

    float radius = 0.5F;

    float falloff_range = 0.615F;

    std::uint32_t slice_count = 2;
    std::uint32_t step_count = 6;

    float intensity = 1.0F;

    float denoise_depth_sigma = 40.0F;
};

struct FogSettings {
    bool enabled = false;
    glm::vec3 colour{0.5F};
    float extinction = 0.003F;
    float inscattering = 1.0F;
};

struct LightLodSettings {
    bool enabled = true;
    float cull_radius_pixels = 2.0F;
    float fade_radius_pixels = 6.0F;
};

struct FrameTimings {
    std::vector<frame_graph::PassTiming> passes;
    float full_frame_ms = 0.0F;

    std::vector<OverlayTiming> overlays;

    bool valid = false;

    std::uint64_t frame_serial = 0;
};

struct FrameStats {
    std::uint32_t submitted_triangle_count = 0;
    std::uint32_t submitted_instance_count = 0;

    std::uint32_t skin_job_count = 0;
    std::uint32_t skinned_vertex_count = 0;
    std::uint64_t skin_scratch_bytes_used = 0;
    std::uint32_t skin_fallback_instance_count = 0;

    std::uint32_t indirect_command_count = 0;
    std::uint32_t opaque_indirect_count = 0;
    std::uint32_t double_sided_indirect_count = 0;
    std::uint32_t mask_indirect_count = 0;
    std::uint32_t blend_indirect_count = 0;

    std::uint32_t visible_instance_count = 0;

    std::uint32_t frustum_visible_instance_count = 0;
    std::uint32_t early_instance_count = 0;
    std::uint32_t occlusion_candidate_count = 0;
    std::uint32_t late_instance_count = 0;
    std::uint32_t occluded_instance_count = 0;
    bool occlusion_stats_valid = false;

    std::uint32_t deferred_meshlet_count = 0;
    std::uint32_t occluded_meshlet_count = 0;
    bool meshlet_occlusion_stats_valid = false;

    std::uint32_t model_submission_count = 0;
    std::uint32_t mesh_submission_count = 0;

    std::uint32_t point_light_count = 0;
    std::uint32_t spot_light_count = 0;
};

inline constexpr std::uint32_t pipeline_stat_count = 4;

enum class OcclusionTestMode : std::uint8_t {
    hiz,

    never_occluded,

    always_defer,
};

struct PipelineStats {
    std::uint64_t clipped_primitive_count = 0;
    std::uint64_t fragment_shader_invocation_count = 0;
    std::uint64_t task_shader_invocation_count = 0;
    std::uint64_t mesh_shader_invocation_count = 0;
    bool mesh_stats_valid = false;

    bool valid = false;
};

struct SwapchainImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
};

enum class CompositeTarget : std::uint8_t {
    swapchain,

    viewport_panel,
};

struct FrameRecordInfo {
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    SwapchainImage swapchain_image{};
    std::uint32_t frame_index = 0;
    CompositeTarget composite_target = CompositeTarget::viewport_panel;
};

struct UBO {
    glm::mat4 view_projection;
    glm::mat4 view;
    glm::mat4 projection;

    glm::mat4 inverse_projection;

    glm::vec3 camera_position;
    glm::vec3 fog_colour;
    float fog_extinction = 0.0F;
    float fog_inscattering = 1.0F;

    std::array<glm::mat4, shadow_cascade_count> cascade_view_projection{};
    glm::vec4 cascade_split_far{};
    glm::vec4 cascade_texel_world{};
    glm::vec4 cascade_depth_scale{};

    glm::vec4 cascade_atlas_offset_u{};
    glm::vec4 cascade_atlas_scale_u{};
    glm::vec4 cascade_atlas_scale_v{};

    glm::vec3 light_direction{0.4F, 0.8F, 0.25F};
    float light_intensity = 3.0F;
    glm::vec3 light_colour{1.0F, 0.97F, 0.92F};
    float shadow_normal_offset_texels = 2.0F;

    std::uint32_t shadow_atlas_texture = 0;
    std::uint32_t shadow_sampler = 0;
    float shadow_depth_bias_world = 0.02F;
    float shadow_pcf_radius_texels = 1.0F;

    std::uint32_t cascade_count = shadow_cascade_count;
    float shadow_atlas_texel_u = 1.0F / static_cast<float>(shadow_atlas_width);
    float shadow_atlas_texel_v = 1.0F / static_cast<float>(shadow_atlas_height);
    std::uint32_t shadow_debug_cascade_tint = 0;

    float time = 0.0F;

    float ambient_intensity = 0.15F;

    float ao_intensity = 1.0F;

    float cluster_z_scale = 1.0F;
    float cluster_z_bias = 0.0F;

    std::uint32_t clustered_lighting = 1;
    std::uint32_t cluster_debug_heatmap = 0;

    std::uint32_t cluster_grid_x = 16;
    std::uint32_t cluster_grid_y = 9;
    std::uint32_t cluster_grid_z = 24;
    std::uint32_t cluster_light_capacity = 256;

    float light_lod_pixel_scale = 0.0F;
    float light_lod_cull_radius_pixels = 0.0F;
    float light_lod_fade_radius_pixels = 0.0F;

    EnvironmentUboBlock environment{};
};

static_assert(sizeof(UBO) == 952, "UBO layout changed -- update the mirror in assets/shaders/scene_types.slang");
static_assert(offsetof(UBO, environment) == 760);
static_assert(std::is_trivially_copyable_v<UBO>);
static_assert(offsetof(UBO, cascade_view_projection) == 288);
static_assert(offsetof(UBO, cascade_atlas_offset_u) == 592);
static_assert(offsetof(UBO, light_direction) == 640);

static_assert(std::is_copy_constructible_v<RendererError>,
              "RendererError must stay copyable -- std::expected<T, RendererError> copies it throughout this codebase");

struct RendererCreateInfo {
    VkExtent2D extent{};

    VkDeviceSize geometry_capacity = 256UZ * 1024UZ * 1024UZ;

    std::uint32_t material_capacity = 4096;
    std::uint32_t mesh_capacity = 4096;
    std::uint32_t model_capacity = 1024;
    std::uint32_t image_capacity = 4096;
    std::uint32_t sampler_capacity = 128;
    std::uint32_t pipeline_capacity = 128;
    std::uint32_t script_capacity = 4096;

    VkFormat hdr_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_SRGB;

    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    std::uint32_t maximum_draw_count = 1'000'000;
    std::uint32_t maximum_submission_count = 1'000'000;

    std::uint32_t maximum_skin_palette_matrices = 1U << 16;
    std::uint32_t maximum_skin_jobs = 1U << 14;
    VkDeviceSize skin_scratch_bytes = VkDeviceSize{32} << 20;
};

struct Renderer final : public IMeshSink, public IModelSink {
    explicit Renderer(VulkanContext &context) noexcept;
    ~Renderer() noexcept;

    Renderer(Renderer const &) = delete;
    auto operator=(Renderer const &) -> Renderer & = delete;

    Renderer(Renderer &&) = delete;
    auto operator=(Renderer &&) -> Renderer & = delete;

    [[nodiscard]]
    auto initialize(RendererCreateInfo const &create_info) -> std::expected<void, RendererError>;

    auto destroy() noexcept -> void;

    [[nodiscard]]
    auto load_model(AssetPath const &path) -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, RendererError> override;

    [[nodiscard]]
    auto install_model(ModelHandle pending, Model const &model) -> std::expected<void, RendererError> override;

    auto retain_model(ModelHandle handle) -> void override;

    auto release_model(ModelHandle handle) -> void override;

    auto register_model_name(ModelHandle handle, std::string_view name) -> void override;

    auto register_model_source(ModelHandle handle, AssetPath const &source) -> void override;

    [[nodiscard]]
    auto cached_model(AssetPath const &source) const -> ModelHandle;

    [[nodiscard]]
    auto model_source(ModelHandle handle) const noexcept -> AssetPath const *;

    [[nodiscard]]
    auto destroy_model(ModelHandle handle) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto create_model_from_cpu_data(ModelCpuData const &cpu_data) -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto create_model(Model const &model, MaterialHandle fallback_material)
            -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto create_model(Model const &model) -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto submit_model(ModelHandle model, glm::mat4 const &transform, MaterialHandle material_override = {},
                      std::span<MaterialSlotOverride const> slot_overrides = {}, bool outlined = false)
            -> std::expected<void, RendererError>;
    [[nodiscard]]
    auto submit_model(ModelHandle model, glm::mat4 &&, MaterialHandle material_override = {},
                      std::span<MaterialSlotOverride const> slot_overrides = {}, bool outlined = false)
            -> std::expected<void, RendererError>;

    [[nodiscard]] auto outline_settings() noexcept -> OutlineSettings & { return outline_settings_; }
    [[nodiscard]] auto outline_settings() const noexcept -> OutlineSettings const & { return outline_settings_; }

    [[nodiscard]]
    auto submit_model_instances(ModelHandle model, std::span<glm::mat4 const> transforms,
                                MaterialHandle material_override = {},
                                std::uint64_t resident_revision = 0,
                                std::span<std::uint32_t const> palette_offsets = {})
            -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto set_skin_palette(std::span<glm::mat4 const> palette) -> std::expected<void, RendererError>;
    [[nodiscard]]
    auto append_skin_palette(std::span<glm::mat4 const> palette) -> std::expected<std::uint32_t, RendererError>;

    [[nodiscard]]
    [[nodiscard]] auto model_animation(ModelHandle model) const -> std::shared_ptr<ModelAnimationData const>;

    auto model_bounds(ModelHandle model) const -> std::optional<std::pair<glm::vec3, glm::vec3>>;

    [[nodiscard]]
    auto model_submesh_bounds(ModelHandle model) const -> std::optional<std::vector<std::pair<glm::vec3, glm::vec3>>>;

    [[nodiscard]]
    auto model_lights(ModelHandle model) const -> std::span<ModelCpuLight const>;

    [[nodiscard]]
    auto model_materials(ModelHandle model) const -> std::vector<MaterialHandle>;

    [[nodiscard]]
    auto create_material(MaterialCreateInfo const &create_info, std::string debug_name = {})
            -> std::expected<MaterialHandle, RendererError>;

    [[nodiscard]]
    auto duplicate_material(MaterialHandle source, std::string debug_name = {})
            -> std::expected<MaterialHandle, RendererError>;

    [[nodiscard]]
    auto update_material(MaterialHandle handle, MaterialCreateInfo const &create_info)
            -> std::expected<void, RendererError>;

    auto retain_material(MaterialHandle handle) -> void;

    auto release_material(MaterialHandle handle) -> void;

    [[nodiscard]]
    auto destroy_material(MaterialHandle handle) -> std::expected<void, RendererError>;

    auto register_material_name(MaterialHandle handle, std::string name) -> bool;

    [[nodiscard]]
    auto create_mesh(MeshCreateInfo const &create_info) -> std::expected<MeshHandle, RendererError> override;

    [[nodiscard]]
    auto destroy_mesh(MeshHandle handle) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto update_submesh_geometry(MeshHandle mesh, std::uint32_t submesh_index, MeshGeometry const &geometry)
            -> std::expected<void, RendererError> override;

    [[nodiscard]]
    auto submit_mesh(MeshHandle mesh, glm::mat4 const &transform, MaterialHandle material_override = {})
            -> std::expected<void, RendererError> override;

    struct CameraMatrices {
        glm::mat4 view;
        glm::mat4 projection;
        float near_clip = 0.1F;
        float far_clip = 10000.0F;
        float vertical_fov_radians = 1.0471976F;
        float aspect_ratio = 1.7777778F;
        float time = 0.0F;
    };

    struct DirectionalLight {
        glm::vec3 direction{0.4F, 0.8F, 0.25F};
        glm::vec3 colour{1.0F, 0.97F, 0.92F};
        float intensity = 3.0F;
    };

    struct PointLight {
        glm::vec3 position{0.0F};
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
    };

    struct SpotLight {
        glm::vec3 position{0.0F};
        glm::vec3 direction{0.0F, -1.0F, 0.0F};
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
        float inner_cone_degrees = 20.0F;
        float outer_cone_degrees = 30.0F;
    };

    [[nodiscard]]
    auto submit_point_light(PointLight const &light) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto submit_spot_light(SpotLight const &light) -> std::expected<void, RendererError>;

    auto set_ambient_intensity(float intensity) noexcept -> void { ambient_intensity_ = intensity; }
    [[nodiscard]] auto ambient_intensity() const noexcept -> float { return ambient_intensity_; }

    auto set_fog_settings(FogSettings const &settings) noexcept -> void { fog_settings_ = settings; }
    [[nodiscard]] auto fog_settings() const noexcept -> FogSettings const & { return fog_settings_; }

    auto set_ao_settings(AoSettings const &settings) noexcept -> void { ao_settings_ = settings; }
    [[nodiscard]] auto ao_settings() const noexcept -> AoSettings const & { return ao_settings_; }

    auto set_bloom_settings(BloomSettings const &settings) noexcept -> void { bloom_settings_ = settings; }
    [[nodiscard]] auto bloom_settings() const noexcept -> BloomSettings const & { return bloom_settings_; }

    auto set_light_lod_settings(LightLodSettings const &settings) noexcept -> void { light_lod_settings_ = settings; }
    [[nodiscard]] auto light_lod_settings() const noexcept -> LightLodSettings const & { return light_lod_settings_; }

    static_assert(shadow_cascade_count == 4, "Renderer shadow-cache defaults assume four cascades");

    struct ShadowSettings {
        ShadowCascadeSettings cascades{};
        float normal_offset_texels = 2.0F;
        float depth_bias_world = 0.02F;
        float pcf_radius_texels = 1.0F;
        float depth_bias_constant = -1.0F;
        float depth_bias_slope = -2.5F;

        std::array<std::uint32_t, shadow_cascade_count> cache_update_periods{1U, 2U, 4U, 8U};
        bool cache_enabled = true;
        bool debug_cascade_tint = false;
    };

    auto set_directional_light(DirectionalLight const &light) noexcept -> void { light_ = light; }

    auto set_environment(SceneEnvironment const &environment) -> void;
    [[nodiscard]] auto environment_system() noexcept -> EnvironmentSystem & { return environment_; }
    [[nodiscard]] auto environment_system() const noexcept -> EnvironmentSystem const & { return environment_; }
    [[nodiscard]] auto directional_light() const noexcept -> DirectionalLight const & { return light_; }

    auto set_shadow_settings(ShadowSettings const &settings) noexcept -> void { shadow_settings_ = settings; }
    [[nodiscard]] auto shadow_settings() const noexcept -> ShadowSettings const & { return shadow_settings_; }

    auto mark_shadow_casters_dirty() noexcept -> void;

    auto mark_dynamic_shadow_casters_dirty() noexcept -> void { dynamic_shadow_casters_dirty_ = true; }

    [[nodiscard]]
    auto prepare_frame(VkCommandBuffer command_buffer, CameraMatrices const &, std::uint32_t frame_index)
            -> std::expected<void, RendererError>;

    [[nodiscard]] auto record_frame(FrameRecordInfo const &info) -> std::expected<void, RendererError>;

    [[nodiscard]] auto submit_batches() const noexcept -> std::span<SubmitBatch const> { return submit_batches_; }

    [[nodiscard]] auto frame_graph_view() const noexcept -> frame_graph::FrameGraphView const & {
        return frame_graph_view_;
    }

    [[nodiscard]] auto frame_graph_timings() const noexcept -> std::span<frame_graph::PassTiming const> {
        return pass_profiler_.timings();
    }

    [[nodiscard]] auto register_overlay(OverlayDesc desc) -> std::expected<OverlayRegistration, RendererError>;

    [[nodiscard]]
    auto resize(VkExtent2D extent) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto default_material() const noexcept -> MaterialHandle {
        return default_material_handle_;
    }

    [[nodiscard]]
    auto geometry_arena() noexcept -> GeometryArena & override {
        return geometry_arena_;
    }

    [[nodiscard]] auto shader_change_queue() noexcept -> ShaderChangeQueue & { return shader_change_queue_; }

    [[nodiscard]] auto aspect(std::uint32_t index) const -> float {
        static_cast<void>(index);
        return static_cast<float>(extent_.width) / static_cast<float>(extent_.height);
    }

    [[nodiscard]] auto viewport_target(std::uint32_t index) const noexcept -> ImageHandle {
        return frames_[index].viewport_target.handle();
    }

    auto queue_render_thread_event(std::move_only_function<void()> &&) -> void;
    auto drain_event_queue() -> void;

    [[nodiscard]] auto context() noexcept -> VulkanContext & { return context_; }
    [[nodiscard]] auto depth_format() const noexcept { return depth_format_; }
    [[nodiscard]] auto hdr_format() const noexcept { return hdr_format_; }
    [[nodiscard]] auto samples() const noexcept { return samples_; }

    // Takes effect at the start of the next frame. Counts the device does not support are clamped down.
    auto set_samples(VkSampleCountFlagBits samples) noexcept -> void;
    [[nodiscard]] auto max_samples() const noexcept -> VkSampleCountFlagBits;

    [[nodiscard]] auto image_storage() noexcept -> ImageStorage & override { return image_storage_; }
    [[nodiscard]] auto material_storage() noexcept -> MaterialStorage & override { return material_storage_; }
    [[nodiscard]] auto sampler_storage() noexcept -> SamplerStorage & override { return sampler_storage_; }
    [[nodiscard]] auto script_storage() noexcept -> ScriptStorage & { return script_storage_; }
    [[nodiscard]] auto script_storage() const noexcept -> ScriptStorage const & { return script_storage_; }
    [[nodiscard]] auto texture_streamer() noexcept -> TextureStreamer & override { return texture_streamer_; }
    [[nodiscard]] auto model_streamer() noexcept -> ModelStreamer & { return model_streamer_; }
    [[nodiscard]] auto resource_table() noexcept -> GpuResourceTable & { return gpu_resource_table_; }

    [[nodiscard]] auto assets() noexcept -> AssetRegistry & { return assets_; }
    [[nodiscard]] auto assets() const noexcept -> AssetRegistry const & { return assets_; }

    [[nodiscard]]
    auto request_texture(AssetPath source_path, TextureRole role, ImageHandle fallback,
                         std::string debug_name) -> ImageHandle;

    [[nodiscard]] auto resolve_pipeline(PipelineNodeHandle handle) const noexcept -> ShaderObjectSet const * {
        return pipeline_graph_.resolve_shader_objects(handle);
    }

    [[nodiscard]] auto register_pipeline(PipelineRegisterInfo info)
            -> std::expected<PipelineNodeHandle, RendererError> {
        auto registered = pipeline_graph_.register_pipeline(std::move(info));

        if (!registered) {
            return std::unexpected(RendererError{
                    .type = RendererErrorType::pipeline_graph_error,
                    .cause = ErrorCause{Boxed<PipelineGraphError>{registered.error()}},
            });
        }

        return *registered;
    }

    [[nodiscard]] auto last_frame_timings() const noexcept -> FrameTimings const & { return last_frame_timings_; }

    [[nodiscard]] auto recorded_frame_count() const noexcept -> std::uint64_t { return recorded_frame_count_; }
    [[nodiscard]] auto last_frame_stats() const noexcept -> FrameStats const & { return last_frame_stats_; }

    [[nodiscard]] auto transient_bytes() const noexcept -> std::uint64_t { return transient_allocator_.total_bytes(); }
    [[nodiscard]] auto transient_unaliased_bytes() const noexcept -> std::uint64_t {
        return transient_allocator_.unaliased_bytes();
    }
    enum AsyncCandidate : std::uint8_t {
        async_light_clustering = 1U << 0U,
        async_occlusion = 1U << 1U,
        async_gtao = 1U << 2U,
    };
    [[nodiscard]] auto async_candidates() const noexcept -> std::uint32_t { return async_candidates_; }
    auto set_async_candidates(std::uint32_t mask) noexcept -> void { async_candidates_ = mask; }

    [[nodiscard]] auto transient_aliasing() const noexcept -> bool { return transient_aliasing_; }
    auto set_transient_aliasing(bool enabled) noexcept -> void { transient_aliasing_ = enabled; }

    auto set_frame_graph_dump(bool enabled) noexcept -> void { dump_frame_graph_ = enabled; }
    auto set_frame_graph_dot(std::string path) -> void { frame_graph_dot_path_ = std::move(path); }
    [[nodiscard]] auto last_frame_pipeline_stats() const noexcept -> PipelineStats const & {
        return last_frame_pipeline_stats_;
    }
    [[nodiscard]] auto debug_draw_light_icons() const noexcept -> bool { return debug_draw_light_icons_; }
    auto set_debug_draw_light_icons(bool enabled) noexcept -> void { debug_draw_light_icons_ = enabled; }

    [[nodiscard]] auto meshlet_culling() const noexcept -> bool { return meshlet_culling_; }
    auto set_meshlet_culling(bool enabled) noexcept -> void { meshlet_culling_ = enabled; }

    [[nodiscard]] auto occlusion_culling() const noexcept -> bool { return occlusion_culling_; }
    auto set_occlusion_culling(bool enabled) noexcept -> void;

    [[nodiscard]] auto occlusion_culling_supported() const noexcept -> bool;

    [[nodiscard]] auto meshlet_occlusion_culling() const noexcept -> bool { return meshlet_occlusion_culling_; }
    auto set_meshlet_occlusion_culling(bool enabled) noexcept -> void { meshlet_occlusion_culling_ = enabled; }

    [[nodiscard]] auto occlusion_test_mode() const noexcept -> OcclusionTestMode { return occlusion_test_mode_; }
    auto set_occlusion_test_mode(OcclusionTestMode mode) noexcept -> void;

    [[nodiscard]] auto hiz_debug_view(std::uint32_t mip) const noexcept -> ImageHandle;
    [[nodiscard]] auto hiz_debug_mip_count() const noexcept -> std::uint32_t { return hiz_.mip_count; }
    [[nodiscard]] auto hiz_depth_extent() const noexcept -> VkExtent2D { return hiz_.depth_extent; }

    [[nodiscard]] auto clustered_lighting() const noexcept -> bool { return clustered_lighting_; }
    auto set_clustered_lighting(bool enabled) noexcept -> void { clustered_lighting_ = enabled; }

    [[nodiscard]] auto cluster_debug_heatmap() const noexcept -> bool { return cluster_debug_heatmap_; }
    auto set_cluster_debug_heatmap(bool enabled) noexcept -> void { cluster_debug_heatmap_ = enabled; }

    [[nodiscard]] auto cluster_grid() const noexcept -> ClusterGridSettings const & { return cluster_grid_; }
    auto set_cluster_grid(ClusterGridSettings const &grid) -> std::expected<void, std::string>;

    [[nodiscard]] auto last_cluster_stats() const noexcept -> ClusterStats const & { return last_cluster_stats_; }

    auto request_screenshot(ScreenshotSource source) noexcept -> void;
    auto mark_lights_dirty() -> void { lights_dirty_mask_ = frames_.empty() ? 0U : ((1U << frames_.size()) - 1U); }
    auto wait_idle() -> std::expected<void, RendererError>;

    static auto compiler() noexcept -> renderer::SlangCompiler &;

    // Starts compiling the engine shaders in the background; call as early as possible, after any shader pack is installed.
    static auto prefetch_shaders() -> void;

private:
    struct Submission {
        MeshHandle mesh{};
        glm::mat4 transform{1.0F};
        MaterialHandle material_override{};
    };

    struct alignas(16) GpuDraw {
        VkDeviceAddress vertex_address = 0;
        VkDeviceAddress meshlet_address = 0;
        VkDeviceAddress meshlet_data_address = 0;

        std::uint32_t material_index = 0;

        std::uint32_t meshlet_visibility_offset = 0;
    };

    static_assert(std::is_trivially_copyable_v<GpuDraw>);

    static_assert(sizeof(GpuDraw) == 32);

    struct alignas(16) GpuCullBounds {
        glm::vec3 bounds_min{-0.5F};
        float wind_padding = 0.0F;
        glm::vec3 bounds_max{0.5F};
        std::uint32_t first_chunk = 0;
    };

    static constexpr std::uint32_t cull_chunk_size = 256;

    static constexpr VkDeviceSize cull_chunk_bytes = (8 + 3 * cull_chunk_size / 32) * sizeof(std::uint32_t);

    struct alignas(16) GpuLodGroup {
        std::uint32_t batch = 0;
        std::uint32_t first_instance = 0;
        std::uint32_t meshlet_count = 0;
        std::uint32_t first_meshlet_bit = 0;
        GpuDraw draw{};
    };

    static_assert(sizeof(GpuLodGroup) == 48);

    struct alignas(16) GpuLodJob {
        VkDeviceAddress transforms_address = 0;
        std::uint32_t instance_count = 0;
        std::uint32_t first_chunk = 0;
        std::uint32_t group_count = 0;
        std::uint32_t lod_groups = 0;
        std::uint32_t pad0 = 0;
        std::uint32_t pad1 = 0;
        std::array<GpuLodGroup, lod_count> groups{};
    };

    static_assert(std::is_trivially_copyable_v<GpuLodJob>);
    static_assert(sizeof(GpuLodJob) == 32 + 48 * lod_count);

    static_assert(std::is_trivially_copyable_v<GpuCullBounds>);

    static_assert(sizeof(GpuCullBounds) == 32);

    struct alignas(16) GpuOcclusionView {
        glm::mat4 view_projection{1.0F};

        VkDeviceAddress meshlet_visibility_address = 0;

        VkDeviceAddress stats_address = 0;

        std::uint32_t hiz_texture_index = 0;
        std::uint32_t hiz_mip_count = 0;
        std::uint32_t depth_width = 0;
        std::uint32_t depth_height = 0;

        std::uint32_t enabled = 0;
        float depth_epsilon = 1e-6F;
        float guard_pixels = 1.0F;
        std::uint32_t _pad0 = 0;
    };

    static_assert(std::is_trivially_copyable_v<GpuOcclusionView>);
    static_assert(sizeof(GpuOcclusionView) == 112);
    static_assert(offsetof(GpuOcclusionView, meshlet_visibility_address) == 64);
    static_assert(offsetof(GpuOcclusionView, hiz_texture_index) == 80);
    static_assert(offsetof(GpuOcclusionView, enabled) == 96);

    static constexpr std::uint32_t occlusion_stat_frustum_visible = 0;
    static constexpr std::uint32_t occlusion_stat_early = 1;
    static constexpr std::uint32_t occlusion_stat_candidates = 2;
    static constexpr std::uint32_t occlusion_stat_late = 3;
    static constexpr std::uint32_t occlusion_stat_deferred_meshlets = 4;
    static constexpr std::uint32_t occlusion_stat_occluded_meshlets = 5;
    static constexpr std::uint32_t occlusion_stat_count = 8;

    static constexpr std::uint32_t maximum_cull_batch_count = 65'535;

    enum class GpuLightType : std::uint32_t {
        point = 0,
        spot = 1,
    };

    struct alignas(16) GpuLight {
        glm::vec3 position{0.0F};
        float range = 10.0F;

        glm::vec3 colour{1.0F};
        float intensity = 1.0F;

        glm::vec3 direction{0.0F, -1.0F, 0.0F};
        float spot_scale = 1.0F;

        float spot_offset = 0.0F;
        GpuLightType type = GpuLightType::point;
        float _pad0 = 0.0F;
        float _pad1 = 0.0F;
    };

    static_assert(std::is_trivially_copyable_v<GpuLight>);
    static_assert(sizeof(GpuLight) == 64);

    static constexpr std::uint32_t maximum_light_count = 65'536;

    static constexpr std::uint32_t cull_plane_count = 6 * (1 + shadow_cascade_count);

    using ShadowCascadeMask = std::uint32_t;
    static constexpr ShadowCascadeMask all_shadow_cascades_mask =
            (ShadowCascadeMask{1} << shadow_cascade_count) - ShadowCascadeMask{1};

    struct ShadowCascadeCacheEntry {
        glm::mat4 view_projection{1.0F};
        float split_far = 0.0F;
        float texel_world = 0.0F;
        float depth_scale = 0.0F;
        std::uint64_t last_update_frame = 0;
        bool valid = false;
    };

    struct RendererFrame {
        Buffer upload_buffer{};
        Buffer draw_buffer{};
        Buffer transform_buffer{};
        Buffer indirect_buffer{};

        Buffer batch_bounds_buffer{};
        Buffer culled_indirect_buffer{};
        Buffer visible_draw_buffer{};
        Buffer visible_transform_buffer{};

        Buffer occlusion_views_buffer{};
        Buffer occlusion_candidates_buffer{};
        Buffer cull_chunks_buffer{};
        Buffer late_indirect_buffer{};
        Buffer merged_indirect_buffer{};

        Buffer occlusion_stats_buffer{};
        Buffer occlusion_stats_readback_buffer{};
        bool occlusion_stats_pending = false;
        bool occlusion_stats_active = false;
        bool meshlet_occlusion_stats_active = false;

        bool occlusion_active = false;

        Buffer meshlet_visibility_buffer{};
        std::uint32_t meshlet_visibility_capacity_words = 0;
        std::uint32_t meshlet_visibility_words = 0;
        bool meshlet_occlusion_active = false;

        Buffer frustum_planes_buffer{};

        Buffer lights_buffer{};

        Buffer visible_lights_buffer{};

        Buffer cluster_lights_buffer{};
        ClusterGridSettings cluster_grid{};

        Buffer cluster_stats_readback_buffer{};
        ClusterGridSettings cluster_stats_grid{};
        bool cluster_stats_pending = false;

        glm::mat4 view_projection{1.0F};

        ImageHolder viewport_target{};

        ShadowCascadeMask shadow_update_mask = all_shadow_cascades_mask;
        std::array<ShadowCascadeCacheEntry, shadow_cascade_count> pending_shadow_cache{};
        glm::vec3 pending_shadow_light_direction{0.0F};
        float pending_shadow_depth_bias_constant = 0.0F;
        float pending_shadow_depth_bias_slope = 0.0F;
        std::uint64_t pending_shadow_caster_revision = 0;
        std::uint64_t pending_shadow_scene_signature = 0;

        VkDeviceSize draw_upload_offset = 0;
        VkDeviceSize transform_upload_offset = 0;
        VkDeviceSize indirect_upload_offset = 0;
        VkDeviceSize batch_bounds_upload_offset = 0;

        std::uint32_t draw_count = 0;
        std::uint32_t transform_count = 0;

        std::vector<GpuDrawCommand> indirect_commands;

        std::vector<GpuCullBounds> batch_bounds;

        std::uint32_t light_count = 0;

        std::uint32_t indirect_command_count = 0;

        std::uint32_t cull_chunk_count = 0;

        std::uint32_t lod_chunk_count = 0;
        glm::vec3 lod_camera_position{0.0F};
        std::vector<GpuLodJob> lod_jobs;
        Buffer lod_jobs_buffer{};

        Buffer skin_upload_buffer{};
        Buffer skin_input_buffer{};
        Buffer skin_scratch_buffer{};
        std::vector<GpuSkinJob> skin_jobs;
        VkDeviceSize skin_scratch_used = 0;
        std::uint32_t skin_palette_count = 0;
        std::uint32_t skin_chunk_total = 0;
        std::uint32_t skin_fallback_instances = 0;
        std::uint32_t skinned_vertices = 0;

        std::vector<Buffer> retired_buffers;

        std::vector<std::pair<std::uint32_t, std::uint32_t>> cpu_instance_ranges;

        std::uint32_t opaque_indirect_count = 0;
        std::uint32_t double_sided_indirect_count = 0;
        std::uint32_t mask_indirect_count = 0;
        std::uint32_t blend_indirect_count = 0;

        std::array<std::uint32_t, shadow_cascade_count> shadow_opaque_indirect_count{};
        std::array<std::uint32_t, shadow_cascade_count> shadow_double_sided_indirect_count{};
        std::array<std::uint32_t, shadow_cascade_count> shadow_mask_indirect_count{};
    };

    struct HizPyramid {
        ImageHolder image;
        std::array<ImageHolder, hiz_max_mip_count> mip_slots;
        VkExtent2D depth_extent{};
        std::uint32_t mip_count = 0;

        bool layout_initialised = false;

        HizPyramid() = default;
        ~HizPyramid() = default;

        HizPyramid(HizPyramid const &) = delete;
        auto operator=(HizPyramid const &) -> HizPyramid & = delete;

        HizPyramid(HizPyramid &&) noexcept = default;

        auto operator=(HizPyramid &&other) noexcept -> HizPyramid & {
            mip_slots = std::move(other.mip_slots);
            image = std::move(other.image);
            depth_extent = other.depth_extent;
            mip_count = other.mip_count;
            layout_initialised = other.layout_initialised;

            return *this;
        }
    };

    [[nodiscard]]
    auto create_hiz_pyramid(VkExtent2D depth_extent) -> std::expected<HizPyramid, RendererError>;

    struct OwnedFrameTargets {
        ImageHolder viewport_target{};
    };

    struct ModelSubmission {
        ModelHandle model{};
        glm::mat4 transform{1.0F};
        MaterialHandle material_override{};

        std::uint32_t slot_override_first = 0;
        std::uint32_t slot_override_count = 0;

        bool outlined = false;
    };

    struct InstancedSubmission {
        ModelHandle model{};
        MaterialHandle material_override{};
        std::uint32_t first_transform = 0;
        std::uint32_t transform_count = 0;
        std::size_t model_submission_position = 0;

        std::uint32_t first_palette_offset = 0;
        std::uint32_t palette_count = 0;

        std::uint64_t resident_revision = 0;
    };

    struct ResidentInstanceSet {
        Buffer transforms{};
        std::uint32_t count = 0;
        std::uint64_t last_used_frame = 0;
    };

    std::unordered_map<std::uint64_t, ResidentInstanceSet> resident_instance_sets_;

    struct PendingResidentUpload {
        Buffer staging{};
        VkBuffer destination = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
    };

    std::vector<PendingResidentUpload> pending_resident_uploads_;

    std::vector<Buffer> retired_resident_buffers_;

    std::uint32_t resident_jobs_this_frame_ = 0;

    struct ResidentLodGroups {
        std::uint32_t count = 0;
        std::uint32_t lod_groups = 0;
        std::array<std::uint32_t, lod_count> representative_lod{};
        std::array<MaterialHandle, lod_count> material{};
    };

    [[nodiscard]]
    auto resident_lod_groups(Submesh const &submesh, MaterialHandle base_material) const noexcept -> ResidentLodGroups;

    [[nodiscard]]
    auto submit_resident_instances(ModelHandle model, std::span<glm::mat4 const> transforms,
                                   MaterialHandle material_override, std::uint64_t revision) -> bool;

    std::uint64_t resident_slots_this_frame_ = 0;

    static constexpr std::uint64_t resident_set_idle_frames = frames_in_flight + 2;

    static constexpr std::uint32_t maximum_lod_job_count = 4096;

    struct BatchEntry {
        MeshHandle mesh{};
        std::uint32_t submesh_index = 0;
        MaterialHandle material{};
        std::uint32_t lod_index = 0;

        static constexpr std::uint32_t no_lod_job = ~0U;
        std::uint32_t lod_job = no_lod_job;
        std::uint32_t lod_group = 0;
        std::uint32_t resident_capacity = 0;

        std::vector<glm::mat4 const *> transforms;

        static constexpr std::uint32_t no_palette = ~0U;
        std::vector<std::uint32_t> palette_offsets;
        bool skinned = false;

        std::uint64_t frame_stamp = 0;
    };

    struct PendingBlendBatch {
        BatchEntry const *entry = nullptr;
        float camera_distance_sq = 0.0F;
    };

    struct BatchKey {
        std::uint32_t mesh_index;
        std::uint32_t submesh_index;
        std::uint32_t material_index;
        std::uint32_t lod_index;

        std::uint32_t lod_job_key = 0;

        auto operator==(BatchKey const &) const noexcept -> bool = default;
    };

    struct BatchKeyHash {
        auto operator()(BatchKey const &key) const noexcept -> std::size_t {
            auto const mesh_hash =
                    std::hash<std::uint64_t>{}((static_cast<std::uint64_t>(key.mesh_index) << 32) | key.submesh_index);

            return mesh_hash ^ (std::hash<std::uint32_t>{}(key.material_index) << 1) ^
                   (std::hash<std::uint32_t>{}(key.lod_index) << 2) ^
                   (std::hash<std::uint32_t>{}(key.lod_job_key) << 3);
        }
    };

    std::unordered_map<BatchKey, BatchEntry, BatchKeyHash> batches_;
    std::vector<BatchEntry *> active_batches_;

    std::vector<BatchEntry const *> opaque_batches_;
    std::vector<BatchEntry const *> double_sided_batches_;
    std::vector<BatchEntry const *> mask_batches_;
    std::vector<PendingBlendBatch> blend_batches_;

    std::uint64_t batch_frame_ = 0;

    [[nodiscard]]
    auto mesh_slot(MeshHandle handle) noexcept -> MeshSlotData *;

    [[nodiscard]]
    auto mesh_slot(MeshHandle handle) const noexcept -> MeshSlotData const *;

    [[nodiscard]]
    auto model_slot(ModelHandle handle) noexcept -> ModelSlotData *;

    [[nodiscard]]
    auto model_slot(ModelHandle handle) const noexcept -> ModelSlotData const *;

    [[nodiscard]]
    auto
    create_model_common(Model const &model, MaterialHandle fallback_material,
                        std::move_only_function<std::expected<ModelHandle, ModelStorageError>(ModelSlotData)> install)
            -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto upload_frame_data(VkCommandBuffer command_buffer, RendererFrame &frame) -> std::expected<void, RendererError>;

    auto clear_submissions() noexcept -> void;

    struct RecordedOverlay {
        std::string name;
        OverlayStage stage = OverlayStage::scene;
        std::uint32_t slot = 0;
    };

    struct FrameTimestamps {
        VkQueryPool query_pool{VK_NULL_HANDLE};
        bool has_results{false};
        std::uint64_t serial{0};
        std::vector<RecordedOverlay> overlays;
    };

    struct FrameTargets {
        Image const *shadow_atlas = nullptr;
        Image const *viewport = nullptr;

        VkExtent2D extent{};
        bool multisampled = false;
    };

    auto consume_culled_readback(RendererFrame &frame) -> void;

    [[nodiscard]]
    auto prepare_cluster_buffers(RendererFrame &frame) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto resolve_frame_targets(RendererFrame const &frame) const -> std::expected<FrameTargets, RendererError>;

    [[nodiscard]]
    auto early_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers;

    [[nodiscard]]
    auto late_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers;

    [[nodiscard]]
    auto forward_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers;

    [[nodiscard]]
    static auto batch_counts(RendererFrame const &frame) noexcept -> render_pass::DrawCounts;

    auto record_resident_instance_uploads(VkCommandBuffer command_buffer, RendererFrame &frame) -> void;

    static constexpr std::uint32_t skin_palette_padding = 256;
    [[nodiscard]] auto skin_jobs_offset() const noexcept -> VkDeviceSize {
        return VkDeviceSize{maximum_skin_palette_matrices_ + skin_palette_padding} * sizeof(glm::mat4);
    }
    [[nodiscard]] auto skin_chunks_offset() const noexcept -> VkDeviceSize {
        return skin_jobs_offset() + VkDeviceSize{maximum_skin_jobs_} * sizeof(GpuSkinJob);
    }
    [[nodiscard]] auto skin_input_size() const noexcept -> VkDeviceSize {
        return skin_chunks_offset() + VkDeviceSize{maximum_skin_jobs_ + 1} * sizeof(std::uint32_t);
    }

    [[nodiscard]]
    auto prepare_skin_upload(RendererFrame &frame) -> std::expected<void, RendererError>;

    auto record_skin_upload(VkCommandBuffer command_buffer, RendererFrame const &frame) -> void;

    [[nodiscard]]
    auto record_skin(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto record_instance_lods(render_pass::Context const &pass_context,
                              RendererFrame const &frame) -> std::expected<void, RendererError>;

    VkDeviceSize cull_chunk_capacity_ = 0;

    static constexpr VkDeviceSize lod_chunk_bytes = 2 * lod_count * sizeof(std::uint32_t);

    auto record_environment_pass(render_pass::Context const &pass_context, RendererFrame const &frame) -> void;

    [[nodiscard]]
    auto record_shadow_pass(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto record_depth_prepass(render_pass::Context const &pass_context, RendererFrame const &frame,
                              FrameTargets const &targets, render_pass::DepthPrepassPhase phase)
            -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto record_hiz_build(render_pass::Context const &pass_context, FrameTargets const &targets,
                          std::uint32_t depth_texture_index) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto record_occlusion_cull_pass(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;

    auto record_occlusion_stats_clear(VkCommandBuffer command_buffer, RendererFrame const &frame) -> void;
    auto record_meshlet_visibility_clear(VkCommandBuffer command_buffer, RendererFrame const &frame) -> void;
    [[nodiscard]]
    auto record_gpu_culling(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;

    auto record_cluster_stats_clear(render_pass::Context const &pass_context, RendererFrame const &frame) -> void;
    [[nodiscard]]
    auto record_light_cull(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;
    [[nodiscard]]
    auto record_light_cluster(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;
    auto record_cluster_stats_readback(render_pass::Context const &pass_context, RendererFrame &frame) -> void;

    auto record_occlusion_stats_readback(VkCommandBuffer command_buffer, RendererFrame &frame) -> void;

    [[nodiscard]]
    auto ambient_occlusion_info(FrameTargets const &targets, std::uint32_t frame_index,
                                std::uint32_t depth_texture_index, std::uint32_t raw_texture_index,
                                std::uint32_t denoised_texture_index) const -> render_pass::AmbientOcclusionInfo;

    [[nodiscard]]
    auto record_forward_pass(render_pass::Context const &pass_context, RendererFrame const &frame,
                             FrameTargets const &targets, std::uint32_t ao_texture_index,
                             std::uint32_t hdr_texture_index, render_pass::Callback scene_overlays)
            -> std::expected<render_pass::HdrTextureIndex, RendererError>;

    [[nodiscard]]
    auto record_bloom_pass(render_pass::Context const &pass_context, FrameTargets const &targets,
                           render_pass::HdrTextureIndex hdr, Image const &bloom_image,
                           std::array<std::uint32_t, render_pass::bloom_mip_count> const &mip_texture_indices)
            -> std::expected<std::optional<render_pass::BloomTextureIndex>, RendererError>;

    [[nodiscard]]
    auto make_pass_context(VkCommandBuffer command_buffer, std::uint32_t frame_index, bool compute_only = false)
            -> render_pass::Context;

    auto record_overlay_prepares(render_pass::Context const &pass_context) -> void;

    auto record_overlay_stage(render_pass::Context const &pass_context, OverlayStage stage, OverlayScope const &scope,
                              glm::mat4 const &view_projection) -> void;

    [[nodiscard]]
    auto register_light_icon_overlay() -> std::expected<void, RendererError>;

    auto read_overlay_timings(FrameTimestamps const &frame_query) -> void;

    [[nodiscard]]
    auto create_frame_targets(std::uint32_t frame_index, VkExtent2D extent)
            -> std::expected<OwnedFrameTargets, RendererError>;

    struct PassHandoff {
        std::uint32_t ao_texture_index = 0;
        render_pass::HdrTextureIndex hdr{};
        std::optional<render_pass::BloomTextureIndex> bloom;
    };

    auto record_frame_end(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void;

    VulkanContext &context_;

    VkFormat hdr_format_ = VK_FORMAT_UNDEFINED;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkSampleCountFlagBits pending_samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkExtent2D extent_{};

    GeometryArena geometry_arena_{};
    MaterialStorage material_storage_{};
    ImageStorage image_storage_{};
    SamplerStorage sampler_storage_{};
    TextureStreamer texture_streamer_{};
    ModelStreamer model_streamer_{};
    AssetRegistry assets_{};
    GpuResourceTable gpu_resource_table_{};

    std::vector<Buffer> ubos_;

    PipelineGraphRepository pipeline_graph_;
    PipelineNodeHandle shadow_pipeline_;
    PipelineNodeHandle shadow_mask_pipeline_;
    PipelineNodeHandle depth_prepass_pipeline_;
    PipelineNodeHandle depth_prepass_mask_pipeline_;
    PipelineNodeHandle forward_pipeline_;

    PipelineNodeHandle forward_outline_pipeline_;
    PipelineNodeHandle forward_outline_instanced_pipeline_;
    PipelineNodeHandle forward_blend_pipeline_;

    PipelineNodeHandle shadow_instanced_pipeline_;
    PipelineNodeHandle shadow_mask_instanced_pipeline_;
    PipelineNodeHandle depth_prepass_instanced_pipeline_;
    PipelineNodeHandle depth_prepass_mask_instanced_pipeline_;
    PipelineNodeHandle forward_instanced_pipeline_;
    PipelineNodeHandle forward_blend_instanced_pipeline_;
    PipelineNodeHandle composite_pipeline_;
    PipelineNodeHandle frustum_cull_pipeline_;
    PipelineNodeHandle instance_lod_pipeline_;
    PipelineNodeHandle occlusion_cull_pipeline_;
    PipelineNodeHandle hiz_build_pipeline_;
    PipelineNodeHandle light_icon_pipeline_;
    PipelineNodeHandle bloom_downsample_pipeline_;
    PipelineNodeHandle bloom_upsample_pipeline_;
    PipelineNodeHandle gtao_pipeline_;
    PipelineNodeHandle gtao_denoise_pipeline_;
    PipelineNodeHandle light_cull_pipeline_;
    PipelineNodeHandle light_cluster_pipeline_;
    PipelineNodeHandle skybox_pipeline_;
    ShaderChangeQueue shader_change_queue_;

    EnvironmentSystem environment_;
    std::uint64_t frame_counter_ = 0;

    BloomSettings bloom_settings_;
    OutlineSettings outline_settings_;

    struct OutlineVariant {
        MaterialHandle source{};
        MaterialHandle variant{};
        std::uint64_t refreshed_frame = 0;
    };
    std::vector<OutlineVariant> outline_variants_;

    [[nodiscard]] auto outline_variant(MaterialHandle source) -> MaterialHandle;
    auto prune_outline_variants() -> void;
    AoSettings ao_settings_;

    ImageHandle light_icon_texture_{};
    bool debug_draw_light_icons_ = false;

    OverlayRegistry overlays_;
    OverlayRegistration light_icon_overlay_;
    bool meshlet_culling_ = true;
    bool occlusion_culling_ = true;
    bool meshlet_occlusion_culling_ = true;
    bool meshlet_visibility_cap_warned_ = false;
    OcclusionTestMode occlusion_test_mode_ = OcclusionTestMode::hiz;
    bool clustered_lighting_ = true;
    bool cluster_debug_heatmap_ = false;
    ClusterGridSettings cluster_grid_{};
    ClusterStats last_cluster_stats_{};
    float light_icon_world_size_ = 0.5F;

    ImageHolder shadow_atlas_{};

    HizPyramid hiz_{};
    glm::mat4 hiz_history_view_projection_{1.0F};
    bool hiz_history_valid_ = false;

    bool hiz_debug_view_supported_ = false;
    std::array<ShadowCascadeCacheEntry, shadow_cascade_count> shadow_cascade_cache_{};
    std::uint64_t shadow_frame_ = 0;
    std::uint64_t shadow_caster_revision_ = 1;
    std::uint64_t cached_shadow_caster_revision_ = 0;
    std::uint64_t shadow_scene_signature_ = 0;
    glm::vec3 cached_shadow_light_direction_{0.0F};
    float cached_shadow_depth_bias_constant_ = 0.0F;
    float cached_shadow_depth_bias_slope_ = 0.0F;
    bool shadow_scene_signature_valid_ = false;
    bool shadow_global_state_valid_ = false;
    bool shadow_atlas_initialized_ = false;
    bool dynamic_shadow_casters_dirty_ = false;

    DirectionalLight light_{};
    ShadowSettings shadow_settings_{};
    float ambient_intensity_ = 0.15F;
    FogSettings fog_settings_{};
    LightLodSettings light_lod_settings_{};

    std::vector<PointLight> point_light_submissions_;
    std::vector<SpotLight> spot_light_submissions_;

    MeshStorage mesh_storage_;
    ModelStorage model_storage_;
    ScriptStorage script_storage_;

    std::unordered_map<std::size_t, ModelHandle> model_cache_;

    std::unordered_map<std::uint64_t, AssetPath> model_sources_;

    std::vector<Submission> submissions_;
    std::vector<ModelSubmission> model_submissions_;
    std::vector<MaterialSlotOverride> slot_override_submissions_;
    std::vector<InstancedSubmission> instanced_submissions_;
    std::vector<glm::mat4> instance_transforms_;
    std::vector<std::uint32_t> instance_palette_offsets_;

    std::vector<glm::mat4> skin_palette_;
    std::uint32_t maximum_skin_palette_matrices_ = 0;
    std::uint32_t maximum_skin_jobs_ = 0;
    VkDeviceSize skin_scratch_capacity_ = 0;
    PipelineNodeHandle skin_pipeline_;

    std::deque<glm::mat4> computed_transforms_;

    [[nodiscard]] auto submitted_model_count() const noexcept -> std::size_t {
        return model_submissions_.size() + instance_transforms_.size();
    }

    std::vector<RendererFrame> frames_;

    MaterialHandle default_material_handle_{};

    std::uint32_t maximum_draw_count_ = 0;
    std::uint32_t maximum_submission_count_ = 0;

    FrameTimings last_frame_timings_{};
    FrameStats last_frame_stats_{};
    std::uint64_t recorded_frame_count_ = 0;

    std::vector<GpuLight> light_staging_;
    std::uint32_t lights_dirty_mask_ = 0;
    std::uint32_t light_count_ = 0;

    std::queue<std::move_only_function<void()>> event_queue_;
    std::atomic_uint32_t queued_events_;
    std::mutex queue_mutex_;

    std::vector<FrameTimestamps> timestamp_queries_;
    float timestamp_period_{1.0F};

    frame_graph::FrameGraph frame_graph_;
    frame_graph::PlanCache plan_cache_;
    frame_graph::CompiledGraph const *frame_plan_ = nullptr;
    frame_graph::FrameGraphView frame_graph_view_;
    std::vector<SubmitBatch> submit_batches_;
    frame_graph::PassProfiler pass_profiler_;

    frame_graph::TransientAllocator transient_allocator_;
    bool transient_aliasing_ = true;
    std::uint32_t async_candidates_ = async_light_clustering | async_occlusion | async_gtao;
    std::uint64_t logged_plan_misses_ = 0;
    std::uint64_t logged_transient_bytes_ = 0;
    bool dump_frame_graph_ = false;
    std::string frame_graph_dot_path_;

    struct FramePipelineQuery {
        VkQueryPool query_pool{VK_NULL_HANDLE};
        bool has_results{false};
    };
    std::vector<FramePipelineQuery> pipeline_stat_queries_;
    PipelineStats last_frame_pipeline_stats_{};

    std::unique_ptr<ScreenshotCapture> screenshot_;

    bool initialized_ = false;
};
