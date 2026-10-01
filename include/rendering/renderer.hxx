#pragma once

#include <atomic>
#include <queue>
#include <volk.h>

#include <glm/glm.hpp>

#include <BS_thread_pool.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
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
#include "rendering/cluster_grid.hxx"
#include "rendering/forward_target.hxx"
#include "rendering/pipeline_graph_repository.hxx"
#include "rendering/render_passes.hxx"
#include "rendering/render_stage.hxx"
#include "rendering/script_storage.hxx"
#include "rendering/shadow_cascades.hxx"

struct BloomSettings {
    bool enabled = true;
    float threshold = 1.0F;
    float knee = 0.5F;

    // Upsample tent radius in texels of the lower mip; 1.0 is the standard 3x3 tent.
    float filter_radius = 1.0F;

    // Scale applied to the bloom before it is added to the HDR colour.
    float intensity = 0.1F;
};

// GTAO (Jimenez et al. 2016): screen-space horizon-based AO from depth alone, denoised with a depth-aware blur
// and multiplied into the ambient term alongside the material's baked occlusion.
struct AoSettings {
    bool enabled = true;

    // View-space sampling radius in world units.
    float radius = 0.5F;

    // Fraction of `radius` over which a sample's contribution fades out.
    float falloff_range = 0.615F;

    std::uint32_t slice_count = 2;
    std::uint32_t step_count = 6;

    // 0 disables screen-space AO, 1 applies it at full strength.
    float intensity = 1.0F;

    // Larger values blur across bigger depth differences: less noise, softer edges.
    float denoise_depth_sigma = 40.0F;
};

// Exponential distance fog, applied after shading.
struct FogSettings {
    bool enabled = false;
    glm::vec3 colour{0.5F};
    float extinction = 0.003F;
    float inscattering = 1.0F;
};

struct StageTimings {
    std::array<float, stage_count> milliseconds{};

    // Overlays that ran in the timed frame, in draw order. Their time is already included in `milliseconds`.
    std::vector<OverlayTiming> overlays;

    bool valid = false;
};

struct FrameStats {
    std::uint32_t submitted_triangle_count = 0;
    std::uint32_t submitted_instance_count = 0;

    std::uint32_t indirect_command_count = 0;
    std::uint32_t opaque_indirect_count = 0;
    std::uint32_t mask_indirect_count = 0;
    std::uint32_t blend_indirect_count = 0;

    // Instances that survived GPU culling. Lags a frames-in-flight cycle behind the rest; 0 until the first
    // readback.
    std::uint32_t visible_instance_count = 0;

    std::uint32_t model_submission_count = 0;
    std::uint32_t mesh_submission_count = 0;

    std::uint32_t point_light_count = 0;
    std::uint32_t spot_light_count = 0;
};

inline constexpr std::uint32_t pipeline_stat_count = 4;

// Forward pass only. Task/mesh counts are valid only when mesh_stats_valid is set; otherwise the clipped count is.
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

// Where the tonemapped scene goes this frame.
enum class CompositeTarget : std::uint8_t {
    // Fullscreen play: straight into the swapchain, UI overlays on top.
    swapchain,

    // Editor and embedded play: into the frame's viewport_target, which the Viewport panel samples.
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
    float fog_extinction = 0.0F; // 0 = disabled.
    float fog_inscattering = 1.0F;

    // PSSM cascades.
    std::array<glm::mat4, shadow_cascade_count> cascade_view_projection{};
    glm::vec4 cascade_split_far{}; // view-space far distance per cascade
    glm::vec4 cascade_texel_world{}; // world-space size of one shadow texel per cascade
    glm::vec4 cascade_depth_scale{}; // 1 / (z_far - z_near) per cascade

    // Atlas tiles have variable widths, so these map a cascade-local [0,1] UV into the atlas.
    glm::vec4 cascade_atlas_offset_u{}; // atlas-normalized U of each tile's left edge
    glm::vec4 cascade_atlas_scale_u{}; // tile width / atlas width, per cascade
    glm::vec4 cascade_atlas_scale_v{}; // tile height / atlas height, per cascade

    glm::vec3 light_direction{0.4F, 0.8F, 0.25F}; // normalized, points from surface to light
    float light_intensity = 3.0F;
    glm::vec3 light_colour{1.0F, 0.97F, 0.92F};
    float shadow_normal_offset_texels = 2.0F;

    std::uint32_t shadow_atlas_texture = 0; // bindless sampled_2d index
    std::uint32_t shadow_sampler = 0; // bindless comparison_samplers index
    float shadow_depth_bias_world = 0.02F;
    float shadow_pcf_radius_texels = 1.0F;

    std::uint32_t cascade_count = shadow_cascade_count;
    float shadow_atlas_texel_u = 1.0F / static_cast<float>(shadow_atlas_width);
    float shadow_atlas_texel_v = 1.0F / static_cast<float>(shadow_atlas_height);
    std::uint32_t shadow_debug_cascade_tint = 0;

    float time = 0.0F; // Seconds since startup; drives wind sway.

    // Flat multiplier on the ambient term, standing in for IBL.
    float ambient_intensity = 0.15F;

    // 0 disables screen-space AO, 1 applies it at full strength. Baked occlusion always applies.
    float ao_intensity = 1.0F;

    // Exponential depth slicing for the light clusters: slice = floor(log(view_z) * scale + bias).
    float cluster_z_scale = 1.0F;
    float cluster_z_bias = 0.0F;

    // 0 shades every light per fragment; otherwise the forward pass reads the cluster light lists.
    std::uint32_t clustered_lighting = 1;
    std::uint32_t cluster_debug_heatmap = 0;

    // ClusterGridSettings of the frame.
    std::uint32_t cluster_grid_x = 16;
    std::uint32_t cluster_grid_y = 9;
    std::uint32_t cluster_grid_z = 24;
    std::uint32_t cluster_light_capacity = 256;
};

static_assert(sizeof(UBO) == 748, "UBO layout changed -- update the mirror in assets/shaders/scene_types.slang");
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
    auto load_model(std::filesystem::path const &path) -> std::expected<ModelHandle, RendererError>;

    // Reserves a handle that renders as `fallback` until install_model() installs the real model.
    [[nodiscard]]
    auto create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, RendererError> override;

    // Installs a model ModelStreamer finished uploading into `pending`. Render thread only. On failure `pending`
    // keeps its fallback content.
    [[nodiscard]]
    auto install_model(ModelHandle pending, Model const &model) -> std::expected<void, RendererError> override;

    // See IModelSink::retain_model. Logs and does nothing if `handle` isn't live.
    auto retain_model(ModelHandle handle) -> void override;

    // See IModelSink::release_model. destroy_model() with the result discarded.
    auto release_model(ModelHandle handle) -> void override;

    auto register_model_name(ModelHandle handle, std::string_view name) -> void override;

    // Also seeds load_model()'s cache, so loading `source` again returns `handle` (with a new reference).
    auto register_model_source(ModelHandle handle, std::filesystem::path const &source) -> void override;

    // The model load_model(`source`) would return from its cache, without taking a reference; invalid if none.
    [[nodiscard]]
    auto cached_model(std::filesystem::path const &source) const -> ModelHandle;

    // The file a model was loaded from; nullptr for procedural models and handles that aren't live.
    [[nodiscard]]
    auto model_source(ModelHandle handle) const noexcept -> std::filesystem::path const *;

    // Drops a reference; the last one destroys the model's meshes, evicts it from the model caches and frees its
    // slot.
    [[nodiscard]]
    auto destroy_model(ModelHandle handle) -> std::expected<void, RendererError>;

    // Uploads CPU-side geometry, e.g. procedural primitives. Bypasses the path-based model cache.
    [[nodiscard]]
    auto create_model_from_cpu_data(ModelCpuData const &cpu_data) -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto create_model(Model const &model, MaterialHandle fallback_material)
            -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto create_model(Model const &model) -> std::expected<ModelHandle, RendererError>;

    // A valid material_override replaces every submesh material for this submission. Slot overrides win over it.
    [[nodiscard]]
    auto submit_model(ModelHandle model, glm::mat4 const &transform, MaterialHandle material_override = {},
                      std::span<MaterialSlotOverride const> slot_overrides = {}) -> std::expected<void, RendererError>;
    [[nodiscard]]
    auto submit_model(ModelHandle model, glm::mat4 &&, MaterialHandle material_override = {},
                      std::span<MaterialSlotOverride const> slot_overrides = {}) -> std::expected<void, RendererError>;

    // Submits many instances of one model sharing a material_override, without an entity per instance. Batching is
    // the same as for individual submissions.
    [[nodiscard]]
    auto submit_model_instances(ModelHandle model, std::span<glm::mat4 const> transforms,
                                MaterialHandle material_override = {}) -> std::expected<void, RendererError>;

    // Model-space AABB over every vertex.
    [[nodiscard]]
    auto model_bounds(ModelHandle model) const -> std::optional<std::pair<glm::vec3, glm::vec3>>;

    // One model-space AABB per (draw, submesh), in draw order, each folded through its draw's local transform.
    // Used to approximate large multi-part models with many boxes (see RigidBody::from_submesh_boxes).
    [[nodiscard]]
    auto model_submesh_bounds(ModelHandle model) const -> std::optional<std::vector<std::pair<glm::vec3, glm::vec3>>>;

    [[nodiscard]]
    auto model_lights(ModelHandle model) const -> std::span<ModelCpuLight const>;

    // Each distinct submesh material, in draw order. Edit them with update_material() to change the model itself
    // rather than overriding it per entity.
    [[nodiscard]]
    auto model_materials(ModelHandle model) const -> std::vector<MaterialHandle>;

    // A non-empty `debug_name` registers the material in assets() so the editor can offer it by name.
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

    // Unnames the material and drops its creator's reference; entities still using it keep it alive.
    [[nodiscard]]
    auto destroy_material(MaterialHandle handle) -> std::expected<void, RendererError>;

    // The name holds its own reference, dropped by destroy_material().
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
        float time = 0.0F; // Seconds since startup.
    };

    struct DirectionalLight {
        glm::vec3 direction{0.4F, 0.8F, 0.25F}; // normalized, points from surface to light
        glm::vec3 colour{1.0F, 0.97F, 0.92F};
        float intensity = 3.0F;
    };

    // Punctual lights are submitted per frame and never cast shadows.
    struct PointLight {
        glm::vec3 position{0.0F};
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
    };

    struct SpotLight {
        glm::vec3 position{0.0F};
        glm::vec3 direction{0.0F, -1.0F, 0.0F}; // normalized, points from the light outward
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

    // Multiplies the ambient term (albedo * AO), standing in for IBL.
    auto set_ambient_intensity(float intensity) noexcept -> void { ambient_intensity_ = intensity; }
    [[nodiscard]] auto ambient_intensity() const noexcept -> float { return ambient_intensity_; }

    auto set_fog_settings(FogSettings const &settings) noexcept -> void { fog_settings_ = settings; }
    [[nodiscard]] auto fog_settings() const noexcept -> FogSettings const & { return fog_settings_; }

    static_assert(shadow_cascade_count == 4, "Renderer shadow-cache defaults assume four cascades");

    struct ShadowSettings {
        ShadowCascadeSettings cascades{};
        float normal_offset_texels = 2.0F;
        float depth_bias_world = 0.02F;
        float pcf_radius_texels = 1.0F;
        float depth_bias_constant = -1.0F; // negative: reverse-Z
        float depth_bias_slope = -2.5F;

        // Minimum redraw interval per cascade, in frames. A cascade only adopts a new matrix when it is redrawn.
        std::array<std::uint32_t, shadow_cascade_count> cache_update_periods{1U, 2U, 4U, 8U};
        bool cache_enabled = true;
        bool debug_cascade_tint = false;
    };

    auto set_directional_light(DirectionalLight const &light) noexcept -> void { light_ = light; }
    [[nodiscard]] auto directional_light() const noexcept -> DirectionalLight const & { return light_; }

    auto set_shadow_settings(ShadowSettings const &settings) noexcept -> void { shadow_settings_ = settings; }
    [[nodiscard]] auto shadow_settings() const noexcept -> ShadowSettings const & { return shadow_settings_; }

    // Forces every cached cascade to redraw. Call after caster add/remove/material changes.
    auto mark_shadow_casters_dirty() noexcept -> void;

    // Call once per frame while casters outside the near cascade move. Far cascades still honour
    // cache_update_periods.
    auto mark_dynamic_shadow_casters_dirty() noexcept -> void { dynamic_shadow_casters_dirty_ = true; }

    [[nodiscard]]
    auto prepare_frame(VkCommandBuffer command_buffer, CameraMatrices const &, std::uint32_t frame_index)
            -> std::expected<void, RendererError>;

    // Records the frame's passes and overlays. Call after prepare_frame() for the same frame_index.
    [[nodiscard]] auto record_frame(FrameRecordInfo const &info) -> std::expected<void, RendererError>;

    // Registers an overlay (see overlay.hxx). It runs until the registration is destroyed, which must happen before
    // the Renderer is.
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
        return static_cast<float>(frames_[index].forward_target.extent().width) /
               static_cast<float>(frames_[index].forward_target.extent().height);
    }

    // Valid once record_frame() has run for this frame_index in embedded mode.
    [[nodiscard]] auto viewport_target(std::uint32_t index) const noexcept -> ImageHandle {
        return frames_[index].viewport_target.handle();
    }

    auto queue_render_thread_event(std::move_only_function<void()> &&) -> void;
    auto drain_event_queue() -> void;

    [[nodiscard]] auto context() noexcept -> VulkanContext & { return context_; }
    [[nodiscard]] auto depth_format() const noexcept { return frames_[0].forward_target.depth_format(); }
    [[nodiscard]] auto hdr_format() const noexcept { return frames_[0].forward_target.hdr_format(); }
    [[nodiscard]] auto samples() const noexcept { return frames_[0].forward_target.samples(); }

    [[nodiscard]] auto image_storage() noexcept -> ImageStorage & override { return image_storage_; }
    [[nodiscard]] auto material_storage() noexcept -> MaterialStorage & override { return material_storage_; }
    [[nodiscard]] auto sampler_storage() noexcept -> SamplerStorage & override { return sampler_storage_; }
    // Shared by editor_scene and runtime_scene so script handles survive the play() clone.
    [[nodiscard]] auto script_storage() noexcept -> ScriptStorage & { return script_storage_; }
    [[nodiscard]] auto script_storage() const noexcept -> ScriptStorage const & { return script_storage_; }
    [[nodiscard]] auto texture_streamer() noexcept -> TextureStreamer & override { return texture_streamer_; }
    [[nodiscard]] auto model_streamer() noexcept -> ModelStreamer & { return model_streamer_; }
    [[nodiscard]] auto resource_table() noexcept -> GpuResourceTable & { return gpu_resource_table_; }

    // Name -> handle lookup for the editor UI.
    [[nodiscard]] auto assets() noexcept -> AssetRegistry & { return assets_; }
    [[nodiscard]] auto assets() const noexcept -> AssetRegistry const & { return assets_; }

    // texture_streamer().request() plus registering `debug_name` in assets().textures().
    [[nodiscard]]
    auto request_texture(std::filesystem::path source_path, TextureRole role, ImageHandle fallback,
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

    [[nodiscard]] auto last_frame_timings() const noexcept -> StageTimings const & { return last_frame_timings_; }
    [[nodiscard]] auto last_frame_stats() const noexcept -> FrameStats const & { return last_frame_stats_; }
    [[nodiscard]] auto last_frame_pipeline_stats() const noexcept -> PipelineStats const & {
        return last_frame_pipeline_stats_;
    }
    [[nodiscard]] auto debug_draw_light_icons() const noexcept -> bool { return debug_draw_light_icons_; }
    auto set_debug_draw_light_icons(bool enabled) noexcept -> void { debug_draw_light_icons_ = enabled; }

    // Per-meshlet frustum and backface-cone culling in the task shader. Turning it off is a debugging aid.
    [[nodiscard]] auto meshlet_culling() const noexcept -> bool { return meshlet_culling_; }
    auto set_meshlet_culling(bool enabled) noexcept -> void { meshlet_culling_ = enabled; }

    // Punctual lights binned into view-space clusters on the GPU, so each fragment only shades the lights that can
    // reach it. Off shades every light per fragment.
    [[nodiscard]] auto clustered_lighting() const noexcept -> bool { return clustered_lighting_; }
    auto set_clustered_lighting(bool enabled) noexcept -> void { clustered_lighting_ = enabled; }

    // Tints the scene by the number of lights in each fragment's cluster.
    [[nodiscard]] auto cluster_debug_heatmap() const noexcept -> bool { return cluster_debug_heatmap_; }
    auto set_cluster_debug_heatmap(bool enabled) noexcept -> void { cluster_debug_heatmap_ = enabled; }

    // The clustered-lighting grid. A new one applies from the next frame: each frame in flight resizes its cluster
    // lists the next time it is prepared. A grid validate_cluster_grid() rejects is refused with its reason.
    [[nodiscard]] auto cluster_grid() const noexcept -> ClusterGridSettings const & { return cluster_grid_; }
    auto set_cluster_grid(ClusterGridSettings const &grid) -> std::expected<void, std::string>;

    // light_cluster.slang's statistics from the latest frame whose readback has landed; invalid while clustering is
    // off.
    [[nodiscard]] auto last_cluster_stats() const noexcept -> ClusterStats const & { return last_cluster_stats_; }

    // Captures the viewport target (the scene alone) or the whole composited window. The viewport is only there in the
    // editor; fullscreen play falls back to the window.
    auto request_screenshot(ScreenshotSource source) noexcept -> void;
    auto mark_lights_dirty() -> void { lights_dirty_mask_ = frames_.empty() ? 0U : ((1U << frames_.size()) - 1U); }
    auto wait_idle() -> std::expected<void, RendererError>;

    static auto compiler() noexcept -> renderer::SlangCompiler &;

private:
    struct Submission {
        MeshHandle mesh{};
        glm::mat4 transform{1.0F};
        MaterialHandle material_override{};
    };

    // Mirrors GpuDraw in scene_types.slang.
    struct alignas(16) GpuDraw {
        VkDeviceAddress vertex_address = 0;
        VkDeviceAddress meshlet_address = 0;
        VkDeviceAddress meshlet_data_address = 0;

        std::uint32_t material_index = 0;
        std::uint32_t transform_index = 0;
    };

    static_assert(std::is_trivially_copyable_v<GpuDraw>);

    static_assert(sizeof(GpuDraw) == 32);

    // Local-space AABB for one batch. Mirrors GpuCullBounds in frustum_cull.slang. wind_padding grows the X/Z
    // extents to cover the maximum wind sway, so swaying foliage doesn't pop at the frustum edge.
    struct alignas(16) GpuCullBounds {
        glm::vec3 bounds_min{-0.5F};
        float wind_padding = 0.0F;
        glm::vec3 bounds_max{0.5F};
        float pad1 = 0.0F;
    };

    static_assert(std::is_trivially_copyable_v<GpuCullBounds>);

    static_assert(sizeof(GpuCullBounds) == 32);

    enum class GpuLightType : std::uint32_t {
        point = 0,
        spot = 1,
    };

    // Mirrors GpuLight in scene_types.slang. spot_scale/spot_offset are the precomputed KHR_lights_punctual cone
    // falloff terms.
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

    // Point and spot lights together. Each frame in flight keeps sizeof(GpuLight) per light of host-visible memory,
    // plus 20 bytes for light_cull.slang's visible list.
    static constexpr std::uint32_t maximum_light_count = 65'536;

    // Camera frustum plus one per shadow cascade, 6 planes each.
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

        // GPU culling, one workgroup per batch: reads indirect_buffer and batch_bounds, writes culled_indirect_buffer
        // and compacts visible instances into visible_draw_buffer/visible_transform_buffer for the main view. The
        // shadow pass draws the un-culled buffers.
        Buffer batch_bounds_buffer{};
        Buffer culled_indirect_buffer{};
        Buffer visible_draw_buffer{};
        Buffer visible_transform_buffer{};

        // Host-visible copy of culled_indirect_buffer, grown on demand. Read at the start of record_frame for the same
        // frame_index, so it lags a frames-in-flight cycle.
        Buffer culled_readback_buffer{};
        std::uint32_t culled_readback_capacity = 0;
        std::uint32_t culled_readback_count = 0;
        bool culled_readback_pending = false;

        // Camera planes first, then 6 per shadow cascade. A separate buffer rather than a UBO array to get an
        // unambiguous 16-byte stride.
        Buffer frustum_planes_buffer{};

        // Punctual lights, maximum_light_count capacity. light_count slots are populated.
        Buffer lights_buffer{};
        std::uint32_t light_count = 0;

        // light_cull.slang's output: maximum_light_count view-space spheres, then as many light indices, then the
        // visible count.
        Buffer visible_lights_buffer{};

        // light_cluster.slang's output, laid out for cluster_grid (cluster_buffer_bytes): the statistics, then the
        // counts and lists the forward fragment shader reads.
        Buffer cluster_lights_buffer{};
        ClusterGridSettings cluster_grid{};

        // Host-visible copy of the statistics, read when this frame slot is next prepared.
        Buffer cluster_stats_readback_buffer{};
        ClusterGridSettings cluster_stats_grid{};
        bool cluster_stats_pending = false;

        // Handed to scene overlays as OverlayRecordContext::view_projection.
        glm::mat4 view_projection{1.0F};

        ForwardTarget forward_target{};

        // LDR composite output sampled by the editor's Viewport panel. Unused in fullscreen play.
        ImageHolder viewport_target{};

        // mip_slots are register_view() aliases of image's mip views, so they're released before it: declared after
        // it for destruction, and assigned first on a move.
        struct BloomTarget {
            ImageHolder image;
            std::array<ImageHolder, render_pass::bloom_mip_count> mip_slots;

            BloomTarget() = default;
            ~BloomTarget() = default;

            BloomTarget(BloomTarget const &) = delete;
            auto operator=(BloomTarget const &) -> BloomTarget & = delete;

            BloomTarget(BloomTarget &&) noexcept = default;

            auto operator=(BloomTarget &&other) noexcept -> BloomTarget & {
                mip_slots = std::move(other.mip_slots);
                image = std::move(other.image);

                return *this;
            }
        };
        BloomTarget bloom_target{};

        // Per-frame GTAO targets: `raw` from the horizon search, `denoised` sampled by the forward pass.
        struct AoTarget {
            ImageHolder raw;
            ImageHolder denoised;
        };
        AoTarget ao_target{};

        // Bit i means cascade i is redrawn into the persistent atlas this frame.
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

        std::vector<GpuDraw> draws;
        std::vector<glm::mat4> transforms;

        // One un-culled command per batch; the shadow pass draws these directly.
        std::vector<GpuDrawCommand> indirect_commands;

        // Parallel to indirect_commands.
        std::vector<GpuCullBounds> batch_bounds;

        // Number of batches, not instances. This is the drawCount for vkCmdDrawMeshTasksIndirectEXT.
        std::uint32_t indirect_command_count = 0;

        // Batches are ordered opaque, mask, blend; culling preserves the order.
        std::uint32_t opaque_indirect_count = 0;
        std::uint32_t mask_indirect_count = 0;
        std::uint32_t blend_indirect_count = 0;

        // Per-cascade prefix of the opaque/mask ranges. Batches are sorted by descending max_shadow_cascade, so a
        // material can skip the far cascades.
        std::array<std::uint32_t, shadow_cascade_count> shadow_opaque_indirect_count{};
        std::array<std::uint32_t, shadow_cascade_count> shadow_mask_indirect_count{};
    };

    // A frame's extent-sized render targets, built together so initialize() and resize() share one path.
    struct OwnedFrameTargets {
        ForwardTarget forward_target{};
        ImageHolder viewport_target{};
        RendererFrame::BloomTarget bloom_target{};
        RendererFrame::AoTarget ao_target{};
    };

    struct ModelSubmission {
        ModelHandle model{};
        glm::mat4 transform{1.0F};
        MaterialHandle material_override{};

        // A range of slot_override_submissions_.
        std::uint32_t slot_override_first = 0;
        std::uint32_t slot_override_count = 0;
    };

    struct BatchEntry {
        MeshHandle mesh{};
        std::uint32_t submesh_index = 0;
        MaterialHandle material{};
        std::uint32_t lod_index = 0;

        std::vector<glm::mat4> transforms;

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

        auto operator==(BatchKey const &) const noexcept -> bool = default;
    };

    struct BatchKeyHash {
        auto operator()(BatchKey const &key) const noexcept -> std::size_t {
            auto const mesh_hash =
                    std::hash<std::uint64_t>{}((static_cast<std::uint64_t>(key.mesh_index) << 32) | key.submesh_index);

            return mesh_hash ^ (std::hash<std::uint32_t>{}(key.material_index) << 1) ^
                   (std::hash<std::uint32_t>{}(key.lod_index) << 2);
        }
    };

    std::unordered_map<BatchKey, BatchEntry, BatchKeyHash> batches_;
    std::vector<BatchEntry *> active_batches_;

    std::vector<BatchEntry const *> opaque_batches_;
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

    // Shared by create_model() and finish_model_load(): creates the model's meshes, flattens its draws and hands the
    // ModelSlotData to `install`. Rolls back the meshes if `install` fails.
    [[nodiscard]]
    auto
    create_model_common(Model const &model, MaterialHandle fallback_material,
                        std::move_only_function<std::expected<ModelHandle, ModelStorageError>(ModelSlotData)> install)
            -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto upload_frame_data(VkCommandBuffer command_buffer, RendererFrame &frame) -> std::expected<void, RendererError>;

    auto clear_submissions() noexcept -> void;

    // Copied at record time so the timing readback doesn't depend on the overlay still being registered.
    struct RecordedOverlay {
        std::string name;
        OverlayStage stage = OverlayStage::scene;
        std::uint32_t slot = 0;
    };

    struct FrameTimestamps {
        VkQueryPool query_pool{VK_NULL_HANDLE};
        bool has_results{false};
        std::vector<RecordedOverlay> overlays;
    };
    // record_frame() and its passes. Each record_*_pass owns one stage and passes its output to the next through
    // its return value.

    // The frame's images, resolved and validated once. resolved_hdr/resolved_depth are the MSAA resolve targets
    // when multisampled, otherwise hdr/depth.
    struct FrameTargets {
        Image const *hdr = nullptr;
        Image const *depth = nullptr;
        Image const *resolved_hdr = nullptr;
        Image const *resolved_depth = nullptr;
        ImageHandle resolved_hdr_handle{};
        ImageHandle resolved_depth_handle{};

        Image const *shadow_atlas = nullptr;
        Image const *ao_raw = nullptr;
        Image const *ao_denoised = nullptr;
        Image const *viewport = nullptr;

        VkExtent2D extent{};
        bool multisampled = false;
    };

    // Folds the culled-indirect readback recorded the last time this frame
    // slot was used into last_frame_stats_.
    auto consume_culled_readback(RendererFrame &frame) -> void;

    // Reads the cluster statistics recorded the last time this frame slot was used into last_cluster_stats_, then
    // resizes the slot's cluster lists to cluster_grid_ if it changed. The slot's fence has been waited on.
    [[nodiscard]]
    auto prepare_cluster_buffers(RendererFrame &frame) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto resolve_frame_targets(RendererFrame const &frame) const -> std::expected<FrameTargets, RendererError>;

    // The culled, compacted buffers the camera passes draw from.
    [[nodiscard]]
    auto main_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers;

    // Opaque/mask/blend batch counts, shared by the culled and un-culled buffers.
    [[nodiscard]]
    static auto batch_counts(RendererFrame const &frame) noexcept -> render_pass::DrawCounts;

    [[nodiscard]]
    auto record_shadow_pass(render_pass::Context const &pass_context, RendererFrame const &frame,
                            FrameTargets const &targets) -> std::expected<void, RendererError>;

    // Also transitions the forward targets into attachment layouts.
    [[nodiscard]]
    auto record_depth_prepass(render_pass::Context const &pass_context, RendererFrame const &frame,
                              FrameTargets const &targets) -> std::expected<void, RendererError>;

    // Returns the AO texture's bindless index: denoised GTAO, or white when disabled.
    [[nodiscard]]
    auto record_ambient_occlusion_pass(render_pass::Context const &pass_context, RendererFrame const &frame,
                                       FrameTargets const &targets) -> std::expected<std::uint32_t, RendererError>;

    [[nodiscard]]
    auto record_forward_pass(render_pass::Context const &pass_context, RendererFrame const &frame,
                             FrameTargets const &targets, std::uint32_t ao_texture_index,
                             render_pass::Callback scene_overlays)
            -> std::expected<render_pass::HdrTextureIndex, RendererError>;

    [[nodiscard]]
    auto record_bloom_pass(render_pass::Context const &pass_context, RendererFrame const &frame,
                           FrameTargets const &targets, render_pass::HdrTextureIndex hdr)
            -> std::expected<std::optional<render_pass::BloomTextureIndex>, RendererError>;

    // Tonemaps hdr + bloom into the swapchain (fullscreen play) or the viewport target, then draws the UI.
    [[nodiscard]]
    auto record_composite_pass(render_pass::Context const &pass_context, FrameTargets const &targets,
                               SwapchainImage const &swapchain_image, render_pass::HdrTextureIndex hdr,
                               std::optional<render_pass::BloomTextureIndex> bloom, CompositeTarget target,
                               render_pass::Callback ui_overlays) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto make_pass_context(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> render_pass::Context;

    // Runs every overlay's prepare(), snapshots the overlay list for timing, and records one barrier if any
    // prepare() wrote GPU data.
    auto record_overlay_prepares(render_pass::Context const &pass_context) -> void;

    // Runs one stage's overlays inside the host pass's rendering scope.
    auto record_overlay_stage(render_pass::Context const &pass_context, OverlayStage stage, OverlayScope const &scope,
                              glm::mat4 const &view_projection) -> void;

    [[nodiscard]]
    auto register_light_icon_overlay() -> std::expected<void, RendererError>;

    // Fills last_frame_timings_.overlays from a retired frame's queries.
    auto read_overlay_timings(FrameTimestamps const &frame_query) -> void;

    // Uses hdr_format_, depth_format_, samples_ and swapchain_format_. On failure, whatever was created is destroyed.
    [[nodiscard]]
    auto create_frame_targets(std::uint32_t frame_index, VkExtent2D extent)
            -> std::expected<OwnedFrameTargets, RendererError>;

    // Screenshot copy or present transition, then the end-of-frame timestamp.
    // viewport is null when the scene was composited straight into the swapchain.
    auto record_frame_end(VkCommandBuffer command_buffer, SwapchainImage const &swapchain_image, Image const *viewport,
                          std::uint32_t frame_index) -> void;

    VulkanContext &context_;

    VkFormat hdr_format_ = VK_FORMAT_UNDEFINED;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    // Kept so resize() can create viewport targets in the composite output format.
    VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
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
    PipelineNodeHandle forward_blend_pipeline_;

    // Vertex-shader variants of the scene pipelines for plain-instanced batches.
    PipelineNodeHandle shadow_instanced_pipeline_;
    PipelineNodeHandle shadow_mask_instanced_pipeline_;
    PipelineNodeHandle depth_prepass_instanced_pipeline_;
    PipelineNodeHandle depth_prepass_mask_instanced_pipeline_;
    PipelineNodeHandle forward_instanced_pipeline_;
    PipelineNodeHandle forward_blend_instanced_pipeline_;
    PipelineNodeHandle composite_pipeline_;
    PipelineNodeHandle frustum_cull_pipeline_;
    PipelineNodeHandle light_icon_pipeline_;
    PipelineNodeHandle bloom_downsample_pipeline_;
    PipelineNodeHandle bloom_upsample_pipeline_;
    PipelineNodeHandle gtao_pipeline_;
    PipelineNodeHandle gtao_denoise_pipeline_;
    PipelineNodeHandle light_cull_pipeline_;
    PipelineNodeHandle light_cluster_pipeline_;
    ShaderChangeQueue shader_change_queue_;

    BloomSettings bloom_settings_;
    AoSettings ao_settings_;

    ImageHandle light_icon_texture_{};
    bool debug_draw_light_icons_ = false;

    // Declared before any registration it hands out, so it outlives them.
    OverlayRegistry overlays_;
    OverlayRegistration light_icon_overlay_;
    bool meshlet_culling_ = true;
    bool clustered_lighting_ = true;
    bool cluster_debug_heatmap_ = false;
    ClusterGridSettings cluster_grid_{};
    ClusterStats last_cluster_stats_{};
    float light_icon_world_size_ = 0.5F;

    // Shared across frames in flight so unchanged tiles persist.
    ImageHolder shadow_atlas_{};
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

    std::vector<PointLight> point_light_submissions_;
    std::vector<SpotLight> spot_light_submissions_;

    MeshStorage mesh_storage_;
    ModelStorage model_storage_;
    ScriptStorage script_storage_;

    std::unordered_map<std::size_t, ModelHandle> model_cache_;

    // Keyed by (generation << 32 | index).
    std::unordered_map<std::uint64_t, std::filesystem::path> model_sources_;

    std::vector<Submission> submissions_;
    std::vector<ModelSubmission> model_submissions_;
    std::vector<MaterialSlotOverride> slot_override_submissions_;

    std::vector<RendererFrame> frames_;

    MaterialHandle default_material_handle_{};

    std::uint32_t maximum_draw_count_ = 0;
    std::uint32_t maximum_submission_count_ = 0;

    StageTimings last_frame_timings_{};
    FrameStats last_frame_stats_{};

    std::vector<GpuLight> light_staging_;
    std::uint32_t lights_dirty_mask_ = 0;
    std::uint32_t light_count_ = 0;

    std::queue<std::move_only_function<void()>> event_queue_;
    std::atomic_uint32_t queued_events_;
    std::mutex queue_mutex_;

    std::vector<FrameTimestamps> timestamp_queries_;
    float timestamp_period_{1.0F};

    struct FramePipelineQuery {
        VkQueryPool query_pool{VK_NULL_HANDLE};
        bool has_results{false};
    };
    std::vector<FramePipelineQuery> pipeline_stat_queries_;
    PipelineStats last_frame_pipeline_stats_{};

    std::unique_ptr<ScreenshotCapture> screenshot_;

    bool initialized_ = false;
};
