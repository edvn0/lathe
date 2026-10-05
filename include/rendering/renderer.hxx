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
#include "gpu/submission_plan.hxx"
#include "rendering/cluster_grid.hxx"
#include "rendering/environment.hxx"
#include "rendering/forward_target.hxx"
#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/compiler.hxx"
#include "rendering/frame_graph/frame_graph.hxx"
#include "rendering/frame_graph/pass_profiler.hxx"
#include "rendering/frame_graph/transient_allocator.hxx"
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

    // Upsample tent radius in texels of the lower mip; 1.0 is the standard 3x3 tent.
    float filter_radius = 1.0F;

    // Scale applied to the bloom before it is added to the HDR colour.
    float intensity = 0.1F;
};

// The selected-object outline: submit_model(..., outlined = true) marks what it draws in a mask the composite pass
// traces. Drawn outside the silhouette, over the tone-mapped image, so the colour is what you see.
struct OutlineSettings {
    glm::vec3 colour{1.0F, 0.78F, 0.15F};

    // Width in pixels of the target the scene is drawn into.
    float thickness_pixels = 3.0F;
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

// Screen-size LOD for punctual lights: a light whose range covers fewer than cull_radius_pixels of radius on screen
// is culled before clustering, and lights fade in over [cull_radius_pixels, fade_radius_pixels].
struct LightLodSettings {
    bool enabled = true;
    float cull_radius_pixels = 2.0F;
    float fade_radius_pixels = 6.0F;
};

// The last finished frame's GPU times: the frame graph's per-pass timings (graphics queue first, in execution order)
// and the frame as a whole, from its first to its last graphics timestamp. Compute work that overlaps runs inside that
// span.
struct FrameTimings {
    std::vector<frame_graph::PassTiming> passes;
    float full_frame_ms = 0.0F;

    // Overlays that ran in the timed frame, in draw order. Their time is already included in the passes'.
    std::vector<OverlayTiming> overlays;

    bool valid = false;

    // Which recorded frame these timings belong to: Renderer::recorded_frame_count() just after that frame was
    // recorded. Readback lags by the frames in flight, so this is how a consumer lines them up with its own per-frame
    // data. 0 for timings that were never tagged.
    std::uint64_t frame_serial = 0;
};

struct FrameStats {
    std::uint32_t submitted_triangle_count = 0;
    std::uint32_t submitted_instance_count = 0;

    std::uint32_t indirect_command_count = 0;
    std::uint32_t opaque_indirect_count = 0;
    std::uint32_t double_sided_indirect_count = 0;
    std::uint32_t mask_indirect_count = 0;
    std::uint32_t blend_indirect_count = 0;

    // Instances drawn by the camera passes after GPU culling (early + late). Like every count below it, this lags a
    // frames-in-flight cycle behind the rest; 0 until the first readback.
    std::uint32_t visible_instance_count = 0;

    // Two-phase occlusion culling (docs/occlusion-culling.md), valid when occlusion_stats_valid. Frustum-visible
    // instances are either drawn in phase 1 (early) or deferred as candidates; phase 2 draws the candidates that
    // pass against this frame's Hi-Z (late) and drops the rest (occluded).
    std::uint32_t frustum_visible_instance_count = 0;
    std::uint32_t early_instance_count = 0;
    std::uint32_t occlusion_candidate_count = 0;
    std::uint32_t late_instance_count = 0;
    std::uint32_t occluded_instance_count = 0;
    bool occlusion_stats_valid = false;

    // Meshlet-level occlusion (docs/occlusion-culling.md, "Meshlet level"), valid when meshlet_occlusion_stats_valid.
    // Of the meshlets that pass the frustum and cone tests: deferred ones were hidden by last frame's Hi-Z in phase 1
    // and retested in phase 2; occluded ones were hidden by this frame's phase-1 depth and are culled for good.
    std::uint32_t deferred_meshlet_count = 0;
    std::uint32_t occluded_meshlet_count = 0;
    bool meshlet_occlusion_stats_valid = false;

    std::uint32_t model_submission_count = 0;
    std::uint32_t mesh_submission_count = 0;

    std::uint32_t point_light_count = 0;
    std::uint32_t spot_light_count = 0;
};

inline constexpr std::uint32_t pipeline_stat_count = 4;

// Renderer::occlusion_test_mode(). The stubs exercise the two-phase draw lists without trusting the Hi-Z test; with
// either, a frame must look exactly as with occlusion culling off.
enum class OcclusionTestMode : std::uint8_t {
    // Phase 1 tests against last frame's Hi-Z, phase 2 against this frame's.
    hiz,

    // Stub: phase 1 draws every frustum-visible instance; phase 2 is empty.
    never_occluded,

    // Stub: phase 1 defers every frustum-visible opaque/mask instance and phase 2 draws them all.
    always_defer,
};

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

    // Screen-size light LOD (LightLodSettings). pixel_scale turns range / distance into an on-screen radius in
    // pixels. fade_radius_pixels of 0 disables it.
    float light_lod_pixel_scale = 0.0F;
    float light_lod_cull_radius_pixels = 0.0F;
    float light_lod_fade_radius_pixels = 0.0F;

    // Environment lighting and the skybox; flat in the shader's UBO (environment_flags and following).
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
    //
    // `outlined` draws the selected-object outline (outline_settings()) around the model. It draws with private
    // copies of its materials that carry the outline flag, so it costs a batch of its own per material. The model's
    // own materials, not InstancedModel ones, are what it supports.
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

    // Submits many instances of one model sharing a material_override, without an entity per instance. Batching is
    // the same as for individual submissions, but the transforms are copied in one block and prepare_frame() resolves
    // each (submesh, LOD) batch once per call rather than once per instance.
    //
    // A non-zero `resident_revision` names this exact set of transforms (Components::InstancedModel::revision): the
    // first submission uploads them to a GPU buffer kept while the revision keeps being submitted, and the GPU picks
    // each instance's LOD (instance_lod.slang), so a frame costs the CPU nothing per instance. A model with one draw
    // at the origin of its node and no blended LOD qualifies; anything else takes the per-instance path.
    [[nodiscard]]
    auto submit_model_instances(ModelHandle model, std::span<glm::mat4 const> transforms,
                                MaterialHandle material_override = {},
                                std::uint64_t resident_revision = 0) -> std::expected<void, RendererError>;

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

    auto set_light_lod_settings(LightLodSettings const &settings) noexcept -> void { light_lod_settings_ = settings; }
    [[nodiscard]] auto light_lod_settings() const noexcept -> LightLodSettings const & { return light_lod_settings_; }

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

    // The scene's environment: IBL, the skybox, fog and, when it asks to, the directional light (from the sun). Call
    // every frame; it only does work when something changed.
    auto set_environment(SceneEnvironment const &environment) -> void;
    [[nodiscard]] auto environment_system() noexcept -> EnvironmentSystem & { return environment_; }
    [[nodiscard]] auto environment_system() const noexcept -> EnvironmentSystem const & { return environment_; }
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
    //
    // The frame runs through the frame graph: today one legacy pass around the old recording body, so the result is
    // one graphics batch in `info.command_buffer`, which stays open for the caller to end. Submit submit_batches()
    // after ending it.
    [[nodiscard]] auto record_frame(FrameRecordInfo const &info) -> std::expected<void, RendererError>;

    // The batches the last record_frame() produced, in submission order. Empty if it failed before producing any.
    // Valid until the next record_frame().
    [[nodiscard]] auto submit_batches() const noexcept -> std::span<SubmitBatch const> { return submit_batches_; }

    // The frame graph's per-pass GPU times from the most recent frame slot that finished (graphics queue first).
    [[nodiscard]] auto frame_graph_timings() const noexcept -> std::span<frame_graph::PassTiming const> {
        return pass_profiler_.timings();
    }

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
        static_cast<void>(index);
        return static_cast<float>(extent_.width) / static_cast<float>(extent_.height);
    }

    // Valid once record_frame() has run for this frame_index in embedded mode.
    [[nodiscard]] auto viewport_target(std::uint32_t index) const noexcept -> ImageHandle {
        return frames_[index].viewport_target.handle();
    }

    auto queue_render_thread_event(std::move_only_function<void()> &&) -> void;
    auto drain_event_queue() -> void;

    [[nodiscard]] auto context() noexcept -> VulkanContext & { return context_; }
    [[nodiscard]] auto depth_format() const noexcept { return depth_format_; }
    [[nodiscard]] auto hdr_format() const noexcept { return hdr_format_; }
    [[nodiscard]] auto samples() const noexcept { return samples_; }

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

    [[nodiscard]] auto last_frame_timings() const noexcept -> FrameTimings const & { return last_frame_timings_; }

    // Frames recorded through record_frame_end() so far. Read it right after recording a frame to get the serial its
    // FrameTimings will carry once they are read back.
    [[nodiscard]] auto recorded_frame_count() const noexcept -> std::uint64_t { return recorded_frame_count_; }
    [[nodiscard]] auto last_frame_stats() const noexcept -> FrameStats const & { return last_frame_stats_; }

    // Frame graph transients (the AO and bloom images, over every frame slot): the device memory they occupy and what
    // they would take if none shared memory. Aliasing is on by default; off is for A/B runs.
    [[nodiscard]] auto transient_bytes() const noexcept -> std::uint64_t { return transient_allocator_.total_bytes(); }
    [[nodiscard]] auto transient_unaliased_bytes() const noexcept -> std::uint64_t {
        return transient_allocator_.unaliased_bytes();
    }
    // Which groups of compute passes are declared with compute-queue affinity (phase 6): each candidate is enabled by
    // measurement. They only run on another queue when the device has one and --async-compute allows it.
    enum AsyncCandidate : std::uint8_t {
        async_light_clustering = 1U << 0U, // light_cull and light_cluster
        async_occlusion = 1U << 1U, // hiz_build and late_cs, overlapping the shadows (declared after the early prepass)
        async_gtao = 1U << 2U, // gtao and its denoise, overlapping the shadows (declared after the late prepass)
    };
    [[nodiscard]] auto async_candidates() const noexcept -> std::uint32_t { return async_candidates_; }
    auto set_async_candidates(std::uint32_t mask) noexcept -> void { async_candidates_ = mask; }

    [[nodiscard]] auto transient_aliasing() const noexcept -> bool { return transient_aliasing_; }
    auto set_transient_aliasing(bool enabled) noexcept -> void { transient_aliasing_ = enabled; }

    // Log the full compiled plan (batches, waits, barriers, transfers, transient placement) when it changes.
    auto set_frame_graph_dump(bool enabled) noexcept -> void { dump_frame_graph_ = enabled; }
    [[nodiscard]] auto last_frame_pipeline_stats() const noexcept -> PipelineStats const & {
        return last_frame_pipeline_stats_;
    }
    [[nodiscard]] auto debug_draw_light_icons() const noexcept -> bool { return debug_draw_light_icons_; }
    auto set_debug_draw_light_icons(bool enabled) noexcept -> void { debug_draw_light_icons_ = enabled; }

    // Per-meshlet frustum and backface-cone culling in the task shader. Turning it off is a debugging aid.
    [[nodiscard]] auto meshlet_culling() const noexcept -> bool { return meshlet_culling_; }
    auto set_meshlet_culling(bool enabled) noexcept -> void { meshlet_culling_ = enabled; }

    // Two-phase Hi-Z occlusion culling of whole instances (docs/occlusion-culling.md). Off by default. Changing it
    // drops the Hi-Z history, so the next frame draws every frustum-visible instance in phase 1.
    [[nodiscard]] auto occlusion_culling() const noexcept -> bool { return occlusion_culling_; }
    auto set_occlusion_culling(bool enabled) noexcept -> void;

    // False under MSAA on devices without VK_RESOLVE_MODE_MIN_BIT depth resolves; occlusion culling then stays
    // inactive whatever occlusion_culling() says.
    [[nodiscard]] auto occlusion_culling_supported() const noexcept -> bool;

    // Hi-Z occlusion of individual meshlets in the task shader (docs/occlusion-culling.md, "Meshlet level"). Off by
    // default. Only active while occlusion_culling() and meshlet_culling() are on; the depth prepass phases record the
    // meshlets they emit and the forward pass replays that record.
    [[nodiscard]] auto meshlet_occlusion_culling() const noexcept -> bool { return meshlet_occlusion_culling_; }
    auto set_meshlet_occlusion_culling(bool enabled) noexcept -> void { meshlet_occlusion_culling_ = enabled; }

    // A debugging aid: the stub modes check the two-phase draw-list plumbing independently of the Hi-Z test.
    [[nodiscard]] auto occlusion_test_mode() const noexcept -> OcclusionTestMode { return occlusion_test_mode_; }
    auto set_occlusion_test_mode(OcclusionTestMode mode) noexcept -> void;

    // One level of the Hi-Z pyramid as a sampled_2d texture for the debug panel: R32, the farthest depth of each texel
    // (reverse-Z, so brighter is nearer). Only the level's logical extent (hiz_level_extent() of hiz_depth_extent())
    // is written; the rest of the power-of-two image is undefined. Invalid until occlusion culling has built the
    // pyramid since the last resize, or when R32_SFLOAT can't be linearly filtered for the UI.
    [[nodiscard]] auto hiz_debug_view(std::uint32_t mip) const noexcept -> ImageHandle;
    [[nodiscard]] auto hiz_debug_mip_count() const noexcept -> std::uint32_t { return hiz_.mip_count; }
    [[nodiscard]] auto hiz_depth_extent() const noexcept -> VkExtent2D { return hiz_.depth_extent; }

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

        // The first bit of this instance's meshlets in the frame's meshlet visibility bitset
        // (include/rendering/meshlet_visibility.hxx); 0 and never read for instanced batches.
        std::uint32_t meshlet_visibility_offset = 0;
    };

    static_assert(std::is_trivially_copyable_v<GpuDraw>);

    static_assert(sizeof(GpuDraw) == 32);

    // Local-space AABB for one batch. Mirrors GpuCullBounds in frustum_cull.slang. wind_padding grows the X/Z
    // extents to cover the maximum wind sway, so swaying foliage doesn't pop at the frustum edge. first_chunk is the
    // batch's first culling chunk (cull_chunk_size instances each); batches' chunks are consecutive.
    struct alignas(16) GpuCullBounds {
        glm::vec3 bounds_min{-0.5F};
        float wind_padding = 0.0F;
        glm::vec3 bounds_max{0.5F};
        std::uint32_t first_chunk = 0;
    };

    // GPU culling splits each batch into chunks of this many instances, one workgroup each (frustum_cull.slang).
    static constexpr std::uint32_t cull_chunk_size = 256;

    // sizeof(CullChunk) in frustum_cull.slang: 8 counters and three cull_chunk_size-bit masks.
    static constexpr VkDeviceSize cull_chunk_bytes = (8 + 3 * cull_chunk_size / 32) * sizeof(std::uint32_t);

    // Mirrors LodGroup in instance_lod.slang: one output batch of a resident instanced model's submesh.
    struct alignas(16) GpuLodGroup {
        std::uint32_t batch = 0;
        std::uint32_t first_instance = 0;
        std::uint32_t meshlet_count = 0;
        std::uint32_t first_meshlet_bit = 0;
        GpuDraw draw{};
    };

    static_assert(sizeof(GpuLodGroup) == 48);

    // Mirrors LodJob in instance_lod.slang: one submesh of one resident instanced model.
    struct alignas(16) GpuLodJob {
        VkDeviceAddress transforms_address = 0;
        std::uint32_t instance_count = 0;
        std::uint32_t first_chunk = 0;
        std::uint32_t group_count = 0;
        std::uint32_t lod_groups = 0; // byte `lod`: the group LOD `lod` draws with
        std::uint32_t pad0 = 0;
        std::uint32_t pad1 = 0;
        std::array<GpuLodGroup, lod_count> groups{};
    };

    static_assert(std::is_trivially_copyable_v<GpuLodJob>);
    static_assert(sizeof(GpuLodJob) == 32 + 48 * lod_count);

    static_assert(std::is_trivially_copyable_v<GpuCullBounds>);

    static_assert(sizeof(GpuCullBounds) == 32);

    // Mirrors OcclusionView in hiz_occlusion.slang. frame.occlusion_views_buffer holds two: [0] last frame's pyramid
    // with the view-projection it was built with (phase 1), [1] this frame's (phase 2).
    struct alignas(16) GpuOcclusionView {
        glm::mat4 view_projection{1.0F};

        // The frame's meshlet visibility bitset while meshlet occlusion is active, else 0 (docs/occlusion-culling.md,
        // "Meshlet level").
        VkDeviceAddress meshlet_visibility_address = 0;

        // The counter the task shader adds this view's occlusion rejections to: occlusion_stat_deferred_meshlets for
        // view [0], occlusion_stat_occluded_meshlets for view [1]. 0 until a task shader needs it.
        VkDeviceAddress stats_address = 0;

        // The whole mip chain, read via sampled_2d_depth[...].Load(int3(texel, level)).
        std::uint32_t hiz_texture_index = 0;
        std::uint32_t hiz_mip_count = 0;
        std::uint32_t depth_width = 0;
        std::uint32_t depth_height = 0;

        // occlusion_view_* in renderer.cxx: disabled (never occluded), enabled, or the always_defer stub.
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

    // Slots of frame.occlusion_stats_buffer, accumulated by frustum_cull.slang. Mirrors occlusion_stat_* there.
    static constexpr std::uint32_t occlusion_stat_frustum_visible = 0;
    static constexpr std::uint32_t occlusion_stat_early = 1;
    static constexpr std::uint32_t occlusion_stat_candidates = 2;
    static constexpr std::uint32_t occlusion_stat_late = 3;
    // Accumulated by the task shader (meshlet_task.slang) through each view's stats_address.
    static constexpr std::uint32_t occlusion_stat_deferred_meshlets = 4;
    static constexpr std::uint32_t occlusion_stat_occluded_meshlets = 5;
    static constexpr std::uint32_t occlusion_stat_count = 8;

    // One cull workgroup per batch, so batches are capped at the guaranteed maxComputeWorkGroupCount[0].
    static constexpr std::uint32_t maximum_cull_batch_count = 65'535;

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

        // Two-phase occlusion culling (docs/occlusion-culling.md). main_cs defers frustum-visible instances last
        // frame's Hi-Z hides: their source indices go to occlusion_candidates_buffer at the batch's first_instance and
        // their count to the batch's first culling chunk. late_cs appends the candidates that pass to the visible
        // buffers after main_cs's survivors and writes late_indirect_buffer (the late prepass's ranges) and
        // merged_indirect_buffer (both phases, for the forward pass). Unused while occlusion_active is false.
        // cull_chunks_buffer is both passes' per-chunk scratch (CullChunk in frustum_cull.slang), and is always used.
        Buffer occlusion_views_buffer{};
        Buffer occlusion_candidates_buffer{};
        Buffer cull_chunks_buffer{};
        Buffer late_indirect_buffer{};
        Buffer merged_indirect_buffer{};

        // occlusion_stat_count counters, cleared in prepare_frame. The readback copy is read when this frame slot is
        // next recorded, so the stats lag a frames-in-flight cycle.
        Buffer occlusion_stats_buffer{};
        Buffer occlusion_stats_readback_buffer{};
        bool occlusion_stats_pending = false;
        bool occlusion_stats_active = false;
        bool meshlet_occlusion_stats_active = false;

        // Decided by prepare_frame; record_frame follows it.
        bool occlusion_active = false;

        // Meshlet-level occlusion (docs/occlusion-culling.md, "Meshlet level"): one bit per meshlet of every opaque and
        // mask meshlet instance, cleared in prepare_frame. The depth prepass phases set the bits of the meshlets they
        // emit; the forward pass replays them. Grown on demand (power-of-two bytes); meshlet_visibility_words is this
        // frame's size.
        Buffer meshlet_visibility_buffer{};
        std::uint32_t meshlet_visibility_capacity_words = 0;
        std::uint32_t meshlet_visibility_words = 0;
        bool meshlet_occlusion_active = false;

        // Camera planes first, then 6 per shadow cascade. A separate buffer rather than a UBO array to get an
        // unambiguous 16-byte stride.
        Buffer frustum_planes_buffer{};

        // Punctual lights, maximum_light_count capacity. light_count (with the counts below, which keeps the struct
        // free of padding holes) slots are populated.
        Buffer lights_buffer{};

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


        // LDR composite output sampled by the editor's Viewport panel. Unused in fullscreen play.
        ImageHolder viewport_target{};


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

        // Draws and transforms are written straight into upload_buffer (at draw_upload_offset and
        // transform_upload_offset) as batches are emitted; these count them. One draw per instance, so they are equal.
        std::uint32_t draw_count = 0;
        std::uint32_t transform_count = 0;

        // One un-culled command per batch; the shadow pass draws these directly.
        std::vector<GpuDrawCommand> indirect_commands;

        // Parallel to indirect_commands.
        std::vector<GpuCullBounds> batch_bounds;

        // Populated slots of lights_buffer.
        std::uint32_t light_count = 0;

        // Number of batches, not instances. This is the drawCount for vkCmdDrawMeshTasksIndirectEXT.
        std::uint32_t indirect_command_count = 0;

        // Culling chunks over all batches (GpuCullBounds::first_chunk).
        std::uint32_t cull_chunk_count = 0;

        // Resident instanced models' LOD jobs (instance_lod.slang), their chunks over all jobs, and the camera
        // position their LODs are picked from. lod_jobs_buffer is host-written, maximum_lod_job_count of them.
        std::uint32_t lod_chunk_count = 0;
        glm::vec3 lod_camera_position{0.0F};
        std::vector<GpuLodJob> lod_jobs;
        Buffer lod_jobs_buffer{};

        // Buffers the GPU may still read until this slot's previous submission completes: freed when the slot is
        // next prepared.
        std::vector<Buffer> retired_buffers;

        // The instance ranges (first, count) emit_batch() wrote draws and transforms for, merged where adjacent.
        // Only these are copied to the device; resident groups' slots are instance_lod.slang's to fill.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> cpu_instance_ranges;

        // Batches are ordered opaque, double-sided (opaque, drawn without back-face culling), mask, blend; culling
        // preserves the order.
        std::uint32_t opaque_indirect_count = 0;
        std::uint32_t double_sided_indirect_count = 0;
        std::uint32_t mask_indirect_count = 0;
        std::uint32_t blend_indirect_count = 0;

        // Per-cascade prefix of the opaque/double-sided/mask ranges. Batches are sorted by descending
        // max_shadow_cascade, so a material can skip the far cascades.
        std::array<std::uint32_t, shadow_cascade_count> shadow_opaque_indirect_count{};
        std::array<std::uint32_t, shadow_cascade_count> shadow_double_sided_indirect_count{};
        std::array<std::uint32_t, shadow_cascade_count> shadow_mask_indirect_count{};
    };

    // The Hi-Z pyramid (docs/occlusion-culling.md): R32_SFLOAT, hiz_image_extent() of the render extent with
    // hiz_mip_count() levels. image's primary view (all levels) is what the occlusion tests read; mip_slots[i] is
    // level i alone, registered as sampled_2d and storage_2d for the build. Shared by every frame in flight: there is
    // one graphics queue and frames are submitted in order, so frame N + 1's phase 1 reads what frame N built.
    //
    // mip_slots are register_view() aliases of image's mip views, so they're released before it: declared after it
    // for destruction, and assigned first on a move.
    struct HizPyramid {
        ImageHolder image;
        std::array<ImageHolder, hiz_max_mip_count> mip_slots;
        VkExtent2D depth_extent{};
        std::uint32_t mip_count = 0;

        // Set once a build has left every level in SHADER_READ_ONLY_OPTIMAL.
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

    // A frame's extent-sized render targets, built together so initialize() and resize() share one path.
    struct OwnedFrameTargets {
        ImageHolder viewport_target{};
    };

    struct ModelSubmission {
        ModelHandle model{};
        glm::mat4 transform{1.0F};
        MaterialHandle material_override{};

        // A range of slot_override_submissions_.
        std::uint32_t slot_override_first = 0;
        std::uint32_t slot_override_count = 0;

        bool outlined = false;
    };

    // One submit_model_instances() call: a range of instance_transforms_, batched in prepare_frame() just before
    // model_submissions_[model_submission_position], so instances keep their order relative to individual
    // submissions.
    struct InstancedSubmission {
        ModelHandle model{};
        MaterialHandle material_override{};
        std::uint32_t first_transform = 0;
        std::uint32_t transform_count = 0;
        std::size_t model_submission_position = 0;

        // Non-zero: the transforms live in resident_instance_sets_ under this revision (transform_count of them) and
        // the GPU picks their LODs; nothing is in instance_transforms_.
        std::uint64_t resident_revision = 0;
    };

    // A resident instanced model's transforms (submit_model_instances()), kept on the GPU while its revision keeps
    // being submitted.
    struct ResidentInstanceSet {
        Buffer transforms{};
        std::uint32_t count = 0;
        std::uint64_t last_used_frame = 0;
    };

    std::unordered_map<std::uint64_t, ResidentInstanceSet> resident_instance_sets_;

    // Staging copies into newly created resident sets, recorded by the next prepare_frame().
    struct PendingResidentUpload {
        Buffer staging{};
        VkBuffer destination = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
    };

    std::vector<PendingResidentUpload> pending_resident_uploads_;

    // Sets replaced since the last prepare_frame(), handed to that frame's retired_buffers.
    std::vector<Buffer> retired_resident_buffers_;

    std::uint32_t resident_jobs_this_frame_ = 0;

    // The distinct (geometry, material) pairs a submesh's LODs draw with, for a resident model.
    struct ResidentLodGroups {
        std::uint32_t count = 0;
        std::uint32_t lod_groups = 0; // byte `lod`: the group LOD `lod` draws with
        std::array<std::uint32_t, lod_count> representative_lod{};
        std::array<MaterialHandle, lod_count> material{};
    };

    [[nodiscard]]
    auto resident_lod_groups(Submesh const &submesh, MaterialHandle base_material) const noexcept -> ResidentLodGroups;

    // submit_model_instances()'s resident path; false when the model doesn't qualify or the frame is full, and the
    // instances take the per-instance path.
    [[nodiscard]]
    auto submit_resident_instances(ModelHandle model, std::span<glm::mat4 const> transforms,
                                   MaterialHandle material_override, std::uint64_t revision) -> bool;

    // Resident instances submitted this frame, against maximum_resident_slots(): every LOD group reserves a slot per
    // instance in the frame's draw arrays.
    std::uint64_t resident_slots_this_frame_ = 0;

    // Resident sets unused this long are freed; frames in flight may still read them for frames_in_flight frames.
    static constexpr std::uint64_t resident_set_idle_frames = frames_in_flight + 2;

    // At most this many LodJobs (resident models x submeshes) per frame; more take the per-instance path.
    static constexpr std::uint32_t maximum_lod_job_count = 4096;

    struct BatchEntry {
        MeshHandle mesh{};
        std::uint32_t submesh_index = 0;
        MaterialHandle material{};
        std::uint32_t lod_index = 0;

        // An LOD group of a resident instanced model (instance_lod.slang): no transforms here, but resident_capacity
        // slots reserved for the GPU to fill, as group lod_group of the frame's LodJob lod_job.
        static constexpr std::uint32_t no_lod_job = ~0U;
        std::uint32_t lod_job = no_lod_job;
        std::uint32_t lod_group = 0;
        std::uint32_t resident_capacity = 0;

        // Where each instance's transform lives until emit_batch() writes it to the upload buffer: the submission
        // itself, instance_transforms_, or computed_transforms_. Pointers rather than copies, so each transform is
        // copied once per frame.
        std::vector<glm::mat4 const *> transforms;

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

        // 1 + the LodJob for a resident model's LOD group, which never shares a batch; 0 otherwise.
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
        // recorded_frame_count_ of the frame that wrote these queries.
        std::uint64_t serial{0};
        std::vector<RecordedOverlay> overlays;
    };
    // record_frame() and its passes. Each record_*_pass owns one stage and passes its output to the next through
    // its return value.

    // The frame's persistent images, resolved and validated once. The HDR and depth targets are transients of the
    // frame graph, which hands their bindless indices to the passes that use them.
    struct FrameTargets {
        Image const *shadow_atlas = nullptr;
        Image const *viewport = nullptr;

        VkExtent2D extent{};
        bool multisampled = false;
    };

    // Folds the occlusion-statistics readback recorded the last time this frame slot was used into
    // last_frame_stats_.
    auto consume_culled_readback(RendererFrame &frame) -> void;

    // Reads the cluster statistics recorded the last time this frame slot was used into last_cluster_stats_, then
    // resizes the slot's cluster lists to cluster_grid_ if it changed. The slot's fence has been waited on.
    [[nodiscard]]
    auto prepare_cluster_buffers(RendererFrame &frame) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto resolve_frame_targets(RendererFrame const &frame) const -> std::expected<FrameTargets, RendererError>;

    // The culled, compacted buffers the camera passes draw from. All three share visible_draw_buffer and
    // visible_transform_buffer; they differ in the indirect commands: phase 1's survivors (culled_indirect_buffer),
    // phase 2's (late_indirect_buffer), and both for the forward pass (merged_indirect_buffer, or
    // culled_indirect_buffer while occlusion is inactive).
    [[nodiscard]]
    auto early_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers;

    [[nodiscard]]
    auto late_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers;

    [[nodiscard]]
    auto forward_view_draws(RendererFrame const &frame) const -> render_pass::DrawBuffers;

    // Opaque/mask/blend batch counts, shared by the culled and un-culled buffers.
    [[nodiscard]]
    static auto batch_counts(RendererFrame const &frame) noexcept -> render_pass::DrawCounts;

    // Copies new resident instance sets into place and frees the ones nobody submits any more.
    auto record_resident_instance_uploads(VkCommandBuffer command_buffer, RendererFrame &frame) -> void;

    // instance_lod.slang over the frame's LodJobs, at the start of the gpu_culling pass.
    [[nodiscard]]
    auto record_instance_lods(render_pass::Context const &pass_context,
                              RendererFrame const &frame) -> std::expected<void, RendererError>;

    // cull_chunks_buffer's capacity in CullChunks; instance_lod.slang's LodChunks share it.
    VkDeviceSize cull_chunk_capacity_ = 0;

    // sizeof(LodChunk) in instance_lod.slang: a count and an offset per LOD group.
    static constexpr VkDeviceSize lod_chunk_bytes = 2 * lod_count * sizeof(std::uint32_t);

    auto record_environment_pass(render_pass::Context const &pass_context, RendererFrame const &frame) -> void;

    [[nodiscard]]
    auto record_shadow_pass(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;

    // The `only`/`early` phases also transition the forward targets into attachment layouts.
    [[nodiscard]]
    auto record_depth_prepass(render_pass::Context const &pass_context, RendererFrame const &frame,
                              FrameTargets const &targets, render_pass::DepthPrepassPhase phase)
            -> std::expected<void, RendererError>;

    // Builds hiz_ from the early prepass's depth (the MIN resolve under MSAA) for late_cs and next frame's main_cs.
    [[nodiscard]]
    auto record_hiz_build(render_pass::Context const &pass_context, FrameTargets const &targets,
                          std::uint32_t depth_texture_index) -> std::expected<void, RendererError>;

    // Phase 2 of occlusion culling: late_cs re-tests main_cs's candidates against this frame's Hi-Z.
    [[nodiscard]]
    auto record_occlusion_cull_pass(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;

    // The first graph passes of the frame (prepare_frame only uploads and validates): clears, main_cs (frustum culling
    // and phase 1 of occlusion culling) and light culling and clustering.
    auto record_occlusion_stats_clear(VkCommandBuffer command_buffer, RendererFrame const &frame) -> void;
    auto record_meshlet_visibility_clear(VkCommandBuffer command_buffer, RendererFrame const &frame) -> void;
    [[nodiscard]]
    auto record_gpu_culling(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;

    // Begins the light clustering stage's timestamps, which record_cluster_stats_readback ends.
    auto record_cluster_stats_clear(render_pass::Context const &pass_context, RendererFrame const &frame) -> void;
    [[nodiscard]]
    auto record_light_cull(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;
    [[nodiscard]]
    auto record_light_cluster(render_pass::Context const &pass_context, RendererFrame const &frame)
            -> std::expected<void, RendererError>;
    auto record_cluster_stats_readback(render_pass::Context const &pass_context, RendererFrame &frame) -> void;

    // Copies the occlusion statistics into the frame's readback buffer.
    auto record_occlusion_stats_readback(VkCommandBuffer command_buffer, RendererFrame &frame) -> void;

    // The parameters of the two GTAO passes (they read AO settings, so they are built when the frame is recorded).
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

    // What the passes hand to the ones after them: bindless indices that used to be locals of one function.
    // record_frame sets the AO index, forward the HDR one and bloom the bloom one.
    struct PassHandoff {
        std::uint32_t ao_texture_index = 0;
        render_pass::HdrTextureIndex hdr{};
        std::optional<render_pass::BloomTextureIndex> bloom;
    };


    // The end-of-frame timestamp and the flags that say this slot's queries hold results.
    auto record_frame_end(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void;

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

    // forward_pipeline_ and forward_instanced_pipeline_ with the outline mask output; see Renderer::initialize.
    PipelineNodeHandle forward_outline_pipeline_;
    PipelineNodeHandle forward_outline_instanced_pipeline_;
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

    // Declared after image_storage_, sampler_storage_ and pipeline_graph_, which it holds slots and nodes in.
    EnvironmentSystem environment_;
    std::uint64_t frame_counter_ = 0;

    BloomSettings bloom_settings_;
    OutlineSettings outline_settings_;

    // A material with MaterialCreateInfo::outlined set for each one an outlined submission has used, so those
    // submissions batch and draw with the flag. Made on first use, refreshed once a frame while in use, freed when the
    // source material is.
    struct OutlineVariant {
        MaterialHandle source{};
        MaterialHandle variant{};
        std::uint64_t refreshed_frame = 0;
    };
    std::vector<OutlineVariant> outline_variants_;

    // Whether any submission this frame was outlined, which is when the forward pass gets its second target.
    bool outline_active_ = false;

    [[nodiscard]] auto outline_variant(MaterialHandle source) -> MaterialHandle;
    auto prune_outline_variants() -> void;
    AoSettings ao_settings_;

    ImageHandle light_icon_texture_{};
    bool debug_draw_light_icons_ = false;

    // Declared before any registration it hands out, so it outlives them.
    OverlayRegistry overlays_;
    OverlayRegistration light_icon_overlay_;
    bool meshlet_culling_ = true;
    // Default off until the GPU checks in docs/occlusion-culling.md have passed.
    bool occlusion_culling_ = false;
    bool meshlet_occlusion_culling_ = false;
    bool meshlet_visibility_cap_warned_ = false;
    OcclusionTestMode occlusion_test_mode_ = OcclusionTestMode::hiz;
    bool clustered_lighting_ = true;
    bool cluster_debug_heatmap_ = false;
    ClusterGridSettings cluster_grid_{};
    ClusterStats last_cluster_stats_{};
    float light_icon_world_size_ = 0.5F;

    // Shared across frames in flight so unchanged tiles persist.
    ImageHolder shadow_atlas_{};

    // Two-phase occlusion culling. The history is the last built pyramid and the view-projection it was built with;
    // phase 1 only uses it while hiz_history_valid_, which a resize, a toggle or a frame without occlusion clears.
    HizPyramid hiz_{};
    glm::mat4 hiz_history_view_projection_{1.0F};
    bool hiz_history_valid_ = false;

    // VK_FORMAT_R32_SFLOAT supports linear filtering, which the UI's sampler needs for hiz_debug_view().
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

    // Keyed by (generation << 32 | index).
    std::unordered_map<std::uint64_t, std::filesystem::path> model_sources_;

    std::vector<Submission> submissions_;
    std::vector<ModelSubmission> model_submissions_;
    std::vector<MaterialSlotOverride> slot_override_submissions_;
    std::vector<InstancedSubmission> instanced_submissions_;
    std::vector<glm::mat4> instance_transforms_;

    // Transforms prepare_frame() has to compute (a submission's transform times a model draw's local transform), kept
    // until the batches are emitted. A deque, so BatchEntry::transforms can point into it while it grows.
    std::deque<glm::mat4> computed_transforms_;

    // Model submissions this frame, individual and instanced: what maximum_submission_count_ bounds.
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

    // The frame graph: rebuilt every frame, recompiled only when its declaration changes (plan_cache_). frame_plan_ and
    // submit_batches_ outlive record_frame() so the caller can submit them (the batches' waits point into the plan).
    frame_graph::FrameGraph frame_graph_;
    frame_graph::PlanCache plan_cache_;
    frame_graph::CompiledGraph const *frame_plan_ = nullptr; // into plan_cache_, valid until the next record_frame
    std::vector<SubmitBatch> submit_batches_;
    frame_graph::PassProfiler pass_profiler_;

    // Backs the graph's transient images, per frame slot (record_frame()). Aliasing can be switched off for debugging.
    // (--frame-graph-alias=on|off)
    frame_graph::TransientAllocator transient_allocator_;
    bool transient_aliasing_ = true;
    std::uint32_t async_candidates_ = 0;
    std::uint64_t logged_plan_misses_ = 0;
    bool dump_frame_graph_ = false;

    struct FramePipelineQuery {
        VkQueryPool query_pool{VK_NULL_HANDLE};
        bool has_results{false};
    };
    std::vector<FramePipelineQuery> pipeline_stat_queries_;
    PipelineStats last_frame_pipeline_stats_{};

    std::unique_ptr<ScreenshotCapture> screenshot_;

    bool initialized_ = false;
};
