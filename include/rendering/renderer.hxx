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
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "assets/geometry_arena.hxx"
#include "assets/load_model.hxx"
#include "assets/material_storage.hxx" // TextureHandle, MaterialCreateInfo now live here
#include "assets/mesh_create_info.hxx"
#include "assets/mesh_sink.hxx"
#include "assets/mesh_storage.hxx"
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
#include "rendering/forward_target.hxx"
#include "rendering/pipeline_graph_repository.hxx"
#include "rendering/render_stage.hxx"
#include "rendering/screenshot.hxx"
#include "rendering/script_storage.hxx"
#include "rendering/shadow_cascades.hxx"

struct BloomSettings {
    bool enabled = true;
    float threshold = 1.0F;
    float knee = 0.5F;
    float filter_radius = 0.005F;
    float intensity = 0.04F;
};

// GTAO (Ground-Truth Ambient Occlusion, Jimenez et al. 2016): a screen-space
// horizon-based AO term computed from the depth buffer alone (this renderer
// has no normal G-buffer -- view-space normals are reconstructed from
// neighbouring depth samples inside the GTAO pass itself), denoised with a
// depth-aware spatial blur, then multiplied into the ambient term in
// forward_geom.slang alongside each material's baked occlusion texture.
struct AoSettings {
    bool enabled = true;

    // View-space sampling radius, in world units (view space is a rigid
    // transform of world space, so the units match).
    float radius = 0.5F;

    // Fraction of `radius` over which a sample's contribution fades to zero
    // rather than being cut off hard at the radius boundary.
    float falloff_range = 0.615F;

    std::uint32_t slice_count = 2;
    std::uint32_t step_count = 6;

    // Blend factor written into UBO::ao_intensity -- see that field's
    // comment for what it does.
    float intensity = 1.0F;

    // Bilateral denoise pass: larger values tolerate bigger view-space depth
    // differences between neighbours before down-weighting them, trading
    // edge sharpness for noise reduction.
    float denoise_depth_sigma = 40.0F;
};


// Exponential distance fog, applied in forward_geom.slang after shading.
// Disabled by default -- opt in via Application::on_ui() or by calling
// Renderer::set_fog_settings() directly.
struct FogSettings {
    bool enabled = false;
    glm::vec3 colour{0.5F};
    float extinction = 0.003F;
    float inscattering = 1.0F;
};

struct StageTimings {
    std::array<float, stage_count> milliseconds{};
    bool valid = false;
};

struct FrameStats {
    std::uint32_t submitted_triangle_count = 0;
    std::uint32_t submitted_instance_count = 0;

    std::uint32_t indirect_command_count = 0;
    std::uint32_t opaque_indirect_count = 0;
    std::uint32_t mask_indirect_count = 0;
    std::uint32_t blend_indirect_count = 0;

    std::uint32_t model_submission_count = 0;
    std::uint32_t mesh_submission_count = 0;

    std::uint32_t point_light_count = 0;
    std::uint32_t spot_light_count = 0;
};

inline constexpr std::uint32_t pipeline_stat_count = 4;

struct PipelineStats {
    std::uint64_t assembled_primitive_count = 0;
    std::uint64_t clipped_primitive_count = 0;
    std::uint64_t assembled_vertex_count = 0;
    std::uint64_t fragment_shader_invocation_count = 0;

    bool valid = false;
};

struct SwapchainImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
};


struct UBO {
    glm::mat4 view_projection;
    glm::mat4 view;
    glm::mat4 projection;

    // Inverse of `projection` -- see the mirrored field's comment in
    // assets/shaders/scene_types.slang.
    glm::mat4 inverse_projection;

    glm::vec3 camera_position;
    glm::vec3 fog_colour;
    float fog_extinction = 0.0F; // 0 = disabled; see Renderer::set_fog_settings
    float fog_inscattering = 1.0F;

    // ---- PSSM cascades ----
    std::array<glm::mat4, shadow_cascade_count> cascade_view_projection{};
    glm::vec4 cascade_split_far{}; // view-space far distance per cascade
    glm::vec4 cascade_texel_world{}; // world-space size of one shadow texel per cascade
    glm::vec4 cascade_depth_scale{}; // 1 / (z_far - z_near) per cascade

    // Atlas tiles are packed side-by-side at variable width (see
    // shadow_cascade_resolutions) rather than a uniform grid, so the shader
    // can't derive a cascade's placement from cascade_count alone -- these
    // remap a cascade-local [0,1] tile UV into the shared atlas texture.
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

    float time = 0.0F; // seconds since startup -- drives wind sway in vertex shaders

    // Flat multiplier on the (already-albedo-scaled) ambient term. A real
    // Lambert BRDF (albedo/pi, see pbr.slang) is noticeably darker than the
    // old ambient = albedo term this replaced, so this exists purely to let
    // the scene be re-tuned back to a sane brightness without an IBL/skybox.
    float ambient_intensity = 0.15F;

    // Blend factor between "no screen-space AO" (1.0 everywhere) and the
    // GTAO pass's output, applied on top of the material's baked occlusion
    // texture in forward_geom.slang. 0 disables the screen-space term
    // entirely (pure baked AO); 1 applies it at full strength.
    float ao_intensity = 1.0F;
};

static_assert(sizeof(UBO) == 716, "UBO layout changed -- update the mirror in assets/shaders/scene_types.slang");
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

    Renderer(Renderer const &) = delete;
    auto operator=(Renderer const &) -> Renderer & = delete;

    Renderer(Renderer &&) = delete;
    auto operator=(Renderer &&) -> Renderer & = delete;

    [[nodiscard]]
    auto initialize(RendererCreateInfo const &create_info) -> std::expected<void, RendererError>;

    auto destroy() noexcept -> void;

    [[nodiscard]]
    auto load_model(std::filesystem::path const &path) -> std::expected<ModelHandle, RendererError>;

    // Reserves a model handle as a copy of `fallback`'s data -- usable
    // immediately (renders and reports bounds as the fallback) -- to be
    // installed into later via finish_model_load(). See ModelStorage::create_pending_model.
    [[nodiscard]]
    auto create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, RendererError> override;

    // Completes an async load started with load_model_cpu_async(): records
    // the GPU upload (meshes, materials, any embedded textures) into
    // `command_buffer` -- which must be a currently-recording command
    // buffer whose submission this caller will wait on via the normal
    // frames-in-flight fence discipline, not a one-time_submit buffer --
    // then installs the result into `pending` in place. Must run on the
    // render thread. On failure `pending` is left showing its original
    // fallback content rather than being torn down.
    [[nodiscard]]
    auto finish_model_load(ModelHandle pending, ModelCpuData const &cpu_data, VkCommandBuffer command_buffer)
            -> std::expected<void, RendererError> override;

    // Uploads already-CPU-side geometry (e.g. procedurally generated engine
    // primitives — see primitive_meshes.hxx / engine_models.hxx) through the
    // same GPU upload path as load_model, without touching disk or a glTF
    // parser. Skips the path-based model cache load_model uses.
    [[nodiscard]]
    auto create_model_from_cpu_data(ModelCpuData const &cpu_data) -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto create_model(Model const &model, MaterialHandle fallback_material)
            -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto create_model(Model const &model) -> std::expected<ModelHandle, RendererError>;

    // material_override, when valid(), replaces every submesh material for
    // this submission -- e.g. force an entity to render solid red regardless
    // of what its model's submeshes normally use.
    [[nodiscard]]
    auto submit_model(ModelHandle model, glm::mat4 const &transform, MaterialHandle material_override = {})
            -> std::expected<void, RendererError>;
    [[nodiscard]]
    auto submit_model(ModelHandle model, glm::mat4 &&, MaterialHandle material_override = {})
            -> std::expected<void, RendererError>;

    // Model-space AABB across every vertex of the model, as loaded --
    // callers can use this to size collision volumes/gizmos to the actual
    // geometry instead of guessing.
    [[nodiscard]]
    auto model_bounds(ModelHandle model) const -> std::optional<std::pair<glm::vec3, glm::vec3>>;
    [[nodiscard]]
    auto model_lights(ModelHandle model) const -> std::span<ModelCpuLight const>;

    [[nodiscard]]
    auto create_material(MaterialCreateInfo const &create_info) -> std::expected<MaterialHandle, RendererError>;

    [[nodiscard]]
    auto update_material(MaterialHandle handle, MaterialCreateInfo const &create_info)
            -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto destroy_material(MaterialHandle handle) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto create_mesh(MeshCreateInfo const &create_info) -> std::expected<MeshHandle, RendererError> override;

    [[nodiscard]]
    auto destroy_mesh(MeshHandle handle) -> std::expected<void, RendererError>;

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
        float time = 0.0F; // seconds since startup -- forwarded to UBO.time for wind sway
    };

    struct DirectionalLight {
        glm::vec3 direction{0.4F, 0.8F, 0.25F}; // normalized, points from surface to light
        glm::vec3 colour{1.0F, 0.97F, 0.92F};
        float intensity = 3.0F;
    };

    // Punctual lights -- submitted per frame like submit_model/submit_mesh,
    // not sticky settings. Unlike DirectionalLight, these never cast
    // shadows (see the plan this shipped under: BRDF + point/spot lights,
    // no shadows yet).
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

    // Multiplies the ambient term (albedo * AO), standing in for IBL/skybox
    // ambient lighting until one exists.
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

        // A cascade may only adopt a new fitted matrix when it is actually
        // redrawn. These periods are minimum update intervals: the near
        // cascade refreshes every frame, while farther cascades can reuse
        // their previous atlas tiles for 2/4/8 frames.
        std::array<std::uint32_t, shadow_cascade_count> cache_update_periods{1U, 2U, 4U, 8U};
        bool cache_enabled = true;
        bool debug_cascade_tint = false;
    };

    auto set_directional_light(DirectionalLight const &light) noexcept -> void { light_ = light; }
    [[nodiscard]] auto directional_light() const noexcept -> DirectionalLight const & { return light_; }

    auto set_shadow_settings(ShadowSettings const &settings) noexcept -> void { shadow_settings_ = settings; }
    [[nodiscard]] auto shadow_settings() const noexcept -> ShadowSettings const & { return shadow_settings_; }

    // Call this after an add/remove/material change that affects shadow
    // casters. It forces every cached cascade to be refreshed immediately.
    auto mark_shadow_casters_dirty() noexcept -> void;

    // Call once per frame while any caster outside the always-fresh near
    // cascade is moving. Far cascades still respect cache_update_periods, so
    // this does not turn caching off; it merely guarantees bounded staleness.
    auto mark_dynamic_shadow_casters_dirty() noexcept -> void { dynamic_shadow_casters_dirty_ = true; }

    [[nodiscard]]
    auto prepare_frame(VkCommandBuffer command_buffer, const CameraMatrices &, std::uint32_t frame_index)
            -> std::expected<void, RendererError>;

    template<typename OverlayPolicy>
    [[nodiscard]] auto record_frame(VkCommandBuffer command_buffer, SwapchainImage const &swapchain_image,
                                    std::uint32_t frame_index, Application const &app, glm::mat4 const &vp)
            -> std::expected<void, RendererError>;

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

    void queue_render_thread_event(std::move_only_function<void()> &&);
    void drain_event_queue();

    [[nodiscard]] auto context() noexcept -> VulkanContext & { return context_; }
    [[nodiscard]] auto depth_format() const noexcept { return frames_[0].forward_target.depth_format(); }
    [[nodiscard]] auto hdr_format() const noexcept { return frames_[0].forward_target.hdr_format(); }
    [[nodiscard]] auto samples() const noexcept { return frames_[0].forward_target.samples(); }

    [[nodiscard]] auto image_storage() noexcept -> ImageStorage & { return image_storage_; }
    [[nodiscard]] auto sampler_storage() noexcept -> SamplerStorage & override { return sampler_storage_; }
    // Shared across editor_scene/runtime_scene -- Scene::get_scripts() forwards here so a
    // ScriptHandle allocated while populating editor_scene still resolves after
    // Application::play() clones entities into the fresh runtime_scene (see
    // Scene::Scene(Renderer&)).
    [[nodiscard]] auto script_storage() noexcept -> ScriptStorage & { return script_storage_; }
    [[nodiscard]] auto script_storage() const noexcept -> ScriptStorage const & { return script_storage_; }
    [[nodiscard]] auto texture_streamer() noexcept -> TextureStreamer & { return texture_streamer_; }
    [[nodiscard]] auto model_streamer() noexcept -> ModelStreamer & { return model_streamer_; }
    [[nodiscard]] auto resource_table() noexcept -> GpuResourceTable & { return gpu_resource_table_; }

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

    auto request_screenshot() noexcept -> void { screenshot_.request(); }
    auto mark_lights_dirty() -> void { lights_dirty_mask_ = frames_.empty() ? 0U : ((1U << frames_.size()) - 1U); }
    auto wait_idle() -> std::expected<void, RendererError>;

    static auto compiler() noexcept -> renderer::SlangCompiler &;

private:
    struct Submission {
        MeshHandle mesh{};
        glm::mat4 transform{1.0F};
        MaterialHandle material_override{};
    };

    struct alignas(16) GpuDraw {
        VkDeviceAddress vertex_address = 0;

        std::uint32_t material_index = 0;
        std::uint32_t transform_index = 0;
    };

    static_assert(std::is_trivially_copyable_v<GpuDraw>);

    static_assert(sizeof(GpuDraw) == 16);

    // Local-space AABB for one batch (mesh+submesh+material), used by the
    // GPU frustum-culling compute pass. Mirrors GpuCullBounds in
    // assets/shaders/frustum_cull.slang. wind_padding conservatively grows
    // the world-space AABB (in transform_aabb) along X/Z to cover the
    // maximum sway wind_offset() (wind.slang) can apply to this batch's
    // material -- without it, wind-swaying foliage can visibly pop as it
    // sways past its static bounds at the frustum edge.
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

    // One punctual light as uploaded to lights_buffer. Mirrors GpuLight in
    // assets/shaders/scene_types.slang. spot_scale/spot_offset are the
    // glTF-style precomputed cone-falloff terms (KHR_lights_punctual),
    // computed once here rather than per-pixel in the shader.
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

    static constexpr std::uint32_t maximum_light_count = 256;

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

        // GPU frustum-culling inputs/outputs. One compute workgroup handles
        // one batch (see mainCs in assets/shaders/frustum_cull.slang):
        // batch_bounds gives that batch's local-space AABB + wind padding;
        // indirect_buffer is read as the source [firstInstance,
        // firstInstance + instanceCount) range for that batch;
        // culled_indirect_buffer receives the same command with a
        // recomputed instanceCount. visible_draw_buffer/
        // visible_transform_buffer are the compaction destination that the
        // main-view passes (depth prepass, forward) read instead of
        // draw_buffer/transform_buffer. The shadow pass is untouched -- it
        // keeps reading draw_buffer/transform_buffer/indirect_buffer
        // directly (see prepare_frame).
        Buffer batch_bounds_buffer{};
        Buffer culled_indirect_buffer{};
        Buffer visible_draw_buffer{};
        Buffer visible_transform_buffer{};

        // 6 world-space frustum planes (vec4 each), written fresh every
        // frame in prepare_frame. Deliberately its own tiny buffer rather
        // than a field on the shared UBO -- a Ptr<float4> array has an
        // unambiguous 16-byte stride under every struct-layout convention,
        // whereas a trailing array field on UBO would depend on exactly
        // which packing rule the Slang compiler applies to that struct,
        // which is not worth staking correctness on.
        Buffer frustum_planes_buffer{};

        // Punctual (point/spot) lights this frame, fixed capacity
        // (maximum_light_count), host-written every frame in prepare_frame
        // just like frustum_planes_buffer above -- same reasoning applies.
        // light_count is how many of lights_buffer's slots are populated;
        // read back in record_frame when filling push constants.
        Buffer lights_buffer{};
        std::uint32_t light_count = 0;

        ForwardTarget forward_target{};

        struct BloomTarget {
            ImageHandle image;
            std::array<ImageHandle, 4> mip_slots;
        };
        BloomTarget bloom_target{};

        // GTAO output at full render resolution: `raw` is the horizon-search
        // pass's noisy output, `denoised` is the depth-aware blur's output
        // and the texture actually sampled by forward_geom.slang. Both are
        // per-frame-in-flight (like bloom_target) rather than shared, since
        // they're written and read entirely within one frame's command
        // buffer with no cross-frame history.
        struct AoTarget {
            ImageHandle raw;
            ImageHandle denoised;
        };
        AoTarget ao_target{};

        // Bit i means cascade i must be cleared and redrawn into the shared,
        // persistent atlas this frame. The atlas itself intentionally does
        // not live in RendererFrame: temporal reuse must survive frame-index
        // rotation.
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

        std::vector<VkDrawIndexedIndirectCommand> indirect_commands;

        // Per-batch local-space AABB + wind padding, parallel to
        // indirect_commands. culled_indirect_buffer is written entirely by
        // the compute pass (mainCs copies each batch's source command and
        // overwrites instanceCount) -- no CPU-side seed vector is needed.
        std::vector<GpuCullBounds> batch_bounds;

        // Number of VkDrawIndexedIndirectCommand entries in indirect_commands
        // (one per unique (mesh, submesh) batch this frame) — NOT the number
        // of GpuDraw / instance entries in `draws`. This is the value that
        // must be passed as drawCount to vkCmdDrawIndexedIndirect.
        std::uint32_t indirect_command_count = 0;

        // indirect_commands (and therefore culled_indirect_buffer, since GPU
        // culling is order-preserving) is partitioned into three contiguous
        // ranges, in this order: [0, opaque_indirect_count), then
        // [opaque_indirect_count, opaque_indirect_count + mask_indirect_count),
        // then the remainder is blend. See prepare_frame's batch partition.
        std::uint32_t opaque_indirect_count = 0;
        std::uint32_t mask_indirect_count = 0;
        std::uint32_t blend_indirect_count = 0;

        // Per-cascade drawCount for the shadow pass -- a prefix of
        // [0, opaque_indirect_count) / [0, mask_indirect_count) respectively,
        // since prepare_frame sorts opaque/mask batches by descending
        // material max_shadow_cascade before emitting them. Lets a batch
        // (e.g. grass) opt out of the farther cascades instead of being
        // rasterized into all shadow_cascade_count of them unconditionally.
        std::array<std::uint32_t, shadow_cascade_count> shadow_opaque_indirect_count{};
        std::array<std::uint32_t, shadow_cascade_count> shadow_mask_indirect_count{};
    };

    struct ModelSubmission {
        ModelHandle model{};
        glm::mat4 transform{1.0F};
        MaterialHandle material_override{};
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

    // Shared by create_model() and finish_model_load(): creates every mesh
    // `model` references, flattens its scene graph into a draw list, and
    // hands the resulting ModelSlotData to `install` -- create_model()
    // passes model_storage_.create_model (a fresh handle), finish_model_load()
    // passes model_storage_.upgrade_pending_model (installs into an
    // already-issued pending handle in place). Rolls back every mesh it
    // created if `install` itself fails (e.g. capacity_exceeded).
    [[nodiscard]]
    auto
    create_model_common(Model const &model, MaterialHandle fallback_material,
                        std::move_only_function<std::expected<ModelHandle, ModelStorageError>(ModelSlotData)> install)
            -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto upload_frame_data(VkCommandBuffer command_buffer, RendererFrame &frame) -> std::expected<void, RendererError>;

    auto clear_submissions() noexcept -> void;


    VulkanContext &context_;

    VkFormat hdr_format_ = VK_FORMAT_UNDEFINED;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkExtent2D extent_{};

    GeometryArena geometry_arena_{};
    MaterialStorage material_storage_{};
    ImageStorage image_storage_{};
    SamplerStorage sampler_storage_{};
    TextureStreamer texture_streamer_{};
    ModelStreamer model_streamer_{};
    GpuResourceTable gpu_resource_table_{};

    std::vector<Buffer> ubos_;

    PipelineGraphRepository pipeline_graph_;
    PipelineNodeHandle shadow_pipeline_;
    PipelineNodeHandle shadow_mask_pipeline_;
    PipelineNodeHandle depth_prepass_pipeline_;
    PipelineNodeHandle depth_prepass_mask_pipeline_;
    PipelineNodeHandle forward_pipeline_;
    PipelineNodeHandle forward_blend_pipeline_;
    PipelineNodeHandle composite_pipeline_;
    PipelineNodeHandle frustum_cull_pipeline_;
    PipelineNodeHandle light_icon_pipeline_;
    PipelineNodeHandle bloom_downsample_pipeline_;
    PipelineNodeHandle bloom_upsample_pipeline_;
    PipelineNodeHandle gtao_pipeline_;
    PipelineNodeHandle gtao_denoise_pipeline_;
    ShaderChangeQueue shader_change_queue_;

    BloomSettings bloom_settings_;
    AoSettings ao_settings_;

    ImageHandle light_icon_texture_{};
    bool debug_draw_light_icons_ = false;
    float light_icon_world_size_ = 0.5F;

    // One atlas for the renderer, not one atlas per frame in flight. Queue
    // ordering plus the shadow-pass barriers serialize writes/sampling while
    // letting unchanged tiles persist across frames.
    ImageHandle shadow_atlas_{};
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

    std::vector<Submission> submissions_;
    std::vector<ModelSubmission> model_submissions_;

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

    struct FrameTimestamps {
        VkQueryPool query_pool{VK_NULL_HANDLE};
        bool has_results{false};
    };
    std::vector<FrameTimestamps> timestamp_queries_;
    float timestamp_period_{1.0F};

    struct FramePipelineQuery {
        VkQueryPool query_pool{VK_NULL_HANDLE};
        bool has_results{false};
    };
    std::vector<FramePipelineQuery> pipeline_stat_queries_;
    PipelineStats last_frame_pipeline_stats_{};

    ScreenshotCapture screenshot_;

    bool initialized_ = false;
};
