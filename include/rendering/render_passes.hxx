#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <type_traits>

#include <volk.h>

#include "core/config.hxx"
#include "core/renderer_error.hxx"
#include "gpu/buffer.hxx"
#include "gpu/gpu_resource_table.hxx"
#include "gpu/image_storage.hxx"
#include "rendering/overlay.hxx"
#include "rendering/pipeline_graph_repository.hxx"
#include "rendering/shadow_cascades.hxx"

namespace render_pass {

    struct HdrTextureIndex {
        std::uint32_t index = 0;
    };

    struct BloomTextureIndex {
        std::uint32_t index = 0;
    };

    struct AoTextureIndex {
        std::uint32_t index = 0;
    };

    struct Context {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        std::uint32_t frame_index = 0;
        PipelineGraphRepository &pipeline_graph;
        GpuResourceTable &resource_table;
        VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
    };

    // `indirect` holds one GpuDrawCommand per batch, ordered opaque | mask | blend like DrawCounts. `index_buffer`
    // is read by the instanced commands.
    struct DrawBuffers {
        Buffer const &draws;
        Buffer const &transforms;
        Buffer const &indirect;
        VkBuffer index_buffer = VK_NULL_HANDLE;
    };

    struct DrawCounts {
        std::uint32_t opaque = 0;
        std::uint32_t mask = 0;
        std::uint32_t blend = 0;
    };

    struct ShadowPassInfo {
        Image const &shadow_atlas;
        DrawBuffers draws;
        DrawCounts counts;
        std::array<std::uint32_t, shadow_cascade_count> const &opaque_cascade_counts;
        std::array<std::uint32_t, shadow_cascade_count> const &mask_cascade_counts;

        // Only tiles in update_mask are cleared and redrawn; the rest persist across frames.
        std::uint32_t update_mask = (1U << shadow_cascade_count) - 1U;
        bool preserve_contents = false;

        bool meshlet_culling = true;

        // Cascade 0's first plane; the task shader offsets by cascade_index * 6.
        VkDeviceAddress cascade_cull_planes_address = 0;
        VkDeviceAddress materials_address = 0;
        VkDeviceAddress ubo_address = 0;
        VkDeviceAddress lights_address = 0;

        // Task/mesh pipelines and their instanced vertex-shader variants.
        PipelineNodeHandle opaque_pipeline{};
        PipelineNodeHandle mask_pipeline{};
        PipelineNodeHandle opaque_instanced_pipeline{};
        PipelineNodeHandle mask_instanced_pipeline{};

        float depth_bias_constant = -1.0F;
        float depth_bias_slope = -2.5F;
    };

    struct ForwardTargets {
        Image const &hdr;
        Image const &depth;
        Image const *resolved_hdr = nullptr;
        Image const *resolved_depth = nullptr;
    };

    // Two-phase occlusion culling splits the prepass (docs/occlusion-culling.md): `early` clears and draws the
    // phase-1 instances, `late` loads that depth and adds the phase-2 ones. `only` is the single pass without it.
    enum class DepthPrepassPhase : std::uint8_t {
        only,
        early,
        late,
    };

    // Meshlet-level occlusion cull bits for DepthPrepassInfo/ForwardGeometryInfo::extra_cull_flags
    // (docs/occlusion-culling.md, "Meshlet level"). Mirrors cull_*_bit in scene_types.slang.
    inline constexpr std::uint32_t cull_occlusion = 4U;
    inline constexpr std::uint32_t cull_skip_recorded = 8U;
    inline constexpr std::uint32_t cull_record = 16U;
    inline constexpr std::uint32_t cull_replay = 32U;
    inline constexpr std::uint32_t cull_stats = 64U;

    struct DepthPrepassInfo {
        Image const &depth;
        Image const *resolved_depth = nullptr;
        VkExtent2D extent{};
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

        // only/early time RenderStage::DepthPrepass and clear; late times RenderStage::DepthPrepassLate and loads.
        DepthPrepassPhase phase = DepthPrepassPhase::only;

        // How `depth` resolves into `resolved_depth` when multisampled. MIN keeps each pixel's farthest sample
        // (reverse-Z), which the Hi-Z needs to stay conservative; SAMPLE_ZERO is what GTAO and the rest expect.
        VkResolveModeFlagBits depth_resolve_mode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;

        DrawBuffers draws;
        DrawCounts counts;

        // The camera's frustum planes, for meshlet culling.
        VkDeviceAddress cull_planes_address = 0;
        VkDeviceAddress materials_address = 0;
        VkDeviceAddress ubo_address = 0;
        VkDeviceAddress lights_address = 0;

        // Meshlet-level occlusion: the OcclusionView (PC::occlusion) and the cull_* bits OR-ed into the opaque and
        // mask draws when meshlet_culling is set. 0 / 0 when it is inactive.
        VkDeviceAddress occlusion_view_address = 0;
        std::uint32_t extra_cull_flags = 0;

        // Task/mesh pipelines and their instanced vertex-shader variants.
        PipelineNodeHandle opaque_pipeline{};
        PipelineNodeHandle mask_pipeline{};
        PipelineNodeHandle opaque_instanced_pipeline{};
        PipelineNodeHandle mask_instanced_pipeline{};

        // Must match ForwardGeometryInfo::meshlet_culling, since forward depth-tests EQUAL.
        bool meshlet_culling = true;
    };

    // GTAO from depth alone, then a depth-aware blur, as two compute dispatches between the prepass and forward.
    // `depth` is the single-sample depth in DEPTH_ATTACHMENT_OPTIMAL and is left in that layout.
    struct AmbientOcclusionInfo {
        bool enabled = true;

        Image const &depth;
        Image const &raw_ao;
        Image const &denoised_ao;

        VkExtent2D extent{};

        std::uint32_t depth_texture_index = 0;
        std::uint32_t raw_ao_texture_index = 0;
        std::uint32_t denoised_ao_texture_index = 0;
        std::uint32_t point_sampler_index = 0;

        VkDeviceAddress ubo_address = 0;

        PipelineNodeHandle gtao_pipeline{};
        PipelineNodeHandle denoise_pipeline{};

        float radius_view = 0.5F;
        float falloff_range = 0.615F;
        std::uint32_t slice_count = 2;
        std::uint32_t step_count = 6;
        float denoise_depth_sigma = 40.0F;
    };

    struct ForwardGeometryInfo {
        Image const &hdr;
        Image const &depth;
        Image const *resolved_hdr = nullptr;
        HdrTextureIndex output_hdr{};

        VkExtent2D extent{};
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

        DrawBuffers draws;
        DrawCounts counts;

        // The camera's frustum planes, for meshlet culling.
        VkDeviceAddress cull_planes_address = 0;
        VkDeviceAddress materials_address = 0;
        VkDeviceAddress ubo_address = 0;
        VkDeviceAddress lights_address = 0;
        std::uint32_t light_count = 0;

        // Per-cluster light counts and lists; only read when the UBO enables clustered lighting.
        VkDeviceAddress cluster_lights_address = 0;

        // Meshlet-level occlusion: the OcclusionView (PC::occlusion) and the cull_* bits OR-ed into the opaque and
        // mask draws when meshlet_culling is set (cull_replay: forward draws exactly what the prepass phases
        // recorded). Blend draws never take them. 0 / 0 when it is inactive.
        VkDeviceAddress occlusion_view_address = 0;
        std::uint32_t extra_cull_flags = 0;

        VkQueryPool pipeline_statistics_query_pool = VK_NULL_HANDLE;

        bool meshlet_culling = true;

        // Task/mesh pipelines and their instanced vertex-shader variants.
        PipelineNodeHandle opaque_pipeline{};
        PipelineNodeHandle blend_pipeline{};
        PipelineNodeHandle opaque_instanced_pipeline{};
        PipelineNodeHandle blend_instanced_pipeline{};

        // The background, drawn between the mask and blend draws (docs/ibl-and-skybox.md).
        PipelineNodeHandle skybox_pipeline{};
        bool draw_skybox = false;

        // Denoised GTAO, or white when AO is disabled.
        std::uint32_t ao_texture_index = 0;
        std::uint32_t ao_sampler_index = 0;
    };

    inline constexpr std::uint32_t bloom_mip_count = 4;

    // Bloom mip chain on `target` (mip 0 is half the HDR resolution): bloom_mip_count downsamples, then
    // bloom_mip_count - 1 upsamples accumulating each level into the one above. mip_texture_indices[i] is a
    // single-mip view registered as sampled_2d and storage_2d. Every level ends in SHADER_READ_ONLY_OPTIMAL.
    struct BloomPassInfo {
        bool enabled = true;

        HdrTextureIndex input_hdr{};
        Image const *target = nullptr;
        std::array<std::uint32_t, bloom_mip_count> mip_texture_indices{};
        VkExtent2D input_extent{};

        PipelineNodeHandle downsample_pipeline{};
        PipelineNodeHandle upsample_pipeline{};

        std::uint32_t linear_sampler_index = 0;

        float threshold = 1.0F;
        float knee = 0.5F;

        // Tent radius in texels of the lower level.
        float filter_radius = 1.0F;
    };

    struct CompositePassInfo {
        VkImage swapchain_image = VK_NULL_HANDLE;
        VkImageView swapchain_view = VK_NULL_HANDLE;
        VkExtent2D extent{};

        HdrTextureIndex hdr{};
        std::optional<BloomTextureIndex> bloom;
        std::uint32_t bloom_fallback_texture_index = 0;
        std::uint32_t linear_sampler_index = 0;

        PipelineNodeHandle pipeline{};

        float exposure = 1.0F;
        float bloom_intensity = 0.0F;
    };

    // Non-owning, allocation-free callback. The callable must outlive the render-pass call.
    struct Callback {
        void *userdata = nullptr;
        void (*invoke)(void *) = nullptr;

        template<typename Function>
        [[nodiscard]] static auto bind(Function &function) noexcept -> Callback {
            static_assert(std::is_invocable_v<Function &>);

            return Callback{
                    .userdata = &function,
                    .invoke = [](void *userdata) { (*static_cast<Function *>(userdata))(); },
            };
        }

        auto operator()() const -> void {
            if (invoke != nullptr) {
                invoke(userdata);
            }
        }
    };

    auto prepare_forward_targets(Context const &context, ForwardTargets const &targets) noexcept -> void;

    auto shadow(Context const &context, ShadowPassInfo const &info) -> std::expected<void, RendererError>;

    auto depth_prepass(Context const &context, DepthPrepassInfo const &info) -> std::expected<void, RendererError>;

    // Builds the Hi-Z pyramid between the early and late prepasses (docs/occlusion-culling.md), as one compute
    // dispatch per level (hiz_build.slang), each reading the level below through mip_texture_indices[level - 1] (or the
    // depth for level 0) and writing mip_texture_indices[level].
    //
    // `source_depth` is the single-sample depth the early prepass wrote: the MIN resolve under MSAA, otherwise the
    // attachment itself. It goes DEPTH_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL -> DEPTH_ATTACHMENT_OPTIMAL, so
    // the late prepass can load or re-resolve it. `multisampled_depth`, when set, is the MSAA attachment, which the
    // late prepass loads. Every pyramid level ends in SHADER_READ_ONLY_OPTIMAL, visible to compute, task and fragment
    // shaders (the occlusion tests and the debug view); the previous contents are discarded.
    struct HizBuildInfo {
        Image const &source_depth;
        std::uint32_t source_texture_index = 0;
        VkExtent2D depth_extent{};
        Image const *multisampled_depth = nullptr;

        Image const &hiz;
        std::span<std::uint32_t const> mip_texture_indices;

        PipelineNodeHandle pipeline{};
    };

    auto build_hiz(Context const &context, HizBuildInfo const &info) -> std::expected<void, RendererError>;

    auto ambient_occlusion(Context const &context, AmbientOcclusionInfo const &info)
            -> std::expected<std::optional<AoTextureIndex>, RendererError>;

    // scene_overlays runs inside the forward rendering scope after the scene draws.
    auto forward_geometry(Context const &context, ForwardGeometryInfo const &info, Callback scene_overlays)
            -> std::expected<HdrTextureIndex, RendererError>;

    // The dynamic state every overlay starts from, set before each overlay's record():
    //
    //   viewport/scissor   the full scope extent. scene uses the forward pass's flipped-Y, reverse-Z viewport
    //                      (y = height, height = -height, depth 1..0); ui uses an unflipped 0..1 viewport.
    //   rasterisation      fill, cull none, counter-clockwise front, no depth bias/clamp, no discard,
    //                      samples = scope.samples with a full mask, no alpha-to-coverage.
    //   input assembly     triangle list, no primitive restart, no vertex bindings/attributes.
    //   depth/stencil      GREATER_OR_EQUAL test when the scope has depth, writes off, stencil off.
    //   colour             one attachment, blending off, RGBA write mask, logic op off.
    //
    // Descriptor sets and push constants are not part of the baseline.
    auto set_overlay_baseline_state(VkCommandBuffer command_buffer, OverlayStage stage,
                                    OverlayScope const &scope) noexcept -> void;

    struct LightIconsInfo {
        VkDeviceAddress lights_address = 0;
        VkDeviceAddress ubo_address = 0;
        std::uint32_t light_count = 0;

        PipelineNodeHandle pipeline{};
        std::uint32_t icon_texture_index = 0;
        std::uint32_t sampler_index = 0;
        float icon_world_size = 0.5F;
    };

    // Billboarded icons at every punctual light, for a scene overlay's record().
    auto light_icons(Context const &context, LightIconsInfo const &info, OverlayScope const &scope) noexcept -> void;

    auto bloom(Context const &context, BloomPassInfo const &info)
            -> std::expected<std::optional<BloomTextureIndex>, RendererError>;

    auto composite(Context const &context, CompositePassInfo const &info, Callback ui_overlay)
            -> std::expected<void, RendererError>;

    // Clears `target_view` and runs `ui_overlay` on it. Used for the swapchain in the editor, where the scene was
    // already composited into the viewport texture.
    struct UiOnlyPassInfo {
        VkImage target_image = VK_NULL_HANDLE;
        VkImageView target_view = VK_NULL_HANDLE;
        VkExtent2D extent{};
        std::array<float, 4> clear_colour{0.0F, 0.0F, 0.0F, 1.0F};
    };

    auto ui_only(Context const &context, UiOnlyPassInfo const &info, Callback ui_overlay) noexcept -> void;

    // COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL.
    auto transition_to_shader_read(VkCommandBuffer command_buffer, Image const &image) noexcept -> void;

    auto present_swapchain(VkCommandBuffer command_buffer, VkImage image) noexcept -> void;

} // namespace render_pass
