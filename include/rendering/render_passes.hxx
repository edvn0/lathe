#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <type_traits>

#include <volk.h>

#include "gpu/buffer.hxx"
#include "core/config.hxx"
#include "gpu/gpu_resource_table.hxx"
#include "gpu/image_storage.hxx"
#include "rendering/pipeline_graph_repository.hxx"
#include "core/renderer_error.hxx"
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

    struct DrawBuffers {
        Buffer const &draws;
        Buffer const &transforms;
        Buffer const &indirect;
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

        // Only tiles selected by update_mask are cleared and rendered. On the
        // first use preserve_contents is false and the old layout is UNDEFINED;
        // subsequent calls preserve the other atlas tiles across frames.
        std::uint32_t update_mask = (1U << shadow_cascade_count) - 1U;
        bool preserve_contents = false;

        VkBuffer index_buffer = VK_NULL_HANDLE;
        VkDeviceAddress materials_address = 0;
        VkDeviceAddress ubo_address = 0;
        VkDeviceAddress lights_address = 0;

        PipelineNodeHandle opaque_pipeline{};
        PipelineNodeHandle mask_pipeline{};

        float depth_bias_constant = -1.0F;
        float depth_bias_slope = -2.5F;
    };

    struct ForwardTargets {
        Image const &hdr;
        Image const &depth;
        Image const *resolved_hdr = nullptr;
        Image const *resolved_depth = nullptr;
    };

    struct DepthPrepassInfo {
        Image const &depth;
        Image const *resolved_depth = nullptr;
        VkExtent2D extent{};
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

        DrawBuffers draws;
        DrawCounts counts;

        VkBuffer index_buffer = VK_NULL_HANDLE;
        VkDeviceAddress materials_address = 0;
        VkDeviceAddress ubo_address = 0;
        VkDeviceAddress lights_address = 0;

        PipelineNodeHandle opaque_pipeline{};
        PipelineNodeHandle mask_pipeline{};
    };

    // GTAO: horizon-based screen-space ambient occlusion computed entirely
    // from the depth buffer (view-space normals are reconstructed from
    // neighbouring depth samples inside the shader -- this renderer has no
    // normal G-buffer to sample instead), then denoised with a depth-aware
    // spatial blur. Runs as two compute dispatches between the depth
    // prepass and the forward pass: `depth` must already hold this frame's
    // single-sample depth (the resolved target when MSAA is on, the main
    // depth target otherwise) in DEPTH_ATTACHMENT_OPTIMAL layout on entry,
    // and is left back in that layout on return so forward_geometry's
    // LOAD_OP_LOAD attachment use is unaffected.
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

        VkBuffer index_buffer = VK_NULL_HANDLE;
        VkDeviceAddress materials_address = 0;
        VkDeviceAddress ubo_address = 0;
        VkDeviceAddress lights_address = 0;
        std::uint32_t light_count = 0;

        VkQueryPool pipeline_statistics_query_pool = VK_NULL_HANDLE;

        PipelineNodeHandle opaque_pipeline{};
        PipelineNodeHandle blend_pipeline{};

        bool draw_light_icons = false;
        PipelineNodeHandle light_icon_pipeline{};
        std::uint32_t light_icon_texture_index = 0;
        std::uint32_t linear_sampler_index = 0;
        float light_icon_world_size = 0.5F;

        // Bindless index of the (denoised) GTAO texture, or a fully-white
        // fallback when AO is disabled -- the caller decides which, since
        // that's a renderer-level policy (AoSettings::enabled), not
        // something this pass should branch on.
        std::uint32_t ao_texture_index = 0;
        std::uint32_t ao_sampler_index = 0;
    };

    struct BloomPassInfo {
        bool enabled = true;

        HdrTextureIndex input_hdr{};
        Image const *target = nullptr;
        std::array<std::uint32_t, 4> mip_texture_indices{};
        VkExtent2D input_extent{};

        PipelineNodeHandle downsample_pipeline{};
        PipelineNodeHandle upsample_pipeline{};

        std::uint32_t linear_sampler_index = 0;

        float threshold = 1.0F;
        float knee = 0.5F;
        float filter_radius = 0.005F;
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

    // Non-owning, allocation-free callback. The bound callable must outlive the
    // render-pass call, which is naturally true for the local overlay lambdas in
    // Renderer::record_frame().
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

    auto ambient_occlusion(Context const &context, AmbientOcclusionInfo const &info)
            -> std::expected<std::optional<AoTextureIndex>, RendererError>;

    auto forward_geometry(Context const &context, ForwardGeometryInfo const &info, Callback debug_overlay)
            -> std::expected<HdrTextureIndex, RendererError>;

    auto bloom(Context const &context, BloomPassInfo const &info)
            -> std::expected<std::optional<BloomTextureIndex>, RendererError>;

    auto composite(Context const &context, CompositePassInfo const &info, Callback ui_overlay)
            -> std::expected<void, RendererError>;

    auto present_swapchain(VkCommandBuffer command_buffer, VkImage image) noexcept -> void;

} // namespace render_pass
