#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <type_traits>

#include <volk.h>

#include <glm/vec3.hpp>

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

        bool compute_only = false;
    };

    struct DrawBuffers {
        Buffer const &draws;
        Buffer const &transforms;
        Buffer const &indirect;
        VkBuffer index_buffer = VK_NULL_HANDLE;
    };

    struct DrawCounts {
        std::uint32_t opaque = 0;
        std::uint32_t double_sided = 0;
        std::uint32_t mask = 0;
        std::uint32_t blend = 0;

        [[nodiscard]] constexpr auto double_sided_first() const noexcept -> std::uint32_t { return opaque; }
        [[nodiscard]] constexpr auto mask_first() const noexcept -> std::uint32_t { return opaque + double_sided; }
        [[nodiscard]] constexpr auto blend_first() const noexcept -> std::uint32_t {
            return opaque + double_sided + mask;
        }
    };

    struct ShadowPassInfo {
        DrawBuffers draws;
        DrawCounts counts;
        std::array<std::uint32_t, shadow_cascade_count> const &opaque_cascade_counts;
        std::array<std::uint32_t, shadow_cascade_count> const &double_sided_cascade_counts;
        std::array<std::uint32_t, shadow_cascade_count> const &mask_cascade_counts;

        std::uint32_t update_mask = (1U << shadow_cascade_count) - 1U;
        bool preserve_contents = false;

        bool meshlet_culling = true;

        VkDeviceAddress cascade_cull_planes_address = 0;
        VkDeviceAddress materials_address = 0;
        VkDeviceAddress ubo_address = 0;
        VkDeviceAddress lights_address = 0;

        PipelineNodeHandle opaque_pipeline{};
        PipelineNodeHandle mask_pipeline{};
        PipelineNodeHandle opaque_instanced_pipeline{};
        PipelineNodeHandle mask_instanced_pipeline{};

        float depth_bias_constant = -1.0F;
        float depth_bias_slope = -2.5F;
    };

    enum class DepthPrepassPhase : std::uint8_t {
        only,
        early,
        late,
    };

    inline constexpr std::uint32_t cull_occlusion = 4U;
    inline constexpr std::uint32_t cull_skip_recorded = 8U;
    inline constexpr std::uint32_t cull_record = 16U;
    inline constexpr std::uint32_t cull_replay = 32U;
    inline constexpr std::uint32_t cull_stats = 64U;

    struct DepthPrepassInfo {
        VkExtent2D extent{};
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

        DepthPrepassPhase phase = DepthPrepassPhase::only;

        DrawBuffers draws;
        DrawCounts counts;

        VkDeviceAddress cull_planes_address = 0;
        VkDeviceAddress materials_address = 0;
        VkDeviceAddress ubo_address = 0;
        VkDeviceAddress lights_address = 0;

        VkDeviceAddress occlusion_view_address = 0;
        std::uint32_t extra_cull_flags = 0;

        PipelineNodeHandle opaque_pipeline{};
        PipelineNodeHandle mask_pipeline{};
        PipelineNodeHandle opaque_instanced_pipeline{};
        PipelineNodeHandle mask_instanced_pipeline{};

        bool meshlet_culling = true;
    };

    struct AmbientOcclusionInfo {
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
        HdrTextureIndex output_hdr{};

        VkExtent2D extent{};
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

        DrawBuffers draws;
        DrawCounts counts;

        VkDeviceAddress cull_planes_address = 0;
        VkDeviceAddress materials_address = 0;
        VkDeviceAddress ubo_address = 0;
        VkDeviceAddress lights_address = 0;
        std::uint32_t light_count = 0;

        VkDeviceAddress cluster_lights_address = 0;

        VkDeviceAddress occlusion_view_address = 0;
        std::uint32_t extra_cull_flags = 0;

        VkQueryPool pipeline_statistics_query_pool = VK_NULL_HANDLE;

        bool meshlet_culling = true;

        PipelineNodeHandle opaque_pipeline{};
        PipelineNodeHandle blend_pipeline{};
        PipelineNodeHandle opaque_instanced_pipeline{};
        PipelineNodeHandle blend_instanced_pipeline{};

        PipelineNodeHandle skybox_pipeline{};
        bool draw_skybox = false;

        std::uint32_t ao_texture_index = 0;
        std::uint32_t ao_sampler_index = 0;

        bool outline_mask = false;
    };

    inline constexpr std::uint32_t bloom_mip_count = 4;

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

        float filter_radius = 1.0F;
    };

    struct CompositePassInfo {
        VkExtent2D extent{};

        HdrTextureIndex hdr{};
        std::optional<BloomTextureIndex> bloom;
        std::uint32_t bloom_fallback_texture_index = 0;
        std::uint32_t linear_sampler_index = 0;

        PipelineNodeHandle pipeline{};

        float exposure = 1.0F;
        float bloom_intensity = 0.0F;

        std::uint32_t outline_texture_index = 0;
        float outline_thickness_pixels = 0.0F;
        glm::vec3 outline_colour{0.0F};
    };

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

    auto shadow(Context const &context, ShadowPassInfo const &info) -> std::expected<void, RendererError>;

    auto depth_prepass(Context const &context, DepthPrepassInfo const &info) -> std::expected<void, RendererError>;

    struct HizBuildInfo {
        std::uint32_t source_texture_index = 0;
        VkExtent2D depth_extent{};

        Image const &hiz;
        std::span<std::uint32_t const> mip_texture_indices;

        PipelineNodeHandle pipeline{};
    };

    auto build_hiz(Context const &context, HizBuildInfo const &info) -> std::expected<void, RendererError>;

    auto gtao(Context const &context, AmbientOcclusionInfo const &info) -> std::expected<void, RendererError>;
    auto gtao_denoise(Context const &context, AmbientOcclusionInfo const &info) -> std::expected<void, RendererError>;

    auto forward_geometry(Context const &context, ForwardGeometryInfo const &info, Callback scene_overlays)
            -> std::expected<HdrTextureIndex, RendererError>;

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

    auto light_icons(Context const &context, LightIconsInfo const &info, OverlayScope const &scope) noexcept -> void;

    auto bloom(Context const &context, BloomPassInfo const &info)
            -> std::expected<std::optional<BloomTextureIndex>, RendererError>;

    auto composite(Context const &context, CompositePassInfo const &info, Callback ui_overlay)
            -> std::expected<void, RendererError>;

}
