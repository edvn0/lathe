#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string_view>

#include <optional>

#include "error_context.hxx"
#include "fly_string.hxx"
#include "forward.hxx"

enum class PipelineErrorType : std::uint8_t {
    invalid_argument,
    layout_creation_failed,
    pipeline_creation_failed,
};

struct PipelineError {
    PipelineErrorType type = PipelineErrorType::invalid_argument;

    // Carries the pipeline's debug_name / rejected-argument reason plus, when
    // applicable, the failing VkResult (see ErrorContext::vk_result).
    std::optional<ErrorContext> context {std::nullopt};
};

template<>
struct std::formatter<PipelineErrorType> : std::formatter<std::string_view> {
    constexpr auto format(PipelineErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case PipelineErrorType::invalid_argument:
                    return "invalid_argument";
                case PipelineErrorType::layout_creation_failed:
                    return "layout_creation_failed";
                case PipelineErrorType::pipeline_creation_failed:
                    return "pipeline_creation_failed";
            }

            return "unknown_pipeline_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct ShaderStageInfo {
    VkShaderStageFlagBits stage{};
    std::span<const std::uint32_t> spirv;
    FlyString entry_point = FlyString{"main"};
    VkPipelineShaderStageCreateFlags flags = 0;
    VkSpecializationInfo const *specialization_info = nullptr;
};

struct GraphicsPipelineCreateInfo {
    std::span<const ShaderStageInfo> shaders;

    /*
     * Additional layouts beginning at descriptor set 1.
     * Set 0 is injected by PipelineStorage.
     */
    std::span<VkDescriptorSetLayout const> additional_descriptor_set_layouts;

    std::span<VkPushConstantRange const> push_constant_ranges;

    std::span<VkDynamicState const> dynamic_states;

    std::span<VkFormat const> colour_formats;

    VkFormat depth_format = VK_FORMAT_UNDEFINED;

    VkFormat stencil_format = VK_FORMAT_UNDEFINED;

    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;

    VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;

    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    bool depth_test = false;
    bool depth_write = false;

    VkCompareOp depth_compare = VK_COMPARE_OP_ALWAYS;

    bool blending = false;

    bool use_vertex_input = false; // VP

    std::string_view debug_name = "graphics_pipeline";
};

struct ComputePipelineCreateInfo {
    ShaderStageInfo shader;

    /*
     * Additional layouts beginning at descriptor set 1.
     * Set 0 is injected by PipelineStorage.
     */
    std::span<VkDescriptorSetLayout const> additional_descriptor_set_layouts;

    std::span<VkPushConstantRange const> push_constant_ranges;

    std::string_view debug_name = "compute_pipeline";
};

class Pipeline {
public:
    Pipeline() = default;
    ~Pipeline();

    Pipeline(Pipeline const &) = delete;

    auto operator=(Pipeline const &) -> Pipeline & = delete;

    Pipeline(Pipeline &&other) noexcept;

    auto operator=(Pipeline &&other) noexcept -> Pipeline &;

    auto destroy() noexcept -> void;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return pipeline_ != VK_NULL_HANDLE;
    }

    [[nodiscard]]
    auto pipeline() const noexcept -> VkPipeline {
        return pipeline_;
    }

    [[nodiscard]]
    auto layout() const noexcept -> VkPipelineLayout {
        return layout_;
    }

    [[nodiscard]]
    auto bind_point() const noexcept -> VkPipelineBindPoint {
        return bind_point_;
    }

private:
    VulkanContext *context_ = nullptr;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;

    VkPipelineBindPoint bind_point_ = VK_PIPELINE_BIND_POINT_GRAPHICS;

    friend class PipelineStorage;

    // pipeline_cache may be built concurrently from multiple threads (see
    // PipelineGraphRepository::register_pipelines_parallel); per the Vulkan
    // spec a VkPipelineCache used this way requires external
    // synchronization. That's handled internally (a translation-unit-local
    // mutex in pipeline.cxx guarding just the vkCreateGraphicsPipelines /
    // vkCreateComputePipelines call) -- callers just pass the handle, or
    // VK_NULL_HANDLE if they don't have a cache.
    [[nodiscard]]
    static auto create_graphics(VulkanContext &context, GraphicsPipelineCreateInfo const &create_info,
                                VkDescriptorSetLayout global_layout, VkPipelineCache pipeline_cache)
            -> std::expected<Pipeline, PipelineError>;

    [[nodiscard]]
    static auto create_compute(VulkanContext &context, ComputePipelineCreateInfo const &create_info,
                               VkDescriptorSetLayout global_layout, VkPipelineCache pipeline_cache)
            -> std::expected<Pipeline, PipelineError>;
};
