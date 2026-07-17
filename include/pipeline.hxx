#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <span>
#include <string_view>

#include "forward.hxx"

enum class PipelineErrorType : std::uint8_t {
    invalid_argument,
    layout_creation_failed,
    pipeline_creation_failed,
};

struct PipelineError {
    PipelineErrorType type = PipelineErrorType::invalid_argument;

    VkResult result = VK_SUCCESS;
};

struct ShaderStageInfo {
    VkShaderStageFlagBits stage{};
    std::span<const std::uint32_t> spirv;
    char const *entry_point = "main";
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

    std::string_view debug_name = "graphics_pipeline";
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

    [[nodiscard]]
    static auto create_graphics(VulkanContext &context, GraphicsPipelineCreateInfo const &create_info,
                                VkDescriptorSetLayout global_layout) -> std::expected<Pipeline, PipelineError>;
};
