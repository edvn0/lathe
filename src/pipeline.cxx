#include "pipeline.hxx"

#include <array>
#include <format>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "context.hxx"
#include "load_model.hxx"
#include "vk_object_name.hxx"

namespace {
    inline constexpr std::array forward_dynamic_states{
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,

            VK_DYNAMIC_STATE_LINE_WIDTH,
            VK_DYNAMIC_STATE_DEPTH_BIAS,
            VK_DYNAMIC_STATE_BLEND_CONSTANTS,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS,
            VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE,

            VK_DYNAMIC_STATE_CULL_MODE,
            VK_DYNAMIC_STATE_FRONT_FACE,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
            VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
            VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
            VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
            VK_DYNAMIC_STATE_STENCIL_OP,

            VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
            VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
            VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
    };

    auto make_error(PipelineErrorType type, std::string_view message = {}, VkResult result = VK_SUCCESS,
                    std::source_location location = std::source_location::current()) noexcept -> PipelineError {
        return PipelineError{
                .type = type,
                .context =
                        ErrorContext{
                                .message = FlyString{message},
                                .vk_result = result != VK_SUCCESS ? std::optional{result} : std::nullopt,
                                .location = location,
                        },
        };
    }
} // namespace

Pipeline::~Pipeline() { destroy(); }

Pipeline::Pipeline(Pipeline &&other) noexcept :
    context_(std::exchange(other.context_, nullptr)), pipeline_(std::exchange(other.pipeline_, VK_NULL_HANDLE)),
    layout_(std::exchange(other.layout_, VK_NULL_HANDLE)),
    bind_point_(std::exchange(other.bind_point_, VK_PIPELINE_BIND_POINT_GRAPHICS)) {}

auto Pipeline::operator=(Pipeline &&other) noexcept -> Pipeline & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);

    pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);

    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);

    bind_point_ = std::exchange(other.bind_point_, VK_PIPELINE_BIND_POINT_GRAPHICS);

    return *this;
}

constexpr auto default_bindings() -> VkPipelineVertexInputStateCreateInfo {
    auto [attributes, bindings] = default_vertex_description();
    VkPipelineVertexInputStateCreateInfo vertex_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .vertexBindingDescriptionCount = static_cast<std::uint32_t>(bindings.size()),
            .pVertexBindingDescriptions = bindings.data(),
            .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size()),
            .pVertexAttributeDescriptions = attributes.data(),
    };

    return vertex_input;
}

auto Pipeline::create_graphics(VulkanContext &context, GraphicsPipelineCreateInfo const &create_info,
                               VkDescriptorSetLayout global_layout) -> std::expected<Pipeline, PipelineError> {
    if (context.device == VK_NULL_HANDLE || create_info.shaders.empty()) {
        return std::unexpected(make_error(PipelineErrorType::invalid_argument,
                                          context.device == VK_NULL_HANDLE ? "device is VK_NULL_HANDLE"
                                                                            : "no shader stages provided"));
    }

    std::vector<VkShaderModuleCreateInfo> shader_module_infos(create_info.shaders.size());

    std::vector<VkPipelineShaderStageCreateInfo> shader_stages(create_info.shaders.size());

    for (std::size_t index = 0; index < create_info.shaders.size(); ++index) {
        auto const &shader = create_info.shaders[index];

        if (shader.spirv.empty() || shader.entry_point.empty() ||
            shader.spirv.size_bytes() % sizeof(std::uint32_t) != 0) {
            return std::unexpected(make_error(
                    PipelineErrorType::invalid_argument,
                    std::format("shader stage {} has invalid SPIR-V or entry point '{}'", index,
                               shader.entry_point.view())));
        }

        shader_module_infos[index] = VkShaderModuleCreateInfo{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = shader.spirv.size_bytes(),
                .pCode = shader.spirv.data(),
        };

        shader_stages[index] = VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = &shader_module_infos[index],
                .flags = shader.flags,
                .stage = shader.stage,
                .module = VK_NULL_HANDLE,
                .pName = shader.entry_point.view().data(),
                .pSpecializationInfo = shader.specialization_info,
        };
    }

    auto layouts = std::vector<VkDescriptorSetLayout>{};

    layouts.push_back(global_layout);

    layouts.insert(layouts.end(), create_info.additional_descriptor_set_layouts.begin(),
                   create_info.additional_descriptor_set_layouts.end());

    VkPipelineLayoutCreateInfo const layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = static_cast<std::uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
            .pushConstantRangeCount = static_cast<std::uint32_t>(create_info.push_constant_ranges.size()),
            .pPushConstantRanges = create_info.push_constant_ranges.data(),
    };

    Pipeline result;
    result.context_ = &context;

    auto vk_result = vkCreatePipelineLayout(context.device, &layout_info, nullptr, &result.layout_);

    if (vk_result != VK_SUCCESS) {
        result.context_ = nullptr;

        return std::unexpected(make_error(
                PipelineErrorType::layout_creation_failed,
                std::format("vkCreatePipelineLayout failed for pipeline '{}'", create_info.debug_name), vk_result));
    }

    VkPipelineVertexInputStateCreateInfo const empty_vertex_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .vertexBindingDescriptionCount = 0,
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr,
    };

    VkPipelineVertexInputStateCreateInfo const vertex_input =
            create_info.use_vertex_input ? default_bindings() : empty_vertex_input;

    VkPipelineInputAssemblyStateCreateInfo const input_assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .topology = create_info.topology,
            .primitiveRestartEnable = VK_FALSE,
    };

    VkPipelineViewportStateCreateInfo const viewport_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr,
    };

    VkPipelineRasterizationStateCreateInfo const rasterization{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = create_info.polygon_mode,
            .cullMode = create_info.cull_mode,
            .frontFace = create_info.front_face,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0F,
            .depthBiasClamp = 0.0F,
            .depthBiasSlopeFactor = 0.0F,
            .lineWidth = 1.0F,
    };

    VkPipelineMultisampleStateCreateInfo const multisample{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .rasterizationSamples = create_info.samples,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 0.0F,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE,
    };

    VkPipelineDepthStencilStateCreateInfo const depth_stencil{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthTestEnable = create_info.depth_test ? VK_TRUE : VK_FALSE,
            .depthWriteEnable = create_info.depth_write ? VK_TRUE : VK_FALSE,
            .depthCompareOp = create_info.depth_compare,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
            .front = {},
            .back = {},
            .minDepthBounds = 0.0F,
            .maxDepthBounds = 1.0F,
    };

    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(create_info.colour_formats.size());

    for (auto &attachment: blend_attachments) {
        attachment = VkPipelineColorBlendAttachmentState{
                .blendEnable = create_info.blending ? VK_TRUE : VK_FALSE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                  VK_COLOR_COMPONENT_A_BIT,
        };
    }

    VkPipelineColorBlendStateCreateInfo const colour_blend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = static_cast<std::uint32_t>(blend_attachments.size()),
            .pAttachments = blend_attachments.data(),
            .blendConstants =
                    {
                            0.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                    },
    };

    std::unordered_set<VkDynamicState> dynamic_states{};
    for (const auto &state: forward_dynamic_states) {
        dynamic_states.insert(state);
    }
    for (const auto &state: create_info.dynamic_states) {
        dynamic_states.insert(state);
    }

    std::vector<VkDynamicState> vector;
    vector.reserve(dynamic_states.size());
    for (auto it = dynamic_states.begin(); it != dynamic_states.end();) {
        vector.push_back(std::move(dynamic_states.extract(it++).value()));
    }

    VkPipelineDynamicStateCreateInfo const dynamic_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .dynamicStateCount = static_cast<std::uint32_t>(vector.size()),
            .pDynamicStates = vector.data(),
    };

    VkPipelineRenderingCreateInfo const rendering_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .pNext = nullptr,
            .viewMask = 0,
            .colorAttachmentCount = static_cast<std::uint32_t>(create_info.colour_formats.size()),
            .pColorAttachmentFormats = create_info.colour_formats.data(),
            .depthAttachmentFormat = create_info.depth_format,
            .stencilAttachmentFormat = create_info.stencil_format,
    };

    VkGraphicsPipelineCreateInfo const pipeline_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering_info,
            .flags = 0,
            .stageCount = static_cast<std::uint32_t>(shader_stages.size()),
            .pStages = shader_stages.data(),
            .pVertexInputState = &vertex_input,
            .pInputAssemblyState = &input_assembly,
            .pTessellationState = nullptr,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &colour_blend,
            .pDynamicState = vector.empty() ? nullptr : &dynamic_state,
            .layout = result.layout_,
            .renderPass = VK_NULL_HANDLE,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
    };

    vk_result =
            vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &result.pipeline_);

    if (vk_result != VK_SUCCESS) {
        vkDestroyPipelineLayout(context.device, result.layout_, nullptr);

        result.layout_ = VK_NULL_HANDLE;

        result.context_ = nullptr;

        return std::unexpected(make_error(
                PipelineErrorType::pipeline_creation_failed,
                std::format("vkCreateGraphicsPipelines failed for pipeline '{}'", create_info.debug_name), vk_result));
    }

    result.bind_point_ = VK_PIPELINE_BIND_POINT_GRAPHICS;

    vk::set_object_name(context.device, VK_OBJECT_TYPE_PIPELINE, vk::object_handle(result.pipeline_),
                        create_info.debug_name);
    auto const layout_name = std::string{create_info.debug_name} + ".layout";
    vk::set_object_name(context.device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, vk::object_handle(result.layout_), layout_name);

    return result;
}

auto Pipeline::create_compute(VulkanContext &context, ComputePipelineCreateInfo const &create_info,
                              VkDescriptorSetLayout global_layout) -> std::expected<Pipeline, PipelineError> {
    if (context.device == VK_NULL_HANDLE || create_info.shader.spirv.empty()) {
        return std::unexpected(make_error(PipelineErrorType::invalid_argument,
                                          context.device == VK_NULL_HANDLE ? "device is VK_NULL_HANDLE"
                                                                            : "no shader stage provided"));
    }

    auto const &shader = create_info.shader;

    if (shader.entry_point.empty() || shader.spirv.size_bytes() % sizeof(std::uint32_t) != 0) {
        return std::unexpected(make_error(
                PipelineErrorType::invalid_argument,
                std::format("compute shader has invalid SPIR-V or entry point '{}'", shader.entry_point.view())));
    }

    VkShaderModuleCreateInfo const shader_module_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = shader.spirv.size_bytes(),
            .pCode = shader.spirv.data(),
    };

    VkPipelineShaderStageCreateInfo const shader_stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &shader_module_info,
            .flags = shader.flags,
            .stage = shader.stage,
            .module = VK_NULL_HANDLE,
            .pName = shader.entry_point.view().data(),
            .pSpecializationInfo = shader.specialization_info,
    };

    auto layouts = std::vector<VkDescriptorSetLayout>{};

    layouts.push_back(global_layout);

    layouts.insert(layouts.end(), create_info.additional_descriptor_set_layouts.begin(),
                   create_info.additional_descriptor_set_layouts.end());

    VkPipelineLayoutCreateInfo const layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = static_cast<std::uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
            .pushConstantRangeCount = static_cast<std::uint32_t>(create_info.push_constant_ranges.size()),
            .pPushConstantRanges = create_info.push_constant_ranges.data(),
    };

    Pipeline result;
    result.context_ = &context;

    auto vk_result = vkCreatePipelineLayout(context.device, &layout_info, nullptr, &result.layout_);

    if (vk_result != VK_SUCCESS) {
        result.context_ = nullptr;

        return std::unexpected(make_error(
                PipelineErrorType::layout_creation_failed,
                std::format("vkCreatePipelineLayout failed for pipeline '{}'", create_info.debug_name), vk_result));
    }

    VkComputePipelineCreateInfo const pipeline_info{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = shader_stage,
            .layout = result.layout_,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
    };

    vk_result =
            vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &result.pipeline_);

    if (vk_result != VK_SUCCESS) {
        vkDestroyPipelineLayout(context.device, result.layout_, nullptr);

        result.layout_ = VK_NULL_HANDLE;

        result.context_ = nullptr;

        return std::unexpected(make_error(
                PipelineErrorType::pipeline_creation_failed,
                std::format("vkCreateComputePipelines failed for pipeline '{}'", create_info.debug_name), vk_result));
    }

    result.bind_point_ = VK_PIPELINE_BIND_POINT_COMPUTE;

    vk::set_object_name(context.device, VK_OBJECT_TYPE_PIPELINE, vk::object_handle(result.pipeline_),
                        create_info.debug_name);
    auto const layout_name = std::string{create_info.debug_name} + ".layout";
    vk::set_object_name(context.device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, vk::object_handle(result.layout_), layout_name);

    return result;
}

auto Pipeline::destroy() noexcept -> void {
    if (context_ == nullptr) {
        return;
    }

    if (pipeline_ != VK_NULL_HANDLE && context_->device != VK_NULL_HANDLE) {
        vkDestroyPipeline(context_->device, pipeline_, nullptr);
    }

    pipeline_ = VK_NULL_HANDLE;

    if (layout_ != VK_NULL_HANDLE && context_->device != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context_->device, layout_, nullptr);
    }

    layout_ = VK_NULL_HANDLE;
    context_ = nullptr;
}
