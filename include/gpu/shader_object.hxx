#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string_view>

#include "core/error_context.hxx"
#include "core/forward.hxx"
#include "gpu/pipeline.hxx" // ShaderStageInfo

enum class ShaderObjectErrorType : std::uint8_t {
    invalid_argument,
    layout_creation_failed,
    shader_creation_failed,
};

struct ShaderObjectError {
    ShaderObjectErrorType type = ShaderObjectErrorType::invalid_argument;

    // Carries the debug_name / rejected-argument reason plus, when
    // applicable, the failing VkResult (see ErrorContext::vk_result).
    std::optional<ErrorContext> context{std::nullopt};
};

template<>
struct std::formatter<ShaderObjectErrorType> : std::formatter<std::string_view> {
    constexpr auto format(ShaderObjectErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case ShaderObjectErrorType::invalid_argument:
                    return "invalid_argument";
                case ShaderObjectErrorType::layout_creation_failed:
                    return "layout_creation_failed";
                case ShaderObjectErrorType::shader_creation_failed:
                    return "shader_creation_failed";
            }

            return "unknown_shader_object_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct ShaderObjectCreateInfo {
    // Stages in pipeline order (e.g. vertex,fragment or task,mesh,fragment).
    // nextStage chaining for the VK_SHADER_CREATE_LINK_STAGE_BIT_EXT group is
    // derived from this order.
    std::span<const ShaderStageInfo> shaders;

    /*
     * Additional layouts beginning at descriptor set 1.
     * Set 0 is injected by the caller (global_layout).
     */
    std::span<VkDescriptorSetLayout const> additional_descriptor_set_layouts;

    std::span<VkPushConstantRange const> push_constant_ranges;

    std::string_view debug_name = "shader_object_set";
};

struct ComputeShaderCreateInfo {
    ShaderStageInfo shader;

    std::span<VkDescriptorSetLayout const> additional_descriptor_set_layouts;

    std::span<VkPushConstantRange const> push_constant_ranges;

    std::string_view debug_name = "compute_shader_object";
};

// Parallel to Pipeline: a move-only RAII wrapper around a linked group of
// VkShaderEXT objects (or a single unlinked compute shader) sharing one
// VkPipelineLayout. See docs/pipeline_to_shader_objects.md Phase 2.
class ShaderObjectSet {
public:
    static constexpr std::uint32_t max_stages = 4; // task, mesh/vertex, geometry(unused), fragment

    ShaderObjectSet() = default;
    ~ShaderObjectSet();

    ShaderObjectSet(ShaderObjectSet const &) = delete;
    auto operator=(ShaderObjectSet const &) -> ShaderObjectSet & = delete;
    ShaderObjectSet(ShaderObjectSet &&other) noexcept;
    auto operator=(ShaderObjectSet &&other) noexcept -> ShaderObjectSet &;

    auto destroy() noexcept -> void;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return count_ != 0;
    }

    [[nodiscard]]
    auto layout() const noexcept -> VkPipelineLayout {
        return layout_;
    }

    [[nodiscard]]
    auto bind_point() const noexcept -> VkPipelineBindPoint {
        return bind_point_;
    }

    auto bind(VkCommandBuffer command_buffer) const noexcept -> void;

private:
    VulkanContext *context_ = nullptr;

    std::array<VkShaderEXT, max_stages> shaders_{};
    std::array<VkShaderStageFlagBits, max_stages> stages_{};
    std::uint32_t count_ = 0;

    VkPipelineLayout layout_ = VK_NULL_HANDLE;

    VkPipelineBindPoint bind_point_ = VK_PIPELINE_BIND_POINT_GRAPHICS;

    friend class ShaderObjectStorage;

    [[nodiscard]]
    static auto create_linked(VulkanContext &context, ShaderObjectCreateInfo const &create_info,
                              VkDescriptorSetLayout global_layout) -> std::expected<ShaderObjectSet, ShaderObjectError>;

    [[nodiscard]]
    static auto create_compute(VulkanContext &context, ComputeShaderCreateInfo const &create_info,
                               VkDescriptorSetLayout global_layout)
            -> std::expected<ShaderObjectSet, ShaderObjectError>;
};
