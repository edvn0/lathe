#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <string>
#include <type_traits>

#include "core/renderer_error.hxx"
#include "gpu/upload_ring.hxx"
#include "rendering/overlay.hxx"
#include "rendering/pipeline_graph_repository.hxx"

struct Renderer;
class ShaderObjectSet;

inline constexpr auto max_overlay_push_bytes = std::size_t{128};

// A graphics shader that is bound and drawn from inside an overlay's record callback: the pipeline, the descriptor
// table, typed push constants and the draw call, without the command buffer's raw calls spread over every renderer.
class BoundOverlayShader {
public:
    BoundOverlayShader(VkCommandBuffer command_buffer, VkPipelineLayout layout) noexcept :
        command_buffer_{command_buffer}, layout_{layout} {}

    // The whole struct is the push constants, visible to every stage.
    template<typename T>
        requires(std::is_trivially_copyable_v<T> && sizeof(T) <= max_overlay_push_bytes)
    auto push(T const &constants) const noexcept -> void {
        vkCmdPushConstants(command_buffer_, layout_, VK_SHADER_STAGE_ALL, 0, sizeof(T), &constants);
    }

    auto set_topology(VkPrimitiveTopology topology) const noexcept -> void {
        vkCmdSetPrimitiveTopology(command_buffer_, topology);
    }

    auto set_line_width(float width) const noexcept -> void { vkCmdSetLineWidth(command_buffer_, width); }

    auto set_viewport(VkViewport const &viewport) const noexcept -> void {
        vkCmdSetViewportWithCount(command_buffer_, 1, &viewport);
    }

    auto set_scissor(VkRect2D const &scissor) const noexcept -> void {
        vkCmdSetScissorWithCount(command_buffer_, 1, &scissor);
    }

    // Draws over whatever is there: no depth test, no depth writes.
    auto disable_depth() const noexcept -> void {
        vkCmdSetDepthCompareOp(command_buffer_, VK_COMPARE_OP_ALWAYS);
        vkCmdSetDepthBounds(command_buffer_, 0.0F, 1.0F);
        vkCmdSetDepthTestEnable(command_buffer_, VK_FALSE);
        vkCmdSetDepthWriteEnable(command_buffer_, VK_FALSE);
    }

    // Source-alpha over the target, all channels written.
    auto enable_alpha_blend() const noexcept -> void {
        constexpr auto enable = VkBool32{VK_TRUE};
        constexpr auto equation = VkColorBlendEquationEXT{
                .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .alphaBlendOp = VK_BLEND_OP_ADD,
        };
        constexpr auto write_mask = VkColorComponentFlags{VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
        vkCmdSetColorBlendEnableEXT(command_buffer_, 0, 1, &enable);
        vkCmdSetColorBlendEquationEXT(command_buffer_, 0, 1, &equation);
        vkCmdSetColorWriteMaskEXT(command_buffer_, 0, 1, &write_mask);
    }

    auto bind_index_buffer(UploadSlice const &slice, VkIndexType type) const noexcept -> void {
        vkCmdBindIndexBuffer(command_buffer_, slice.buffer, slice.offset, type);
    }

    auto draw(std::uint32_t vertex_count, std::uint32_t instance_count = 1, std::uint32_t first_vertex = 0,
              std::uint32_t first_instance = 0) const noexcept -> void {
        vkCmdDraw(command_buffer_, vertex_count, instance_count, first_vertex, first_instance);
    }

    auto draw_indexed(std::uint32_t index_count, std::uint32_t instance_count = 1, std::uint32_t first_index = 0,
                      std::int32_t vertex_offset = 0, std::uint32_t first_instance = 0) const noexcept -> void {
        vkCmdDrawIndexed(command_buffer_, index_count, instance_count, first_index, vertex_offset, first_instance);
    }

    [[nodiscard]] auto command_buffer() const noexcept -> VkCommandBuffer { return command_buffer_; }
    [[nodiscard]] auto layout() const noexcept -> VkPipelineLayout { return layout_; }

private:
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};

struct OverlayShaderInfo {
    // Relative to the data directory.
    std::string source;
    std::string vertex_entry_point = "main_vs";
    std::string fragment_entry_point = "main_fs";
    // Source-alpha blending instead of opaque.
    bool blending = false;
    std::string debug_name;
};

// The shader of one overlay. The pipeline is registered the first time the overlay records, and again if the scope
// it records into changes (another colour or depth format, or sample count), so an overlay never has to know what
// it is drawing into until it is asked to draw.
class OverlayShader {
public:
    OverlayShader(Renderer &renderer, OverlayShaderInfo info) : renderer_{&renderer}, info_{std::move(info)} {}

    // Binds the pipeline and the descriptor table for `context`'s frame.
    [[nodiscard]] auto bind(OverlayRecordContext const &context) -> std::expected<BoundOverlayShader, RendererError>;

    [[nodiscard]] auto registered() const noexcept -> bool { return node_.valid(); }

    // Registers the pipeline again the next time it is bound.
    auto invalidate() noexcept -> void { node_ = {}; }

private:
    Renderer *renderer_;
    OverlayShaderInfo info_;
    PipelineNodeHandle node_{};
    OverlayScope scope_{};
};
