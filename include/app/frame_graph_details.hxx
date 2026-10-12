#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "rendering/frame_graph/view.hxx"

class Renderer;

namespace gui {

    // What the selected passes do: every resource access with its use, and the barriers inserted before the pass.
    auto draw_pass_details(frame_graph::FrameGraphView const &view, std::span<std::uint32_t const> passes,
                           Renderer &renderer) -> void;

    // What a link carries: format and size, who writes and reads it, and which memory it shares.
    auto draw_resource_details(frame_graph::FrameGraphView const &view, std::span<std::uint32_t const> resources,
                               Renderer &renderer) -> void;

    // The image the renderer was asked to preview, if the current graph still has it and it can be sampled.
    [[nodiscard]] auto previewed_resource(frame_graph::FrameGraphView const &view, Renderer const &renderer)
            -> std::optional<std::uint32_t>;

    auto draw_preview(frame_graph::FrameGraphView const &view, std::uint32_t resource, Renderer &renderer) -> void;

}
