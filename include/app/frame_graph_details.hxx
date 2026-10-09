#pragma once

#include <cstdint>
#include <span>

#include "rendering/frame_graph/view.hxx"

class Renderer;

namespace gui {

    // What the selected passes do: every resource access with its use, and the barriers inserted before the pass.
    auto draw_pass_details(frame_graph::FrameGraphView const &view, std::span<std::uint32_t const> passes,
                           Renderer const &renderer) -> void;

}
