#pragma once

#include <cstdint>

#include "rendering/frame_graph/aliasing.hxx"
#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"

namespace frame_graph {

    // A copy of the last distinct frame graph, for tools that outlive the frame. The graph is rebuilt every frame
    // and the compiled plan lives in the plan cache, so neither can be held across frames. Pass indices in
    // `compiled` refer to `graph.passes`.
    struct FrameGraphView {
        GraphDesc graph;
        CompiledGraph compiled;
        TransientPlan transients;
        std::uint64_t revision = 0;
    };

}
