#pragma once

#include <cstdint>
#include <vector>

#include "rendering/frame_graph/types.hxx"
#include "rendering/frame_graph/view.hxx"

namespace frame_graph {

    struct LayoutParams {
        float column_width = 280.0F;
        float row_height = 140.0F;
        float lane_gap = 90.0F;
    };

    struct LayoutNode {
        std::uint32_t pass = 0;
        LogicalQueue queue = LogicalQueue::graphics;
        std::uint32_t depth = 0;
        bool culled = false;
        float x = 0.0F;
        float y = 0.0F;
    };

    // `to` accesses `resource` at the version `from` produced.
    struct LayoutEdge {
        std::uint32_t from = 0;
        std::uint32_t to = 0;
        std::uint32_t resource = 0;
    };

    struct GraphLayout {
        std::vector<LayoutNode> nodes;
        std::vector<LayoutEdge> edges;
    };

    // Columns are dependency depth, rows are stacked within a lane per logical queue. Nodes are indexed by pass.
    [[nodiscard]] auto layout(FrameGraphView const &view, LayoutParams const &params = {}) -> GraphLayout;

}
