#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "rendering/frame_graph/types.hxx"
#include "rendering/frame_graph/view.hxx"

namespace frame_graph {

    inline constexpr auto no_group = ~std::uint32_t{0};

    struct LayoutParams {
        float column_width = 340.0F;
        float row_gap = 28.0F;
        float lane_gap = 90.0F;
        float default_height = 110.0F;
    };

    // Passes sharing a group collapse into one node; no_group keeps a pass on its own. Both spans may be empty or
    // shorter than needed: missing entries mean no_group and default_height.
    struct LayoutInput {
        std::span<std::uint32_t const> group_of_pass;
        std::span<float const> group_heights;
    };

    struct LayoutNode {
        std::uint32_t group = no_group;
        std::vector<std::uint32_t> passes;
        LogicalQueue queue = LogicalQueue::graphics;
        std::uint32_t depth = 0;
        bool culled = false;
        float x = 0.0F;
        float y = 0.0F;
    };

    // `to` accesses `resources` at the versions `from` produced. Nodes are indexed by first pass.
    struct LayoutEdge {
        std::uint32_t from = 0;
        std::uint32_t to = 0;
        std::vector<std::uint32_t> resources;
        bool cross_queue = false;
    };

    struct GraphLayout {
        std::vector<LayoutNode> nodes;
        std::vector<LayoutEdge> edges;
    };

    // Columns are dependency depth, rows stack within a lane per logical queue. Grouping can make dependencies
    // circular; edges that run against declaration order are drawn but do not affect depth.
    [[nodiscard]] auto layout(FrameGraphView const &view, LayoutParams const &params = {},
                              LayoutInput const &input = {}) -> GraphLayout;

}
