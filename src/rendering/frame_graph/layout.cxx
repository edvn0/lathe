#include "rendering/frame_graph/layout.hxx"

#include <algorithm>
#include <array>
#include <map>
#include <utility>

namespace frame_graph {

    auto layout(FrameGraphView const &view, LayoutParams const &params) -> GraphLayout {
        auto const &graph = view.graph;
        auto const &compiled = view.compiled;

        auto result = GraphLayout{};
        result.nodes.resize(graph.passes.size());

        for (auto pass = std::uint32_t{0}; pass < graph.passes.size(); ++pass) {
            auto &node = result.nodes[pass];
            node.pass = pass;
            node.queue = pass < compiled.pass_queue.size() ? compiled.pass_queue[pass] : LogicalQueue::graphics;
            node.culled = pass < compiled.pass_culled.size() && compiled.pass_culled[pass];

            for (auto const &access: graph.passes[pass].accesses) {
                if (access.resource >= graph.producers.size() ||
                    access.version >= graph.producers[access.resource].size()) {
                    continue;
                }

                auto const producer = graph.producers[access.resource][access.version];
                if (producer < 0 || std::cmp_greater_equal(producer, pass)) {
                    continue;
                }

                auto const from = static_cast<std::uint32_t>(producer);
                result.edges.push_back({.from = from, .to = pass, .resource = access.resource});
                node.depth = std::max(node.depth, result.nodes[from].depth + 1);
            }
        }

        auto stack_height = std::map<std::pair<LogicalQueue, std::uint32_t>, std::uint32_t>{};
        auto row_of = std::vector<std::uint32_t>(result.nodes.size());
        auto lane_rows = std::array<std::uint32_t, logical_queue_count>{};

        for (auto const &node: result.nodes) {
            auto &row = stack_height[{node.queue, node.depth}];
            row_of[node.pass] = row++;
            auto &rows = lane_rows[static_cast<std::size_t>(node.queue)];
            rows = std::max(rows, row);
        }

        auto lane_top = std::array<float, logical_queue_count>{};
        auto cursor = 0.0F;
        for (auto lane = std::size_t{0}; lane < logical_queue_count; ++lane) {
            if (lane_rows[lane] == 0) {
                continue;
            }
            lane_top[lane] = cursor;
            cursor += (static_cast<float>(lane_rows[lane]) * params.row_height) + params.lane_gap;
        }

        for (auto &node: result.nodes) {
            node.x = static_cast<float>(node.depth) * params.column_width;
            node.y = lane_top[static_cast<std::size_t>(node.queue)] +
                     (static_cast<float>(row_of[node.pass]) * params.row_height);
        }

        return result;
    }

}
