#include "rendering/frame_graph/layout.hxx"

#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <utility>

namespace frame_graph {
    namespace {

        auto group_of(LayoutInput const &input, std::uint32_t pass) -> std::uint32_t {
            return pass < input.group_of_pass.size() ? input.group_of_pass[pass] : no_group;
        }

        auto height_of(LayoutParams const &params, LayoutInput const &input, std::uint32_t group) -> float {
            return group < input.group_heights.size() ? input.group_heights[group] : params.default_height;
        }

    }

    auto layout(FrameGraphView const &view, LayoutParams const &params, LayoutInput const &input) -> GraphLayout {
        auto const &graph = view.graph;
        auto const &compiled = view.compiled;

        auto result = GraphLayout{};
        auto unit_of_pass = std::vector<std::uint32_t>(graph.passes.size());
        auto unit_of_group = std::unordered_map<std::uint32_t, std::uint32_t>{};

        for (auto pass = std::uint32_t{0}; pass < graph.passes.size(); ++pass) {
            auto const group = group_of(input, pass);
            auto const known = group == no_group ? unit_of_group.end() : unit_of_group.find(group);

            if (known != unit_of_group.end()) {
                unit_of_pass[pass] = known->second;
                result.nodes[known->second].passes.push_back(pass);
                continue;
            }

            auto const unit = static_cast<std::uint32_t>(result.nodes.size());
            unit_of_pass[pass] = unit;
            if (group != no_group) {
                unit_of_group.emplace(group, unit);
            }
            result.nodes.push_back(LayoutNode{
                    .group = group,
                    .passes = {pass},
                    .queue = pass < compiled.pass_queue.size() ? compiled.pass_queue[pass] : LogicalQueue::graphics,
            });
        }

        for (auto &node: result.nodes) {
            node.culled = std::ranges::all_of(node.passes, [&](std::uint32_t pass) {
                return pass < compiled.pass_culled.size() && compiled.pass_culled[pass];
            });
        }

        auto edge_index = std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t>{};
        for (auto pass = std::uint32_t{0}; pass < graph.passes.size(); ++pass) {
            for (auto const &access: graph.passes[pass].accesses) {
                if (access.resource >= graph.producers.size() ||
                    access.version >= graph.producers[access.resource].size()) {
                    continue;
                }

                auto const producer = graph.producers[access.resource][access.version];
                if (producer < 0 || std::cmp_greater_equal(producer, pass)) {
                    continue;
                }

                auto const from = unit_of_pass[static_cast<std::size_t>(producer)];
                auto const to = unit_of_pass[pass];
                if (from == to) {
                    continue;
                }

                auto [found, inserted] = edge_index.try_emplace({from, to}, result.edges.size());
                if (inserted) {
                    result.edges.push_back(LayoutEdge{
                            .from = from,
                            .to = to,
                            .cross_queue = result.nodes[from].queue != result.nodes[to].queue,
                    });
                }

                auto &resources = result.edges[found->second].resources;
                if (std::ranges::find(resources, access.resource) == resources.end()) {
                    resources.push_back(access.resource);
                }
            }
        }

        for (auto unit = std::uint32_t{0}; unit < result.nodes.size(); ++unit) {
            for (auto const &edge: result.edges) {
                if (edge.to == unit && edge.from < unit) {
                    result.nodes[unit].depth = std::max(result.nodes[unit].depth, result.nodes[edge.from].depth + 1);
                }
            }
        }

        auto stack_top = std::map<std::pair<LogicalQueue, std::uint32_t>, float>{};
        auto offset_in_stack = std::vector<float>(result.nodes.size());
        auto lane_extent = std::array<float, logical_queue_count>{};
        auto lane_used = std::array<bool, logical_queue_count>{};

        for (auto unit = std::size_t{0}; unit < result.nodes.size(); ++unit) {
            auto const &node = result.nodes[unit];
            auto &top = stack_top[{node.queue, node.depth}];
            offset_in_stack[unit] = top;
            top += height_of(params, input, node.group) + params.row_gap;

            auto const lane = static_cast<std::size_t>(node.queue);
            lane_used[lane] = true;
            lane_extent[lane] = std::max(lane_extent[lane], top);
        }

        auto lane_top = std::array<float, logical_queue_count>{};
        auto cursor = 0.0F;
        for (auto lane = std::size_t{0}; lane < logical_queue_count; ++lane) {
            if (!lane_used[lane]) {
                continue;
            }
            lane_top[lane] = cursor;
            cursor += lane_extent[lane] + params.lane_gap;
        }

        for (auto unit = std::size_t{0}; unit < result.nodes.size(); ++unit) {
            auto &node = result.nodes[unit];
            node.x = static_cast<float>(node.depth) * params.column_width;
            node.y = lane_top[static_cast<std::size_t>(node.queue)] + offset_in_stack[unit];
        }

        return result;
    }

}
