#include "rendering/frame_graph/scheduler.hxx"

#include <algorithm>
#include <limits>

namespace frame_graph {

    auto dependency_successors(GraphDesc const &graph, std::vector<bool> const &live)
            -> std::vector<std::vector<std::size_t>> {
        auto successors = std::vector<std::vector<std::size_t>>(graph.passes.size());
        struct History {
            std::vector<std::size_t> accessors_since_write;
            std::int64_t last_write = -1;
        };
        auto history = std::vector<History>(graph.resources.size());
        for (auto index = std::size_t{0}; index < graph.passes.size(); ++index) {
            if (!live[index]) {
                continue;
            }
            for (auto const &access: graph.passes[index].accesses) {
                auto &h = history[access.resource];
                if (h.last_write >= 0) {
                    successors[static_cast<std::size_t>(h.last_write)].push_back(index);
                }
                if (access.produces) {
                    for (auto const reader: h.accessors_since_write) {
                        successors[reader].push_back(index);
                    }
                    h.accessors_since_write.clear();
                    h.last_write = static_cast<std::int64_t>(index);
                } else {
                    h.accessors_since_write.push_back(index);
                }
            }
        }
        return successors;
    }

    auto schedule(GraphDesc const &graph, std::vector<bool> const &live, std::vector<LogicalQueue> const &queues,
                  SchedulerMode mode) -> std::vector<std::uint32_t> {
        auto order = std::vector<std::uint32_t>{};
        for (auto index = std::size_t{0}; index < graph.passes.size(); ++index) {
            if (live[index]) {
                order.push_back(static_cast<std::uint32_t>(index));
            }
        }
        if (mode == SchedulerMode::declaration_order || order.size() < 2) {
            return order;
        }

        auto const count = graph.passes.size();
        auto successors = dependency_successors(graph, live);

        auto cross_predecessors = std::vector<std::vector<std::size_t>>(count);
        for (auto source = std::size_t{0}; source < count; ++source) {
            for (auto const target: successors[source]) {
                if (queues[source] != queues[target]) {
                    cross_predecessors[target].push_back(source);
                }
            }
        }

        for (auto const fence: order) {
            auto const &pass = graph.passes[fence];
            if (!pass.pinned && !pass.legacy) {
                continue;
            }
            for (auto const other: order) {
                if (other < fence) {
                    successors[other].push_back(fence);
                } else if (other > fence) {
                    successors[fence].push_back(other);
                }
            }
        }

        auto indegree = std::vector<std::size_t>(count, 0);
        for (auto const source: order) {
            for (auto const target: successors[source]) {
                indegree[target] += 1;
            }
        }

        constexpr auto never = std::numeric_limits<std::size_t>::max();
        auto scheduled_at = std::vector<std::size_t>(count, never);
        auto result = std::vector<std::uint32_t>{};
        result.reserve(order.size());

        while (result.size() < order.size()) {
            auto best = std::int64_t{-1};
            auto best_delayed = std::int64_t{-1};
            for (auto const candidate: order) {
                if (scheduled_at[candidate] != never || indegree[candidate] != 0) {
                    continue;
                }
                auto delayed = false;
                for (auto const producer: cross_predecessors[candidate]) {
                    delayed = delayed || (scheduled_at[producer] != never &&
                                          result.size() - scheduled_at[producer] < overlap_window);
                }
                auto &slot = delayed ? best_delayed : best;
                if (slot < 0) {
                    slot = static_cast<std::int64_t>(candidate);
                }
            }
            auto const pick = static_cast<std::size_t>(best >= 0 ? best : best_delayed);
            scheduled_at[pick] = result.size();
            result.push_back(static_cast<std::uint32_t>(pick));
            for (auto const target: successors[pick]) {
                indegree[target] -= 1;
            }
        }
        return result;
    }

}
