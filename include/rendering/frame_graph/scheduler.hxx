#pragma once

#include <cstdint>
#include <vector>

#include "rendering/frame_graph/frame_graph.hxx"

namespace frame_graph {

    // declaration_order: the declared order is the schedule (phases 0 to 5 compile this way, so output stays
    // bit-identical). overlap: a list scheduler may hoist independent passes into the gap between a cross-queue
    // producer and its consumer.
    enum class SchedulerMode : std::uint8_t { declaration_order, overlap };

    // A consumer of a pass on the other queue is held back while that producer was scheduled fewer than this many
    // passes ago, so that independent passes can fill the gap.
    inline constexpr std::size_t overlap_window = 4;

    // For each live pass, the later live passes that depend on it: they touch a shared resource and at least one of
    // the two writes it. Indexed by declaration index; every successor has a higher index than its source.
    [[nodiscard]] auto dependency_successors(GraphDesc const &graph,
                                             std::vector<bool> const &live) -> std::vector<std::vector<std::size_t>>;

    // The execution order of the live passes as declaration indices. Always a topological order of the dependency
    // graph. `pinned` and `legacy()` passes never move and are never crossed. Ties break by declaration index, so the
    // result is deterministic.
    [[nodiscard]] auto schedule(GraphDesc const &graph, std::vector<bool> const &live,
                                std::vector<LogicalQueue> const &queues,
                                SchedulerMode mode) -> std::vector<std::uint32_t>;

} // namespace frame_graph
