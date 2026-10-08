#pragma once

#include <cstdint>
#include <vector>

#include "rendering/frame_graph/frame_graph.hxx"

namespace frame_graph {

    enum class SchedulerMode : std::uint8_t { declaration_order, overlap };

    inline constexpr std::size_t overlap_window = 4;

    [[nodiscard]] auto dependency_successors(GraphDesc const &graph, std::vector<bool> const &live)
            -> std::vector<std::vector<std::size_t>>;

    [[nodiscard]] auto schedule(GraphDesc const &graph, std::vector<bool> const &live,
                                std::vector<LogicalQueue> const &queues, SchedulerMode mode)
            -> std::vector<std::uint32_t>;

}
