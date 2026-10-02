#pragma once

#include <array>
#include <expected>

#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"
#include "rendering/frame_graph/scheduler.hxx"

namespace frame_graph {

    struct QueueTopology {
        std::array<std::uint32_t, logical_queue_count> family{};
        std::array<std::uint32_t, logical_queue_count> queue_index{};

        [[nodiscard]] constexpr auto same_queue(LogicalQueue a, LogicalQueue b) const noexcept -> bool {
            auto const i = static_cast<std::size_t>(a);
            auto const j = static_cast<std::size_t>(b);
            return family[i] == family[j] && queue_index[i] == queue_index[j];
        }

        [[nodiscard]] constexpr auto same_family(LogicalQueue a, LogicalQueue b) const noexcept -> bool {
            return family[static_cast<std::size_t>(a)] == family[static_cast<std::size_t>(b)];
        }
    };

    struct CompileOptions {
        bool async_compute = true;
        SchedulerMode scheduler = SchedulerMode::declaration_order;
        bool serialize = false; // debug: ALL_COMMANDS barriers between all passes
    };

    // Compiles the declared graph into barriers, batches, semaphore waits, ownership transfers and timestamp slots.
    // Declaration order is the schedule. A topology with one queue, or async_compute = false, puts every pass on
    // graphics.
    [[nodiscard]] auto compile(GraphDesc const &graph, QueueTopology const &topology,
                               CompileOptions const &options = {}) -> std::expected<CompiledGraph, FrameGraphError>;

    [[nodiscard]] auto compile(FrameGraph const &graph, QueueTopology const &topology,
                               CompileOptions const &options = {}) -> std::expected<CompiledGraph, FrameGraphError>;

} // namespace frame_graph
