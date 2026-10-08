#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>

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
        bool serialize = false;
    };

    [[nodiscard]] auto compile(GraphDesc const &graph, QueueTopology const &topology,
                               CompileOptions const &options = {}) -> std::expected<CompiledGraph, FrameGraphError>;

    [[nodiscard]] auto compile(FrameGraph const &graph, QueueTopology const &topology,
                               CompileOptions const &options = {}) -> std::expected<CompiledGraph, FrameGraphError>;

    [[nodiscard]] auto declaration_hash(GraphDesc const &graph, QueueTopology const &topology,
                                        CompileOptions const &options) -> std::uint64_t;

    class PlanCache {
    public:
        [[nodiscard]] auto compile(FrameGraph const &graph, QueueTopology const &topology,
                                   CompileOptions const &options = {})
                -> std::expected<CompiledGraph const *, FrameGraphError>;

        auto invalidate() -> void { plan_.reset(); }

        [[nodiscard]] auto hits() const noexcept -> std::uint64_t { return hits_; }
        [[nodiscard]] auto misses() const noexcept -> std::uint64_t { return misses_; }

    private:
        std::optional<CompiledGraph> plan_;
        std::uint64_t hits_ = 0;
        std::uint64_t misses_ = 0;
    };

}
