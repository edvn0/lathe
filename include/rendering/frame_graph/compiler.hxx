#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <list>
#include <optional>

#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"
#include "rendering/frame_graph/scheduler.hxx"

namespace frame_graph {

    using ::QueueTopology;

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

        auto invalidate() -> void { plans_.clear(); }

        [[nodiscard]] auto hits() const noexcept -> std::uint64_t { return hits_; }
        [[nodiscard]] auto misses() const noexcept -> std::uint64_t { return misses_; }

    private:
        // Graphs that alternate between a few shapes (a pass that runs every other frame) each keep their plan. A list so
        // a plan's address stays valid while others move; the most recently used is first.
        static constexpr std::size_t max_plans = 8;
        std::list<CompiledGraph> plans_;
        std::uint64_t hits_ = 0;
        std::uint64_t misses_ = 0;
    };

}
