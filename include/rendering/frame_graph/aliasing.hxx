#pragma once

#include <volk.h>

#include <cstdint>
#include <span>
#include <vector>

#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"

namespace frame_graph {

    struct MemoryRequirement {
        std::uint64_t size = 0;
        std::uint64_t alignment = 1;
        std::uint32_t memory_type_bits = ~0U;
    };

    struct TransientBlock {
        std::uint64_t size = 0;
        std::uint64_t alignment = 1;
        std::uint32_t memory_type_bits = ~0U;
    };

    struct TransientPlacement {
        std::uint32_t resource = 0;
        std::uint32_t block = 0;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };

    struct AliasingBarrier {
        std::uint32_t pass = 0;
        MemoryBarrier barrier;
    };

    struct TransientPlan {
        std::vector<TransientBlock> blocks;
        std::vector<TransientPlacement> placements;
        std::vector<AliasingBarrier> barriers;

        std::uint64_t total_bytes = 0;
        std::uint64_t unaliased_bytes = 0;

        [[nodiscard]] auto placement_of(std::uint32_t resource) const noexcept -> TransientPlacement const * {
            for (auto const &placement: placements) {
                if (placement.resource == resource) {
                    return &placement;
                }
            }
            return nullptr;
        }
    };

    [[nodiscard]] auto transient_usage(GraphDesc const &graph, CompiledGraph const &compiled, std::uint32_t resource)
            -> VkImageUsageFlags;

    [[nodiscard]] auto transients_disjoint(GraphDesc const &graph, CompiledGraph const &compiled,
                                           std::uint32_t first_resource, std::uint32_t second_resource) -> bool;

    [[nodiscard]] auto plan_transients(GraphDesc const &graph, CompiledGraph const &compiled,
                                       std::span<MemoryRequirement const> requirements, bool alias) -> TransientPlan;

}
