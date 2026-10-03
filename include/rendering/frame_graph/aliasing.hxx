#pragma once

#include <volk.h>

#include <cstdint>
#include <span>
#include <vector>

#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"

// Transient memory planning (docs/frame-graph.md, phase 5). Pure: it reads a compiled plan and the size of every
// transient, and decides which transients share memory. The caller turns that into allocations and aliasing images.
namespace frame_graph {

    // What creating a transient image needs, from vkGetDeviceImageMemoryRequirements. Indexed by resource slot, so the
    // planner needs no Vulkan; slots that are not transient images stay empty (size 0).
    struct MemoryRequirement {
        std::uint64_t size = 0;
        std::uint64_t alignment = 1;
        std::uint32_t memory_type_bits = ~0U;
    };

    // One device memory allocation, shared by every transient placed in it.
    struct TransientBlock {
        std::uint64_t size = 0;
        std::uint64_t alignment = 1;
        std::uint32_t memory_type_bits = ~0U; // the types every placed transient accepts
    };

    struct TransientPlacement {
        std::uint32_t resource = 0; // resource slot
        std::uint32_t block = 0;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };

    // A memory dependency to record before `pass`: the transient that first uses memory another transient used
    // earlier must wait for those accesses, and the contents it finds there are not its own (its first-use layout
    // transition already discards them).
    struct AliasingBarrier {
        std::uint32_t pass = 0; // declaration index
        MemoryBarrier barrier;
    };

    struct TransientPlan {
        std::vector<TransientBlock> blocks;
        std::vector<TransientPlacement> placements; // one per live transient image, in resource slot order
        std::vector<AliasingBarrier> barriers;

        std::uint64_t total_bytes = 0; // the blocks' sizes
        std::uint64_t unaliased_bytes = 0; // what every transient would take with a block of its own

        [[nodiscard]] auto placement_of(std::uint32_t resource) const noexcept -> TransientPlacement const * {
            for (auto const &placement: placements) {
                if (placement.resource == resource) {
                    return &placement;
                }
            }
            return nullptr;
        }
    };

    // Whether every access of `first` happens before every access of `second` in `compiled`: on one queue by schedule
    // order, across queues through the semaphore waits (a wait on a timeline value orders the whole earlier prefix of
    // the signalling queue before the waiting batch). Exposed so tests can check the planner against it.
    [[nodiscard]] auto transients_disjoint(GraphDesc const &graph, CompiledGraph const &compiled,
                                           std::uint32_t first_resource, std::uint32_t second_resource) -> bool;

    // Packs the live transient images of `compiled`. Two transients may overlap in memory only if one's accesses all
    // happen before the other's (see transients_disjoint), so the answer is the same on every queue topology. Greedy,
    // largest first: each goes at the lowest aligned offset of the first block (of a compatible memory type) where it
    // overlaps nothing it conflicts with, else in a block of its own. With `alias` false every transient gets its
    // own block.
    [[nodiscard]] auto plan_transients(GraphDesc const &graph, CompiledGraph const &compiled,
                                       std::span<MemoryRequirement const> requirements, bool alias) -> TransientPlan;

} // namespace frame_graph
