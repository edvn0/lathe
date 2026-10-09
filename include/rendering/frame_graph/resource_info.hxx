#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "rendering/frame_graph/view.hxx"

namespace frame_graph {

    // Who touches a resource and where it lives, from a snapshot. Pass indices follow declaration order.
    struct ResourceInfo {
        std::vector<std::uint32_t> producers;
        std::vector<std::uint32_t> consumers;
        std::optional<std::uint32_t> first_pass;
        std::optional<std::uint32_t> last_pass;

        std::optional<TransientPlacement> placement;
        // Other transients whose memory overlaps this one's: only possible when aliasing is on.
        std::vector<std::uint32_t> shares_memory_with;
    };

    [[nodiscard]] auto describe_resource(FrameGraphView const &view, std::uint32_t resource) -> ResourceInfo;

}
