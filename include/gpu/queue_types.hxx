#pragma once

#include <volk.h>

#include <array>
#include <cstddef>
#include <cstdint>

// The queue vocabulary shared by the frame graph (which plans work per logical queue) and the gpu layer (which maps
// logical queues onto Vulkan queues and submits). Lives here so gpu does not depend on rendering/frame_graph.

enum class LogicalQueue : std::uint8_t { graphics, compute };
inline constexpr std::size_t logical_queue_count = 2;

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

struct SemaphoreWait {
    LogicalQueue queue = LogicalQueue::graphics;
    std::uint32_t signal_index = 0;
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;

    auto operator==(SemaphoreWait const &) const -> bool = default;
};
