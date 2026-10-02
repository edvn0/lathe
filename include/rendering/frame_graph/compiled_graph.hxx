#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "rendering/frame_graph/types.hxx"

namespace frame_graph {

    struct ImageBarrier {
        std::uint32_t resource = 0; // resource slot
        VkPipelineStageFlags2 src_stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 src_access = VK_ACCESS_2_NONE;
        VkPipelineStageFlags2 dst_stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 dst_access = VK_ACCESS_2_NONE;
        VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;

        auto operator==(ImageBarrier const &) const -> bool = default;
    };

    // Always whole-buffer.
    struct BufferBarrier {
        std::uint32_t resource = 0;
        VkPipelineStageFlags2 src_stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 src_access = VK_ACCESS_2_NONE;
        VkPipelineStageFlags2 dst_stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 dst_access = VK_ACCESS_2_NONE;

        auto operator==(BufferBarrier const &) const -> bool = default;
    };

    struct MemoryBarrier {
        VkPipelineStageFlags2 src_stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 src_access = VK_ACCESS_2_NONE;
        VkPipelineStageFlags2 dst_stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 dst_access = VK_ACCESS_2_NONE;

        auto operator==(MemoryBarrier const &) const -> bool = default;
    };

    // Everything one vkCmdPipelineBarrier2 records.
    struct BarrierSet {
        std::vector<ImageBarrier> images;
        std::vector<BufferBarrier> buffers;
        std::vector<MemoryBarrier> memory;

        [[nodiscard]] auto empty() const noexcept -> bool {
            return images.empty() && buffers.empty() && memory.empty();
        }
    };

    struct CompiledPass {
        std::uint32_t pass = 0; // index into GraphDesc::passes
        BarrierSet before;
        std::uint32_t timestamp_slot = 0; // begin = 2 * slot, end = 2 * slot + 1, in this queue's pool
    };

    struct Batch {
        LogicalQueue queue = LogicalQueue::graphics;
        std::vector<CompiledPass> passes;
        BarrierSet epilogue; // import exit transitions, recorded after the last pass
        bool waits_swapchain_acquire = false;
        VkPipelineStageFlags2 swapchain_wait_stages = VK_PIPELINE_STAGE_2_NONE;
        bool signals_render_finished = false;
    };

    struct CompiledGraph {
        std::vector<Batch> batches; // submission order
        std::vector<bool> pass_culled;
        std::array<std::vector<std::uint32_t>, logical_queue_count> timestamp_passes; // slot -> pass
        std::vector<LogicalQueue> pass_queue;
        std::uint64_t hash = 0;
    };

} // namespace frame_graph
