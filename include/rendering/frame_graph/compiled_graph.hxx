#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "rendering/frame_graph/types.hxx"

namespace frame_graph {

    enum class OwnershipOp : std::uint8_t { none, release, acquire };

    struct ImageBarrier {
        std::uint32_t resource = 0; // resource slot
        VkPipelineStageFlags2 src_stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 src_access = VK_ACCESS_2_NONE;
        VkPipelineStageFlags2 dst_stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 dst_access = VK_ACCESS_2_NONE;
        VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint32_t src_family = VK_QUEUE_FAMILY_IGNORED;
        std::uint32_t dst_family = VK_QUEUE_FAMILY_IGNORED;
        OwnershipOp op = OwnershipOp::none;

        auto operator==(ImageBarrier const &) const -> bool = default;
    };

    // Always whole-buffer.
    struct BufferBarrier {
        std::uint32_t resource = 0;
        VkPipelineStageFlags2 src_stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 src_access = VK_ACCESS_2_NONE;
        VkPipelineStageFlags2 dst_stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 dst_access = VK_ACCESS_2_NONE;
        std::uint32_t src_family = VK_QUEUE_FAMILY_IGNORED;
        std::uint32_t dst_family = VK_QUEUE_FAMILY_IGNORED;
        OwnershipOp op = OwnershipOp::none;

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

    // Wait for another queue's timeline to reach `signal_index` (relative to the frame) before this batch starts.
    struct SemaphoreWait {
        LogicalQueue queue = LogicalQueue::graphics;
        std::uint32_t signal_index = 0;
        VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;

        auto operator==(SemaphoreWait const &) const -> bool = default;
    };

    struct Batch {
        LogicalQueue queue = LogicalQueue::graphics;
        BarrierSet acquires; // recorded first: ownership acquires on entry
        std::vector<CompiledPass> passes;
        BarrierSet releases; // recorded after the last pass: ownership releases
        BarrierSet epilogue; // import exit transitions, last graphics batch only
        std::vector<SemaphoreWait> waits; // at most one per other queue
        std::uint32_t signal_index = 0; // every batch signals its queue's timeline
        bool is_prologue = false; // the graphics batch that begins the frame
        bool waits_swapchain_acquire = false;
        VkPipelineStageFlags2 swapchain_wait_stages = VK_PIPELINE_STAGE_2_NONE;
        bool signals_render_finished = false;
    };

    struct OwnershipTransfer {
        std::uint32_t resource = 0;
        bool is_image = false;
        LogicalQueue from = LogicalQueue::graphics;
        LogicalQueue to = LogicalQueue::graphics;
        std::uint32_t release_batch = 0;
        std::uint32_t acquire_batch = 0;
        VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct CompiledGraph {
        std::vector<Batch> batches; // submission order: every wait refers to an earlier batch
        std::vector<OwnershipTransfer> transfers;
        std::array<std::uint32_t, logical_queue_count> signal_count{};
        std::vector<bool> pass_culled;
        std::array<std::vector<std::uint32_t>, logical_queue_count> timestamp_passes; // slot -> pass
        std::vector<LogicalQueue> pass_queue;
        std::vector<std::uint32_t> schedule; // live passes in execution order, as declaration indices
        std::uint64_t hash = 0;
    };

} // namespace frame_graph
