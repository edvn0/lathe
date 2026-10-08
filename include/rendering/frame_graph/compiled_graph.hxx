#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "rendering/frame_graph/rendering_desc.hxx"
#include "rendering/frame_graph/types.hxx"

namespace frame_graph {

    enum class OwnershipOp : std::uint8_t { none, release, acquire };

    struct ImageBarrier {
        std::uint32_t resource = 0;
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

    struct BarrierSet {
        std::vector<ImageBarrier> images;
        std::vector<BufferBarrier> buffers;
        std::vector<MemoryBarrier> memory;

        [[nodiscard]] auto empty() const noexcept -> bool {
            return images.empty() && buffers.empty() && memory.empty();
        }
    };

    struct CompiledPass {
        std::uint32_t pass = 0;
        BarrierSet before;
        std::uint32_t timestamp_slot = 0;
    };

    struct SemaphoreWait {
        LogicalQueue queue = LogicalQueue::graphics;
        std::uint32_t signal_index = 0;
        VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;

        auto operator==(SemaphoreWait const &) const -> bool = default;
    };

    struct Batch {
        LogicalQueue queue = LogicalQueue::graphics;
        BarrierSet acquires;
        std::vector<CompiledPass> passes;
        BarrierSet releases;
        BarrierSet epilogue;
        std::vector<SemaphoreWait> waits;
        std::uint32_t signal_index = 0;
        bool is_prologue = false;
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
        std::vector<Batch> batches;
        std::vector<OwnershipTransfer> transfers;
        std::array<std::uint32_t, logical_queue_count> signal_count{};
        std::vector<bool> pass_culled;
        std::array<std::vector<std::uint32_t>, logical_queue_count> timestamp_passes;
        std::vector<LogicalQueue> pass_queue;
        std::vector<std::uint32_t> schedule;
        std::uint64_t hash = 0;
    };

}
