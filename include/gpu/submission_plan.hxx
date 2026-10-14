#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "gpu/queue_types.hxx"

inline constexpr std::size_t gpu_queue_count = logical_queue_count;

struct SubmitBatch {
    LogicalQueue queue = LogicalQueue::graphics;

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;

    std::span<SemaphoreWait const> waits;
    std::uint32_t signal_index = 0;

    bool waits_swapchain_acquire = false;
    VkPipelineStageFlags2 swapchain_wait_stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    bool signals_render_finished = false;
};

struct TimelineWait {
    std::size_t timeline = 0;
    std::uint64_t value = 0;
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    auto operator==(TimelineWait const &) const -> bool = default;
};

struct PlannedSubmit {
    std::size_t batch = 0;
    std::vector<TimelineWait> waits;
    bool waits_acquire = false;
    VkPipelineStageFlags2 acquire_stages = VK_PIPELINE_STAGE_2_NONE;
    std::size_t signal_timeline = 0;
    std::uint64_t signal_value = 0;
    bool signals_render_finished = false;
};

struct SubmissionPlan {
    std::vector<PlannedSubmit> submits;

    std::array<std::uint64_t, gpu_queue_count> timeline_values{};
};

enum class SubmissionPlanError : std::uint8_t {
    signal_out_of_order,
    wait_before_signal,
};

[[nodiscard]] auto plan_submissions(std::span<SubmitBatch const> batches,
                                    std::array<std::uint64_t, gpu_queue_count> timeline_values,
                                    std::array<std::size_t, gpu_queue_count> timeline_of_queue)
        -> std::expected<SubmissionPlan, SubmissionPlanError>;
