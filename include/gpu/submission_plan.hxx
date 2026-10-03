#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "rendering/frame_graph/compiled_graph.hxx"

// Turns a frame's batches into timeline submissions. Pure: no Vulkan calls, so the value arithmetic is unit-tested
// without a device; QueueSet makes the calls from the result.

inline constexpr std::size_t gpu_queue_count = frame_graph::logical_queue_count;

// One vkQueueSubmit2. `waits` come straight from a compiled frame_graph::Batch; `signal_index` is relative to the
// frame and QueueSet turns it into an absolute timeline value.
struct SubmitBatch {
    frame_graph::LogicalQueue queue = frame_graph::LogicalQueue::graphics;

    // Null for a batch with no commands, which only waits and signals.
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;

    std::span<frame_graph::SemaphoreWait const> waits;
    std::uint32_t signal_index = 0;

    bool waits_swapchain_acquire = false;
    VkPipelineStageFlags2 swapchain_wait_stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    bool signals_render_finished = false;
};

struct TimelineWait {
    std::size_t timeline = 0; // index of a physical queue's timeline
    std::uint64_t value = 0;
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    auto operator==(TimelineWait const &) const -> bool = default;
};

struct PlannedSubmit {
    std::size_t batch = 0; // index into the batches that were planned
    std::vector<TimelineWait> waits;
    bool waits_acquire = false;
    VkPipelineStageFlags2 acquire_stages = VK_PIPELINE_STAGE_2_NONE;
    std::size_t signal_timeline = 0;
    std::uint64_t signal_value = 0;
    bool signals_render_finished = false;
};

struct SubmissionPlan {
    std::vector<PlannedSubmit> submits;

    // The value of the last signal on each timeline once the plan has run.
    std::array<std::uint64_t, gpu_queue_count> timeline_values{};
};

enum class SubmissionPlanError : std::uint8_t {
    signal_out_of_order, // a batch's signal_index is not the next one for its queue
    wait_before_signal, // a batch waits for a signal that no earlier batch makes
};

// Turns relative signal indices into absolute timeline values. `timeline_values` is the last value signalled on each
// timeline so far, and `timeline_of_queue` maps a logical queue to its physical timeline (both logical queues share
// timeline 0 when they are the same physical queue). Pure: the Vulkan calls are made from the result.
[[nodiscard]] auto plan_submissions(std::span<SubmitBatch const> batches,
                                    std::array<std::uint64_t, gpu_queue_count> timeline_values,
                                    std::array<std::size_t, gpu_queue_count> timeline_of_queue)
        -> std::expected<SubmissionPlan, SubmissionPlanError>;
