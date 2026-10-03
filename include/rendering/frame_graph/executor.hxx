#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include <tracy/TracyVulkan.hpp>

#include "gpu/queue_set.hxx"
#include "gpu/submission_plan.hxx"
#include "rendering/frame_graph/aliasing.hxx"
#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"
#include "rendering/frame_graph/pass_profiler.hxx"
#include "rendering/frame_graph/vk_translate.hxx"

// The recording backend (docs/frame-graph.md, phase 3): walks a compiled graph and records its batches into command
// buffers.
namespace frame_graph {

    struct ExecuteError {
        std::string message;
    };

    struct ExecuteInfo {
        GraphDesc const &graph;
        CompiledGraph const &compiled;

        // The record lambdas, in declaration order (FrameGraph::records()).
        std::span<RecordFn> records;

        // The handles behind every resource the barriers and passes touch.
        PhysicalResources const &resources;

        // Where transients share memory: the dependencies to record before the pass that first uses recycled memory.
        // Null when nothing is aliased.
        TransientPlan const *transients = nullptr;

        // Batches after the first get fresh command buffers from here.
        QueueSet &queue_set;
        PassProfiler &profiler;

        // Tracy's GPU context per logical queue; null skips the GPU zone.
        std::array<tracy::VkCtx *, logical_queue_count> tracy_contexts{};

        // The graphics command buffer the frame has already begun (and may have recorded into). Batch 0 appends to it
        // and leaves it open, so the owner of the frame ends it. Every later batch's buffer is ended here.
        VkCommandBuffer prologue = VK_NULL_HANDLE;

        std::uint32_t frame_index = 0;
    };

    // For each batch, in submission order: acquire barriers, then every pass (its barriers, a CPU and GPU Tracy zone, a
    // begin timestamp, the record lambda, an end timestamp), then releases and the epilogue. Returns the batches to
    // submit, with waits pointing into `compiled`, which must outlive the submission.
    //
    // A pass's record lambda runs even if an earlier one reported a failure through its own channel; the lambda
    // decides what to do. The executor only fails when it cannot translate a barrier or get a command buffer.
    [[nodiscard]] auto record(ExecuteInfo const &info) -> std::expected<std::vector<SubmitBatch>, ExecuteError>;

} // namespace frame_graph
