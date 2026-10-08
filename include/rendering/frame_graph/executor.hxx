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

namespace frame_graph {

    struct ExecuteError {
        std::string message;
    };

    struct ExecuteInfo {
        GraphDesc const &graph;
        CompiledGraph const &compiled;

        std::span<RecordFn> records;

        PhysicalResources const &resources;

        TransientPlan const *transients = nullptr;

        QueueSet &queue_set;
        PassProfiler &profiler;

        std::array<tracy::VkCtx *, logical_queue_count> tracy_contexts{};

        VkCommandBuffer prologue = VK_NULL_HANDLE;

        std::uint32_t frame_index = 0;
    };

    [[nodiscard]] auto record(ExecuteInfo const &info) -> std::expected<std::vector<SubmitBatch>, ExecuteError>;

}
