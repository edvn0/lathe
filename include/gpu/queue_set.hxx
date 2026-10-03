#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include "core/error_context.hxx"
#include "gpu/queue_selection.hxx"
#include "gpu/submission_plan.hxx"
#include "rendering/frame_graph/compiler.hxx"

// The queues a frame is submitted to, one timeline semaphore per physical queue, and the per-frame-slot command pools
// (docs/frame-graph.md, phase 2). A timeline wait replaces the old per-frame fence: a slot is free again once every
// timeline has reached the last value that slot signalled.

struct QueueSetError {
    enum class Kind : std::uint8_t {
        device_lost,
        fatal_error,
    };

    Kind kind = Kind::fatal_error;

    std::optional<ErrorContext> context{std::nullopt};
};

struct QueueSetCreateInfo {
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE; // equals graphics_queue on the single topology

    std::uint32_t graphics_family = 0;
    std::uint32_t compute_family = 0;
    std::uint32_t compute_queue_index = 0;
};

class QueueSet {
public:
    QueueSet() = default;
    ~QueueSet();

    QueueSet(QueueSet const &) = delete;
    auto operator=(QueueSet const &) -> QueueSet & = delete;

    QueueSet(QueueSet &&) = delete;
    auto operator=(QueueSet &&) -> QueueSet & = delete;

    [[nodiscard]]
    auto initialize(QueueSetCreateInfo const &create_info) noexcept -> std::expected<void, QueueSetError>;

    auto destroy() noexcept -> void;

    // Waits until every timeline has reached the last value this slot signalled (bounded: a timeout reports
    // device_lost, like a hung fence did), then resets the slot's command pools.
    [[nodiscard]]
    auto begin_slot(std::uint32_t slot) noexcept -> std::expected<void, QueueSetError>;

    // A primary command buffer from the current slot's pool for `queue`, already begun with ONE_TIME_SUBMIT. Grows the
    // pool on demand; everything is reclaimed by the next begin_slot of the same slot.
    [[nodiscard]]
    auto command_buffer(frame_graph::LogicalQueue queue) noexcept -> std::expected<VkCommandBuffer, QueueSetError>;

    // One vkQueueSubmit2 per batch, in order. `acquire` is the binary semaphore the swapchain acquire signals and
    // `render_finished` the binary semaphore presentation waits on.
    [[nodiscard]]
    auto submit(std::span<SubmitBatch const> batches, VkSemaphore acquire,
                VkSemaphore render_finished) noexcept -> std::expected<void, QueueSetError>;

    [[nodiscard]]
    auto topology() const noexcept -> frame_graph::QueueTopology;

    // True when both logical queues are the same physical queue: one pool and one timeline serve both.
    [[nodiscard]]
    auto aliased() const noexcept -> bool {
        return physical_count_ == 1;
    }

private:
    struct PhysicalQueue {
        VkQueue queue = VK_NULL_HANDLE;
        std::uint32_t family = 0;
        VkSemaphore timeline = VK_NULL_HANDLE;
        std::uint64_t value = 0; // the last value signalled
    };

    struct FrameSlot {
        std::array<VkCommandPool, gpu_queue_count> pools{};
        std::array<std::vector<VkCommandBuffer>, gpu_queue_count> buffers{};
        std::array<std::uint32_t, gpu_queue_count> used{};
        std::array<std::uint64_t, gpu_queue_count> last_value{};
    };

    VkDevice device_ = VK_NULL_HANDLE;

    std::array<PhysicalQueue, gpu_queue_count> queues_{};
    std::size_t physical_count_ = 0;
    std::array<std::size_t, gpu_queue_count> physical_of_queue_{};

    std::uint32_t compute_queue_index_ = 0;

    std::vector<FrameSlot> slots_;
    std::uint32_t current_slot_ = 0;
};
