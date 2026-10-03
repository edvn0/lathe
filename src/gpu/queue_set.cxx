#include "gpu/queue_set.hxx"

#include <string_view>

#include "core/config.hxx"
#include "core/logger.hxx"
#include "gpu/device_wait.hxx"

namespace {

    constexpr auto queue_index(frame_graph::LogicalQueue queue) noexcept -> std::size_t {
        return static_cast<std::size_t>(queue);
    }

    auto make_error(QueueSetError::Kind kind, std::string_view message) noexcept -> QueueSetError {
        return QueueSetError{
                .kind = kind,
                .context = ErrorContext{.message = FlyString{message}},
        };
    }

    auto make_vk_error(std::string_view operation, VkResult result) noexcept -> QueueSetError {
        error("{} failed with VkResult {}", operation, static_cast<int>(result));

        return QueueSetError{
                .kind = is_device_failure(result) ? QueueSetError::Kind::device_lost : QueueSetError::Kind::fatal_error,
                .context =
                        ErrorContext{
                                .message = FlyString{operation},
                                .vk_result = result,
                        },
        };
    }

} // namespace

QueueSet::~QueueSet() { destroy(); }

auto QueueSet::initialize(QueueSetCreateInfo const &create_info) noexcept -> std::expected<void, QueueSetError> {
    destroy();

    if (create_info.device == VK_NULL_HANDLE || create_info.graphics_queue == VK_NULL_HANDLE ||
        create_info.compute_queue == VK_NULL_HANDLE) {
        return std::unexpected(make_error(QueueSetError::Kind::fatal_error, "invalid queue set create info"));
    }

    device_ = create_info.device;
    compute_queue_index_ = create_info.compute_queue_index;

    queues_[0].queue = create_info.graphics_queue;
    queues_[0].family = create_info.graphics_family;
    physical_of_queue_[queue_index(frame_graph::LogicalQueue::graphics)] = 0;

    if (create_info.compute_queue == create_info.graphics_queue) {
        // One physical queue serves both logical queues: the compute pool and timeline are the graphics ones.
        physical_count_ = 1;
        physical_of_queue_[queue_index(frame_graph::LogicalQueue::compute)] = 0;
    } else {
        physical_count_ = 2;
        queues_[1].queue = create_info.compute_queue;
        queues_[1].family = create_info.compute_family;
        physical_of_queue_[queue_index(frame_graph::LogicalQueue::compute)] = 1;
    }

    VkSemaphoreTypeCreateInfo const timeline_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = nullptr,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = 0,
    };
    VkSemaphoreCreateInfo const semaphore_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &timeline_info,
            .flags = 0,
    };

    slots_.resize(frames_in_flight);

    for (auto physical = std::size_t{0}; physical < physical_count_; ++physical) {
        auto const result = vkCreateSemaphore(device_, &semaphore_info, nullptr, &queues_[physical].timeline);
        if (result != VK_SUCCESS) {
            auto failure = make_vk_error("vkCreateSemaphore(timeline)", result);
            destroy();
            return std::unexpected(std::move(failure));
        }

        for (auto &slot: slots_) {
            // Transient: every buffer is recorded once per use and the pool is reset wholesale each time the slot is
            // reused.
            VkCommandPoolCreateInfo const pool_info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                    .queueFamilyIndex = queues_[physical].family,
            };

            auto const pool_result = vkCreateCommandPool(device_, &pool_info, nullptr, &slot.pools[physical]);
            if (pool_result != VK_SUCCESS) {
                auto failure = make_vk_error("vkCreateCommandPool(queue set)", pool_result);
                destroy();
                return std::unexpected(std::move(failure));
            }
        }
    }

    return {};
}

auto QueueSet::destroy() noexcept -> void {
    if (device_ != VK_NULL_HANDLE) {
        // Destroying a pool frees its buffers. The caller has already waited for the device to go idle.
        for (auto &slot: slots_) {
            for (auto &pool: slot.pools) {
                if (pool != VK_NULL_HANDLE) {
                    vkDestroyCommandPool(device_, pool, nullptr);
                    pool = VK_NULL_HANDLE;
                }
            }
        }

        for (auto &queue: queues_) {
            if (queue.timeline != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, queue.timeline, nullptr);
                queue.timeline = VK_NULL_HANDLE;
            }
        }
    }

    slots_.clear();
    queues_ = {};
    physical_count_ = 0;
    physical_of_queue_ = {};
    compute_queue_index_ = 0;
    current_slot_ = 0;
    device_ = VK_NULL_HANDLE;
}

auto QueueSet::begin_slot(std::uint32_t slot) noexcept -> std::expected<void, QueueSetError> {
    if (device_ == VK_NULL_HANDLE || slot >= slots_.size()) {
        return std::unexpected(make_error(QueueSetError::Kind::fatal_error, "invalid frame slot"));
    }

    current_slot_ = slot;
    auto &frame = slots_[slot];

    // A bounded wait so a stuck GPU surfaces as a loss instead of hanging shutdown.
    constexpr std::uint64_t slot_wait_timeout_ns = 2'000'000'000ULL;

    for (auto physical = std::size_t{0}; physical < physical_count_; ++physical) {
        auto const value = frame.last_value[physical];
        if (value == 0) {
            continue;
        }

        VkSemaphoreWaitInfo const wait_info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .pNext = nullptr,
                .flags = 0,
                .semaphoreCount = 1,
                .pSemaphores = &queues_[physical].timeline,
                .pValues = &value,
        };

        // The wait is bounded, so a GPU that hangs rather than faults shows up as a timeout. Report it like a loss:
        // the device is unusable either way and the user gets the same restart notice.
        auto const result = vkWaitSemaphores(device_, &wait_info, slot_wait_timeout_ns);
        if (result != VK_SUCCESS) {
            return std::unexpected(make_vk_error("vkWaitSemaphores", result));
        }
    }

    for (auto physical = std::size_t{0}; physical < physical_count_; ++physical) {
        auto const result = vkResetCommandPool(device_, frame.pools[physical], 0);
        if (result != VK_SUCCESS) {
            return std::unexpected(make_vk_error("vkResetCommandPool", result));
        }
        frame.used[physical] = 0;
    }

    return {};
}

auto QueueSet::command_buffer(frame_graph::LogicalQueue queue) noexcept
        -> std::expected<VkCommandBuffer, QueueSetError> {
    if (device_ == VK_NULL_HANDLE || current_slot_ >= slots_.size()) {
        return std::unexpected(make_error(QueueSetError::Kind::fatal_error, "queue set is not initialized"));
    }

    auto const physical = physical_of_queue_[queue_index(queue)];
    auto &frame = slots_[current_slot_];

    if (frame.used[physical] == frame.buffers[physical].size()) {
        VkCommandBufferAllocateInfo const allocate_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = frame.pools[physical],
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
        };

        VkCommandBuffer buffer = VK_NULL_HANDLE;
        auto const result = vkAllocateCommandBuffers(device_, &allocate_info, &buffer);
        if (result != VK_SUCCESS) {
            return std::unexpected(make_vk_error("vkAllocateCommandBuffers", result));
        }
        frame.buffers[physical].push_back(buffer);
    }

    auto const buffer = frame.buffers[physical][frame.used[physical]];
    frame.used[physical] += 1;

    VkCommandBufferBeginInfo const begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
    };

    auto const result = vkBeginCommandBuffer(buffer, &begin_info);
    if (result != VK_SUCCESS) {
        return std::unexpected(make_vk_error("vkBeginCommandBuffer", result));
    }

    return buffer;
}

auto QueueSet::submit(std::span<SubmitBatch const> batches, VkSemaphore acquire,
                      VkSemaphore render_finished) noexcept -> std::expected<void, QueueSetError> {
    if (device_ == VK_NULL_HANDLE || current_slot_ >= slots_.size()) {
        return std::unexpected(make_error(QueueSetError::Kind::fatal_error, "queue set is not initialized"));
    }

    auto timeline_values = std::array<std::uint64_t, gpu_queue_count>{};
    for (auto physical = std::size_t{0}; physical < physical_count_; ++physical) {
        timeline_values[physical] = queues_[physical].value;
    }

    auto const plan = plan_submissions(batches, timeline_values, physical_of_queue_);
    if (!plan) {
        return std::unexpected(make_error(QueueSetError::Kind::fatal_error,
                                          plan.error() == SubmissionPlanError::signal_out_of_order
                                                  ? "submit batches signal out of order"
                                                  : "a submit batch waits for a signal no earlier batch makes"));
    }

    for (auto const &planned: plan->submits) {
        auto const &batch = batches[planned.batch];
        auto const &queue = queues_[physical_of_queue_[queue_index(batch.queue)]];

        std::vector<VkSemaphoreSubmitInfo> waits;
        if (planned.waits_acquire) {
            waits.push_back(VkSemaphoreSubmitInfo{
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .pNext = nullptr,
                    .semaphore = acquire,
                    .value = 0,
                    .stageMask = planned.acquire_stages,
                    .deviceIndex = 0,
            });
        }
        for (auto const &wait: planned.waits) {
            waits.push_back(VkSemaphoreSubmitInfo{
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .pNext = nullptr,
                    .semaphore = queues_[wait.timeline].timeline,
                    .value = wait.value,
                    .stageMask = wait.stages,
                    .deviceIndex = 0,
            });
        }

        std::array<VkSemaphoreSubmitInfo, 2> signals{};
        auto signal_count = std::uint32_t{0};
        signals[signal_count++] = VkSemaphoreSubmitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .semaphore = queues_[planned.signal_timeline].timeline,
                .value = planned.signal_value,
                .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .deviceIndex = 0,
        };
        if (planned.signals_render_finished) {
            signals[signal_count++] = VkSemaphoreSubmitInfo{
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .pNext = nullptr,
                    .semaphore = render_finished,
                    .value = 0,
                    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    .deviceIndex = 0,
            };
        }

        VkCommandBufferSubmitInfo const command_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                .pNext = nullptr,
                .commandBuffer = batch.command_buffer,
                .deviceMask = 0,
        };

        VkSubmitInfo2 const submit_info{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                .pNext = nullptr,
                .flags = 0,
                .waitSemaphoreInfoCount = static_cast<std::uint32_t>(waits.size()),
                .pWaitSemaphoreInfos = waits.data(),
                .commandBufferInfoCount = batch.command_buffer != VK_NULL_HANDLE ? 1U : 0U,
                .pCommandBufferInfos = batch.command_buffer != VK_NULL_HANDLE ? &command_info : nullptr,
                .signalSemaphoreInfoCount = signal_count,
                .pSignalSemaphoreInfos = signals.data(),
        };

        auto const result = vkQueueSubmit2(queue.queue, 1, &submit_info, VK_NULL_HANDLE);
        if (result != VK_SUCCESS) {
            return std::unexpected(make_vk_error("vkQueueSubmit2", result));
        }
    }

    // Every timeline that advanced is now what this slot must wait for before it is reused.
    auto &frame = slots_[current_slot_];
    for (auto physical = std::size_t{0}; physical < physical_count_; ++physical) {
        if (plan->timeline_values[physical] != queues_[physical].value) {
            queues_[physical].value = plan->timeline_values[physical];
            frame.last_value[physical] = plan->timeline_values[physical];
        }
    }

    return {};
}

auto QueueSet::topology() const noexcept -> frame_graph::QueueTopology {
    auto topology = frame_graph::QueueTopology{};
    topology.family[queue_index(frame_graph::LogicalQueue::graphics)] = queues_[0].family;
    topology.queue_index[queue_index(frame_graph::LogicalQueue::graphics)] = 0;
    topology.family[queue_index(frame_graph::LogicalQueue::compute)] =
            queues_[physical_of_queue_[queue_index(frame_graph::LogicalQueue::compute)]].family;
    topology.queue_index[queue_index(frame_graph::LogicalQueue::compute)] = aliased() ? 0 : compute_queue_index_;
    return topology;
}
