#pragma once

#include "core/forward.hxx"
#include "gpu/buffer.hxx"

#include <volk.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Captures the composited swapchain image to a PNG in three stages:
//
//   1. record()       records a copy into a mapped readback buffer for the current frame slot.
//   2. try_resolve()  when that slot comes round again its fence has completed; hands the buffer to a worker.
//   3. worker thread  invalidates if needed, copies the pixels out, releases the slot, then converts and
//                     encodes the PNG.
//
// request() is thread-safe; record() and try_resolve() are render-thread only.
class ScreenshotCapture {
public:
    ScreenshotCapture() = default;
    ~ScreenshotCapture();

    ScreenshotCapture(ScreenshotCapture const &) = delete;
    auto operator=(ScreenshotCapture const &) -> ScreenshotCapture & = delete;

    ScreenshotCapture(ScreenshotCapture &&) = delete;
    auto operator=(ScreenshotCapture &&) -> ScreenshotCapture & = delete;

    auto request() noexcept -> void { requested_.store(true, std::memory_order_relaxed); }

    // Records the copy if a capture is pending and the slot is free, leaving the image in PRESENT_SRC_KHR. Returns
    // true in that case, and the caller must skip its own present transition.
    [[nodiscard]]
    auto record(VulkanContext &ctx, VkCommandBuffer command_buffer, VkImage image, VkFormat format, VkExtent2D extent,
                std::uint32_t frame_index) -> bool;

    // Call once frame_index's fence has been waited on. Hands a completed copy to the worker.
    auto try_resolve(std::uint32_t frame_index) -> void;

    // Waits for pending PNG writes, then frees the readback buffers. The allocator must still be alive.
    auto close() noexcept -> void;

private:
    struct ReadbackSlot {
        Buffer buffer;
        std::atomic<bool> cpu_busy{false};

        bool gpu_pending = false;

        VkExtent2D extent{};
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::size_t byte_size = 0;
    };

    [[nodiscard]]
    auto get_or_create_slot(std::uint32_t frame_index) -> ReadbackSlot &;

    std::atomic<bool> requested_{false};

    std::vector<std::unique_ptr<ReadbackSlot>> slots_;

    std::mutex mutex_;
    std::condition_variable cv_;
};
