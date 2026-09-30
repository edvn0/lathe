#pragma once

#include "core/forward.hxx"
#include "gpu/buffer.hxx"

#include <volk.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

// What a screenshot captures.
enum class ScreenshotSource : std::uint8_t {
    window,   // The composited swapchain image: the whole app including the UI.
    viewport, // The editor's viewport target: just the rendered scene.
};

// An image to copy from, with the layout and sync it is in when record() is called and the ones to leave it in.
struct ScreenshotImage {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};

    VkImageLayout layout_before = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stage_before = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access_before = VK_ACCESS_2_NONE;

    VkImageLayout layout_after = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stage_after = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access_after = VK_ACCESS_2_NONE;
};

// Captures an image (the swapchain or the viewport target) to a PNG in three stages:
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

    auto request(ScreenshotSource source) noexcept -> void {
        source_.store(source, std::memory_order_relaxed);
        requested_.store(true, std::memory_order_release);
    }

    // The source of the pending request, if any, so the caller can pick the image to hand to record().
    [[nodiscard]]
    auto pending_source() const noexcept -> std::optional<ScreenshotSource> {
        if (!requested_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        return source_.load(std::memory_order_relaxed);
    }

    // Records the copy if a capture is pending and the slot is free, leaving the image in source.layout_after.
    // Returns whether a copy was recorded.
    [[nodiscard]]
    auto record(VulkanContext &ctx, VkCommandBuffer command_buffer, ScreenshotImage const &source,
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
    std::atomic<ScreenshotSource> source_{ScreenshotSource::window};

    std::vector<std::unique_ptr<ReadbackSlot>> slots_;

    std::mutex mutex_;
    std::condition_variable cv_;
};
