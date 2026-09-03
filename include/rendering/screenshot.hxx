#pragma once

#include "gpu/buffer.hxx"
#include "core/forward.hxx"

#include <volk.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Captures the composited swapchain image (scene + ImGui overlay) to a PNG.
//
// Capture is split into three stages:
//
//   1. record()
//      Records an asynchronous GPU image -> persistently mapped readback-buffer
//      copy into the current frame-in-flight slot.
//
//   2. try_resolve()
//      Called when that frame-in-flight slot recurs. At that point the frame
//      fence has already completed, so the GPU no longer accesses the buffer.
//      The mapped allocation is handed to a worker thread.
//
//   3. worker thread
//      Invalidates the mapped allocation if necessary, memcpy()s the pixels
//      into CPU-owned memory, releases the readback slot, then performs channel
//      conversion and PNG compression entirely independently of Vulkan.
//
// request() may be called from any thread.
// record()/try_resolve() must only be called from the render thread.
class ScreenshotCapture {
public:
    ScreenshotCapture() = default;
    ~ScreenshotCapture();

    ScreenshotCapture(ScreenshotCapture const &) = delete;
    auto operator=(ScreenshotCapture const &) -> ScreenshotCapture & = delete;

    ScreenshotCapture(ScreenshotCapture &&) = delete;
    auto operator=(ScreenshotCapture &&) -> ScreenshotCapture & = delete;

    auto request() noexcept -> void { requested_.store(true, std::memory_order_relaxed); }

    // If a capture was requested and the current frame slot is available,
    // records an image -> buffer copy and transitions the image to
    // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
    //
    // Returns true when the screenshot path performed the present transition.
    // The caller must skip its normal swapchain-to-present transition in that
    // case.
    [[nodiscard]]
    auto record(VulkanContext &ctx, VkCommandBuffer command_buffer, VkImage image, VkFormat format, VkExtent2D extent,
                std::uint32_t frame_index) -> bool;

    // Called once the frame-in-flight slot identified by frame_index has had
    // its fence waited/reset by the renderer.
    //
    // If that slot contains a completed screenshot copy, ownership of its
    // mapped memory is temporarily handed to a background worker.
    auto try_resolve(std::uint32_t frame_index) -> void;

private:
    struct ReadbackSlot {
        Buffer buffer;

        // True while a CPU worker may still be reading buffer.mapped_data().
        std::atomic<bool> cpu_busy{false};

        // True between record() submitting the GPU copy and the corresponding
        // frame-in-flight slot becoming GPU-complete.
        bool gpu_pending = false;

        VkExtent2D extent{};
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::size_t byte_size = 0;
    };

    [[nodiscard]]
    auto get_or_create_slot(std::uint32_t frame_index) -> ReadbackSlot &;

    std::atomic<bool> requested_{false};

    // unique_ptr keeps each ReadbackSlot at a stable address even if the vector
    // itself reallocates. Background workers are therefore free to retain a
    // ReadbackSlot* until their memcpy has completed.
    std::vector<std::unique_ptr<ReadbackSlot>> slots_;
};
