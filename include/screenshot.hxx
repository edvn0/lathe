#pragma once

#include "buffer.hxx"
#include "forward.hxx"

#include <volk.h>

#include <atomic>
#include <cstdint>

// Captures the composited swapchain image (scene + ImGui overlay) to a PNG on
// disk, without stalling the render thread. request() is safe to call from
// any thread (e.g. an input callback); record()/try_resolve() must only be
// called from the render thread, driven from Renderer::record_frame().
//
// This is a "request -> record -> deferred pickup" pipeline: record() copies
// the swapchain image into a readback buffer using this frame's command
// buffer, then try_resolve() reads that buffer back once the same
// frame-in-flight slot recurs -- which Swapchain::begin_frame() already
// guarantees is GPU-complete by then, so no extra fence is needed here.
class ScreenshotCapture {
public:
    auto request() noexcept -> void { requested_.store(true, std::memory_order_relaxed); }

    // If a capture was requested and none is already pending, records a copy
    // of `image` into an internal readback buffer using `command_buffer`,
    // leaving `image` in VK_IMAGE_LAYOUT_PRESENT_SRC_KHR. Returns true if it
    // did so -- the caller must then skip its own swapchain-to-present
    // transition, since this already performed it.
    [[nodiscard]]
    auto record(VulkanContext &ctx, VkCommandBuffer command_buffer, VkImage image, VkFormat format,
               VkExtent2D extent, std::uint32_t frame_index) -> bool;

    // Call once per frame, before record(), with that frame's frame_index. If
    // that slot is the one record() used, reads the buffer back and writes
    // the PNG on a detached thread.
    auto try_resolve(std::uint32_t frame_index) -> void;

private:
    std::atomic<bool> requested_{false};

    bool pending_ = false;
    std::uint32_t pending_frame_index_ = 0;
    VkExtent2D pending_extent_{};
    VkFormat pending_format_ = VK_FORMAT_UNDEFINED;

    Buffer readback_buffer_;
};
