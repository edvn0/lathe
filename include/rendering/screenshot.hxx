#pragma once

#include "core/forward.hxx"
#include "gpu/buffer.hxx"

#include <volk.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

enum class ScreenshotSource : std::uint8_t {
    window,
    viewport,
};

struct ScreenshotImage {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};

    bool managed_by_graph = false;

    VkImageLayout layout_before = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stage_before = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access_before = VK_ACCESS_2_NONE;

    VkImageLayout layout_after = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stage_after = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access_after = VK_ACCESS_2_NONE;
};

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

    [[nodiscard]]
    auto pending_source() const noexcept -> std::optional<ScreenshotSource> {
        if (!requested_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        return source_.load(std::memory_order_relaxed);
    }

    [[nodiscard]]
    auto record(VulkanContext &ctx, VkCommandBuffer command_buffer, ScreenshotImage const &source,
                std::uint32_t frame_index) -> bool;

    auto try_resolve(std::uint32_t frame_index) -> void;

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
