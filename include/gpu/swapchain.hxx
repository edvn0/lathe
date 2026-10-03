#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string_view>
#include <vector>

#include "core/error_context.hxx"

struct SwapchainCreateInfo {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;

    std::uint32_t graphics_queue_family = 0;
    std::uint32_t present_queue_family = 0;

    VkExtent2D framebuffer_extent{};
    bool vsync = true;
};

enum class SwapchainFrameResult : std::uint8_t {
    success,
    recreated,
    device_lost,
    fatal_error,
};

struct SwapchainBeginFrameError {
    enum class Kind : std::uint8_t {
        recreated,
        device_lost,
        fatal_error,
    };

    Kind kind = Kind::fatal_error;

    std::optional<ErrorContext> context{std::nullopt};
};

template<>
struct std::formatter<SwapchainFrameResult> : std::formatter<std::string_view> {
    constexpr auto format(SwapchainFrameResult error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case SwapchainFrameResult::success:
                    return "success";
                case SwapchainFrameResult::recreated:
                    return "recreated";
                case SwapchainFrameResult::device_lost:
                    return "device_lost";
                case SwapchainFrameResult::fatal_error:
                    return "fatal_error";
            }

            return "unknown_swapchain_frame_result";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

template<>
struct std::formatter<SwapchainBeginFrameError::Kind> : std::formatter<std::string_view> {
    constexpr auto format(SwapchainBeginFrameError::Kind error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case SwapchainBeginFrameError::Kind::recreated:
                    return "recreated";
                case SwapchainBeginFrameError::Kind::device_lost:
                    return "device_lost";
                case SwapchainBeginFrameError::Kind::fatal_error:
                    return "fatal_error";
            }

            return "unknown_swapchain_begin_frame_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct SwapchainFrame {
    // Filled in by the caller from QueueSet::command_buffer(); acquire() leaves it null.
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;

    VkImage image = VK_NULL_HANDLE;
    VkImageView image_view = VK_NULL_HANDLE;

    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;

    std::uint32_t image_index = 0;
    std::uint32_t frame_index = 0;

    bool acquire_suboptimal = false;
};

class Swapchain {
public:
    Swapchain() = default;
    ~Swapchain();

    Swapchain(Swapchain const &) = delete;
    auto operator=(Swapchain const &) -> Swapchain & = delete;

    Swapchain(Swapchain &&) = delete;
    auto operator=(Swapchain &&) -> Swapchain & = delete;

    [[nodiscard]]
    auto initialize(SwapchainCreateInfo const &create_info) noexcept -> bool;

    auto destroy() noexcept -> void;

    [[nodiscard]]
    auto frame_count() const noexcept -> std::uint32_t {
        return static_cast<std::uint32_t>(frames_.size());
    }

    auto request_recreate(VkExtent2D framebuffer_extent) noexcept -> void {
        requested_extent_ = framebuffer_extent;
        recreate_requested_ = true;
    }

    // The frame slot the next acquire() is for; QueueSet::begin_slot() must have run for it. The slot advances when a
    // frame is presented.
    [[nodiscard]]
    auto current_slot() const noexcept -> std::uint32_t {
        return current_frame_;
    }

    // Acquires the next image into `slot`'s image_available semaphore. Handles pending and out-of-date recreation
    // (reported as Kind::recreated, with no image acquired). The returned frame has no command buffer: the caller
    // records into one from QueueSet.
    [[nodiscard]]
    auto acquire(std::uint32_t slot) noexcept -> std::expected<SwapchainFrame, SwapchainBeginFrameError>;

    // Presents the frame's image, waiting on render_finished(frame.image_index), then advances the slot. The caller
    // has submitted work that signals that semaphore.
    [[nodiscard]]
    auto present(SwapchainFrame const &frame) noexcept -> SwapchainFrameResult;

    // Binary semaphores the acquire signals and presentation waits on, for the submit in between.
    [[nodiscard]]
    auto image_available(std::uint32_t slot) const noexcept -> VkSemaphore {
        return slot < frames_.size() ? frames_[slot].image_available : VK_NULL_HANDLE;
    }

    [[nodiscard]]
    auto render_finished(std::uint32_t image_index) const noexcept -> VkSemaphore {
        return image_index < render_finished_semaphores_.size() ? render_finished_semaphores_[image_index]
                                                                : VK_NULL_HANDLE;
    }

    [[nodiscard]]
    auto extent() const noexcept -> VkExtent2D {
        return extent_;
    }

    [[nodiscard]]
    auto format() const noexcept -> VkFormat {
        return surface_format_.format;
    }

    // The mode the current swapchain was created with.
    [[nodiscard]]
    auto present_mode() const noexcept -> VkPresentModeKHR {
        return present_mode_;
    }

private:
    struct FrameResources {
        VkSemaphore image_available = VK_NULL_HANDLE;
    };

    [[nodiscard]]
    auto create_swapchain(VkSwapchainKHR old_swapchain) noexcept -> bool;

    [[nodiscard]]
    auto create_image_views() noexcept -> bool;

    [[nodiscard]]
    auto create_synchronization() noexcept -> bool;

    [[nodiscard]]
    auto recreate() noexcept -> bool;

    auto destroy_swapchain_resources() noexcept -> void;
    auto destroy_frame_resources() noexcept -> void;

    [[nodiscard]]
    auto choose_surface_format(std::vector<VkSurfaceFormatKHR> const &formats) const noexcept -> VkSurfaceFormatKHR;

    [[nodiscard]]
    auto choose_present_mode(std::vector<VkPresentModeKHR> const &present_modes) const noexcept -> VkPresentModeKHR;

    [[nodiscard]]
    auto choose_extent(VkSurfaceCapabilitiesKHR const &capabilities) const noexcept -> VkExtent2D;

    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;

    VkDevice device_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue present_queue_ = VK_NULL_HANDLE;

    std::uint32_t graphics_queue_family_ = 0;
    std::uint32_t present_queue_family_ = 0;

    bool vsync_ = true;
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkSurfaceFormatKHR surface_format_{};

    VkExtent2D extent_{};
    VkExtent2D requested_extent_{};

    std::vector<VkImage> images_;
    std::vector<VkImageView> image_views_;

    std::vector<VkSemaphore> render_finished_semaphores_;

    std::vector<FrameResources> frames_;

    std::uint32_t current_frame_ = 0;
    bool recreate_requested_ = false;
    bool device_lost_ = false; // recreate() found the device lost or hung
};
