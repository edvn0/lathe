#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <vector>

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

enum class SwapchainFrameResult {
    success,
    recreated,
    device_lost,
    fatal_error,
};

enum class SwapchainBeginFrameError {
    recreated,
    device_lost,
    fatal_error,
};

struct SwapchainFrame {
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

    [[nodiscard]]
    auto begin_frame() noexcept -> std::expected<SwapchainFrame, SwapchainBeginFrameError>;

    [[nodiscard]]
    auto end_frame(SwapchainFrame const &frame) noexcept -> SwapchainFrameResult;

    [[nodiscard]]
    auto extent() const noexcept -> VkExtent2D {
        return extent_;
    }

    [[nodiscard]]
    auto format() const noexcept -> VkFormat {
        return surface_format_.format;
    }

private:
    struct FrameResources {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;

        VkSemaphore image_available = VK_NULL_HANDLE;

        VkFence in_flight = VK_NULL_HANDLE;
    };

    [[nodiscard]]
    auto create_swapchain(VkSwapchainKHR old_swapchain) noexcept -> bool;

    [[nodiscard]]
    auto create_image_views() noexcept -> bool;

    [[nodiscard]]
    auto create_command_resources() noexcept -> bool;

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

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkSurfaceFormatKHR surface_format_{};

    VkExtent2D extent_{};
    VkExtent2D requested_extent_{};

    std::vector<VkImage> images_;
    std::vector<VkImageView> image_views_;

    std::vector<VkSemaphore> render_finished_semaphores_;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<FrameResources> frames_;

    std::uint32_t current_frame_ = 0;
    bool recreate_requested_ = false;
};
