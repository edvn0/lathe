#include "gpu/swapchain.hxx"
#include "core/perf_events.hxx"

#include "core/logger.hxx"
#include "gpu/device_wait.hxx"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <format>
#include <future>
#include <limits>
#include <string_view>
#include <utility>

#include "core/config.hxx"

namespace {

    auto report_vk_error(char const *operation, VkResult result) noexcept -> void {
        error("{} failed with VkResult {}", operation, static_cast<int>(result));
    }

    auto make_vk_error(SwapchainBeginFrameError::Kind kind, std::string_view operation,
                       VkResult result) noexcept -> SwapchainBeginFrameError {
        error("{} failed with VkResult {}", operation, static_cast<int>(result));

        return SwapchainBeginFrameError{
                .kind = kind,
                .context =
                        ErrorContext{
                                .message = FlyString{operation},
                                .vk_result = result,
                        },
        };
    }

    auto make_error(SwapchainBeginFrameError::Kind kind,
                    std::string_view message = {}) noexcept -> SwapchainBeginFrameError {
        if (message.empty()) {
            return SwapchainBeginFrameError{.kind = kind};
        }

        return SwapchainBeginFrameError{
                .kind = kind,
                .context = ErrorContext{.message = FlyString{message}},
        };
    }

    auto wait_idle_or_exit(VkDevice device, std::string_view label) noexcept -> VkResult {
        auto const result = wait_idle_bounded(device, label);

        if (result == VK_TIMEOUT) {
            std::_Exit(EXIT_FAILURE);
        }

        return result;
    }

}

Swapchain::~Swapchain() { destroy(); }

auto Swapchain::initialize(SwapchainCreateInfo const &create_info) noexcept -> bool {
    destroy();

    if (create_info.physical_device == VK_NULL_HANDLE || create_info.device == VK_NULL_HANDLE ||
        create_info.surface == VK_NULL_HANDLE || create_info.graphics_queue == VK_NULL_HANDLE ||
        create_info.present_queue == VK_NULL_HANDLE) {
        error("Invalid swapchain create info");
        return false;
    }

    physical_device_ = create_info.physical_device;
    device_ = create_info.device;
    surface_ = create_info.surface;
    graphics_queue_ = create_info.graphics_queue;
    present_queue_ = create_info.present_queue;
    graphics_queue_family_ = create_info.graphics_queue_family;
    present_queue_family_ = create_info.present_queue_family;
    requested_extent_ = create_info.framebuffer_extent;
    vsync_ = create_info.vsync;
    preferred_present_mode_ = create_info.preferred_present_mode;
    requested_image_count_ = create_info.image_count;

    if (requested_extent_.width == 0 || requested_extent_.height == 0) {
        error("Initial framebuffer extent must be non-zero");
        destroy();
        return false;
    }

    frames_.resize(frames_in_flight);

    if (!create_swapchain(VK_NULL_HANDLE) || !create_image_views() || !create_synchronization()) {
        destroy();
        return false;
    }

    info("Swapchain created: {}x{}, {} images, format {}", extent_.width, extent_.height, images_.size(),
         static_cast<int>(surface_format_.format));

    return true;
}

auto Swapchain::acquire(std::uint32_t slot) noexcept -> std::expected<SwapchainFrame, SwapchainBeginFrameError> {
    using Kind = SwapchainBeginFrameError::Kind;

    if (recreate_requested_) {
        recreate_requested_ = false;

        if (!recreate()) {
            return std::unexpected(device_lost_ ? make_error(Kind::device_lost, "swapchain recreate")
                                                : make_error(Kind::fatal_error, "swapchain recreate failed"));
        }

        return std::unexpected(make_error(Kind::recreated));
    }

    if (frames_.empty() || swapchain_ == VK_NULL_HANDLE) {
        return std::unexpected(make_error(Kind::fatal_error, "swapchain has no frames / VK_NULL_HANDLE"));
    }

    if (slot != current_frame_) {
        return std::unexpected(
                make_error(Kind::fatal_error,
                           std::format("acquire for slot {} but the swapchain is on slot {}", slot, current_frame_)));
    }

    auto &frame = frames_[current_frame_];

    constexpr std::uint64_t acquire_timeout_ns = 2'000'000'000ULL;

    std::uint32_t image_index = 0;

    auto result = vkAcquireNextImageKHR(device_, swapchain_, acquire_timeout_ns, frame.image_available, VK_NULL_HANDLE,
                                        &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        if (!recreate()) {
            return std::unexpected(device_lost_
                                           ? make_error(Kind::device_lost, "swapchain recreate (out of date)")
                                           : make_error(Kind::fatal_error, "swapchain recreate failed (out of date)"));
        }

        return std::unexpected(make_error(Kind::recreated));
    }

    if (result == VK_ERROR_DEVICE_LOST) {
        return std::unexpected(make_error(Kind::device_lost, "vkAcquireNextImageKHR"));
    }

    if (result == VK_TIMEOUT || result == VK_NOT_READY) {
        return std::unexpected(make_error(Kind::device_lost, "vkAcquireNextImageKHR timed out"));
    }

    auto const acquire_suboptimal = result == VK_SUBOPTIMAL_KHR;

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        return std::unexpected(make_vk_error(Kind::fatal_error, "vkAcquireNextImageKHR", result));
    }

    if (image_index >= images_.size() || image_index >= image_views_.size()) {
        return std::unexpected(
                make_error(Kind::fatal_error, std::format("swapchain returned invalid image index {}", image_index)));
    }

    return SwapchainFrame{
            .command_buffer = VK_NULL_HANDLE,
            .image = images_[image_index],
            .image_view = image_views_[image_index],
            .extent = extent_,
            .format = surface_format_.format,
            .image_index = image_index,
            .frame_index = current_frame_,
            .acquire_suboptimal = acquire_suboptimal,
    };
}

auto Swapchain::present(SwapchainFrame const &active_frame) noexcept -> SwapchainFrameResult {
    if (active_frame.frame_index >= frames_.size() || active_frame.image_index >= render_finished_semaphores_.size() ||
        active_frame.frame_index != current_frame_) {
        error("Invalid swapchain frame passed to present");

        return SwapchainFrameResult::fatal_error;
    }

    auto const render_finished = render_finished_semaphores_[active_frame.image_index];

    VkPresentInfoKHR const present_info{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_finished,
            .swapchainCount = 1,
            .pSwapchains = &swapchain_,
            .pImageIndices = &active_frame.image_index,
            .pResults = nullptr,
    };

    auto const result = vkQueuePresentKHR(present_queue_, &present_info);

    auto const advance_frame = [this] {
        current_frame_ = (current_frame_ + 1) % static_cast<std::uint32_t>(frames_.size());
    };

    if (result == VK_ERROR_DEVICE_LOST) {
        advance_frame();
        return SwapchainFrameResult::device_lost;
    }

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || active_frame.acquire_suboptimal) {
        recreate_requested_ = true;
        advance_frame();

        return SwapchainFrameResult::recreated;
    }

    if (result != VK_SUCCESS) {
        report_vk_error("vkQueuePresentKHR", result);

        advance_frame();
        return SwapchainFrameResult::fatal_error;
    }

    advance_frame();

    return SwapchainFrameResult::success;
}

auto Swapchain::destroy() noexcept -> void {
    if (device_ != VK_NULL_HANDLE) {
        const VkResult wait_result = wait_idle_or_exit(device_, "Swapchain::destroy");

        if (wait_result != VK_SUCCESS && wait_result != VK_ERROR_DEVICE_LOST) {
            report_vk_error("vkDeviceWaitIdle(swapchain destroy)", wait_result);
        }

        destroy_frame_resources();
        destroy_swapchain_resources();
    }

    physical_device_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    surface_ = VK_NULL_HANDLE;
    graphics_queue_ = VK_NULL_HANDLE;
    present_queue_ = VK_NULL_HANDLE;
    graphics_queue_family_ = 0;
    present_queue_family_ = 0;
    current_frame_ = 0;
    requested_extent_ = {};
    recreate_requested_ = false;
}

auto Swapchain::create_swapchain(VkSwapchainKHR old_swapchain) noexcept -> bool {
    VkSurfaceCapabilitiesKHR capabilities{};

    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &capabilities);

    if (result != VK_SUCCESS) {
        report_vk_error("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", result);
        return false;
    }

    std::uint32_t format_count = 0;

    result = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, nullptr);

    if (result != VK_SUCCESS || format_count == 0) {
        report_vk_error("vkGetPhysicalDeviceSurfaceFormatsKHR(count)", result);
        return false;
    }

    std::vector<VkSurfaceFormatKHR> formats(format_count);

    result = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, formats.data());

    if (result != VK_SUCCESS) {
        report_vk_error("vkGetPhysicalDeviceSurfaceFormatsKHR(list)", result);
        return false;
    }

    std::uint32_t present_mode_count = 0;

    result = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &present_mode_count, nullptr);

    if (result != VK_SUCCESS || present_mode_count == 0) {
        report_vk_error("vkGetPhysicalDeviceSurfacePresentModesKHR(count)", result);
        return false;
    }

    std::vector<VkPresentModeKHR> present_modes(present_mode_count);

    result = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &present_mode_count,
                                                       present_modes.data());

    if (result != VK_SUCCESS) {
        report_vk_error("vkGetPhysicalDeviceSurfacePresentModesKHR(list)", result);
        return false;
    }

    constexpr VkImageUsageFlags required_usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    if ((capabilities.supportedUsageFlags & required_usage) != required_usage) {
        error("Surface does not support required swapchain "
              "image usage flags");
        return false;
    }

    surface_format_ = choose_surface_format(formats);
    const VkPresentModeKHR present_mode = choose_present_mode(present_modes);
    present_mode_ = present_mode;
    extent_ = choose_extent(capabilities);

    if (extent_.width == 0 || extent_.height == 0) {
        return false;
    }

    auto const requested = requested_image_count_ != 0 ? requested_image_count_ : 3U;
    std::uint32_t image_count =
            std::max(requested, capabilities.minImageCount + (requested_image_count_ != 0 ? 0U : 1U));

    if (capabilities.maxImageCount != 0) {
        image_count = std::min(image_count, capabilities.maxImageCount);
    }

    const std::array queue_family_indices{
            graphics_queue_family_,
            present_queue_family_,
    };

    const bool separate_queue_families = graphics_queue_family_ != present_queue_family_;

    const VkSwapchainCreateInfoKHR create_info{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .pNext = nullptr,
            .flags = 0,
            .surface = surface_,
            .minImageCount = image_count,
            .imageFormat = surface_format_.format,
            .imageColorSpace = surface_format_.colorSpace,
            .imageExtent = extent_,
            .imageArrayLayers = 1,
            .imageUsage = required_usage,
            .imageSharingMode = separate_queue_families ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount =
                    separate_queue_families ? static_cast<std::uint32_t>(queue_family_indices.size()) : 0,
            .pQueueFamilyIndices = separate_queue_families ? queue_family_indices.data() : nullptr,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = present_mode,
            .clipped = VK_TRUE,
            .oldSwapchain = old_swapchain,
    };

    VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;

    result = vkCreateSwapchainKHR(device_, &create_info, nullptr, &new_swapchain);

    if (result != VK_SUCCESS) {
        report_vk_error("vkCreateSwapchainKHR", result);
        return false;
    }

    swapchain_ = new_swapchain;

    std::uint32_t actual_image_count = 0;

    result = vkGetSwapchainImagesKHR(device_, swapchain_, &actual_image_count, nullptr);

    if (result != VK_SUCCESS || actual_image_count == 0) {
        report_vk_error("vkGetSwapchainImagesKHR(count)", result);
        return false;
    }

    images_.resize(actual_image_count);

    result = vkGetSwapchainImagesKHR(device_, swapchain_, &actual_image_count, images_.data());

    if (result != VK_SUCCESS) {
        report_vk_error("vkGetSwapchainImagesKHR(list)", result);
        return false;
    }

    return true;
}

auto Swapchain::create_image_views() noexcept -> bool {
    image_views_.resize(images_.size());

    for (std::size_t index = 0; index < images_.size(); ++index) {
        const VkImageViewCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = images_[index],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = surface_format_.format,
                .components =
                        VkComponentMapping{
                                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                        },
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
        };

        const VkResult result = vkCreateImageView(device_, &create_info, nullptr, &image_views_[index]);

        if (result != VK_SUCCESS) {
            report_vk_error("vkCreateImageView", result);
            return false;
        }
    }

    return true;
}

auto Swapchain::create_synchronization() noexcept -> bool {
    const VkSemaphoreCreateInfo semaphore_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
    };

    for (FrameResources &frame: frames_) {
        VkResult result = vkCreateSemaphore(device_, &semaphore_info, nullptr, &frame.image_available);

        if (result != VK_SUCCESS) {
            report_vk_error("vkCreateSemaphore(image available)", result);
            return false;
        }
    }

    render_finished_semaphores_.resize(images_.size());

    for (VkSemaphore &semaphore: render_finished_semaphores_) {
        const VkResult result = vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore);

        if (result != VK_SUCCESS) {
            report_vk_error("vkCreateSemaphore(render finished)", result);
            return false;
        }
    }

    return true;
}

auto Swapchain::recreate() noexcept -> bool {
    if (requested_extent_.width == 0 || requested_extent_.height == 0) {
        return true;
    }

    perf_events::record(PerfEvent::swapchain_recreate);

    const VkResult wait_result = wait_idle_bounded(device_, "Swapchain::recreate");

    if (wait_result != VK_SUCCESS) {
        device_lost_ = is_device_failure(wait_result);
        report_vk_error("vkDeviceWaitIdle(swapchain recreate)", wait_result);
        return false;
    }

    VkSwapchainKHR old_swapchain = swapchain_;
    swapchain_ = VK_NULL_HANDLE;

    for (VkImageView image_view: image_views_) {
        vkDestroyImageView(device_, image_view, nullptr);
    }

    image_views_.clear();
    images_.clear();

    for (VkSemaphore semaphore: render_finished_semaphores_) {
        vkDestroySemaphore(device_, semaphore, nullptr);
    }

    render_finished_semaphores_.clear();

    if (!create_swapchain(old_swapchain) || !create_image_views()) {
        if (old_swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, old_swapchain, nullptr);
        }

        if (extent_.width == 0 || extent_.height == 0) {
            recreate_requested_ = true;
            return true;
        }

        return false;
    }

    if (old_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, old_swapchain, nullptr);
    }

    const VkSemaphoreCreateInfo semaphore_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
    };

    render_finished_semaphores_.resize(images_.size());

    for (VkSemaphore &semaphore: render_finished_semaphores_) {
        const VkResult result = vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore);

        if (result != VK_SUCCESS) {
            report_vk_error("vkCreateSemaphore(render finished)", result);
            return false;
        }
    }

    info("Swapchain recreated: {}x{}, {} images", extent_.width, extent_.height, images_.size());

    return true;
}

auto Swapchain::destroy_swapchain_resources() noexcept -> void {
    for (VkSemaphore semaphore: render_finished_semaphores_) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
    }

    render_finished_semaphores_.clear();

    for (VkImageView image_view: image_views_) {
        if (image_view != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, image_view, nullptr);
        }
    }

    image_views_.clear();
    images_.clear();

    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }

    extent_ = {};
    surface_format_ = {};
}

auto Swapchain::destroy_frame_resources() noexcept -> void {
    for (FrameResources &frame: frames_) {
        if (frame.image_available != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, frame.image_available, nullptr);
            frame.image_available = VK_NULL_HANDLE;
        }
    }

    frames_.clear();
}

auto Swapchain::choose_surface_format(std::vector<VkSurfaceFormatKHR> const &formats) const noexcept
        -> VkSurfaceFormatKHR {
    constexpr std::array preferred_formats{
            VK_FORMAT_B8G8R8A8_SRGB,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_FORMAT_R8G8B8A8_UNORM,
    };

    for (VkFormat preferred_format: preferred_formats) {
        auto const iterator = std::ranges::find_if(formats, [preferred_format](VkSurfaceFormatKHR const &format) {
            return format.format == preferred_format && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });

        if (iterator != formats.end()) {
            return *iterator;
        }
    }

    return formats.front();
}

auto Swapchain::choose_present_mode(std::vector<VkPresentModeKHR> const &present_modes) const noexcept
        -> VkPresentModeKHR {
    if (preferred_present_mode_) {
        if (std::ranges::find(present_modes, *preferred_present_mode_) != present_modes.end()) {
            return *preferred_present_mode_;
        }
        warn("The requested present mode ({}) is not supported by this surface; choosing another",
             static_cast<int>(*preferred_present_mode_));
    }

    if (!vsync_) {
        auto const mailbox = std::ranges::find(present_modes, VK_PRESENT_MODE_MAILBOX_KHR);

        if (mailbox != present_modes.end()) {
            return *mailbox;
        }

        auto const immediate = std::ranges::find(present_modes, VK_PRESENT_MODE_IMMEDIATE_KHR);

        if (immediate != present_modes.end()) {
            return *immediate;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

auto Swapchain::choose_extent(VkSurfaceCapabilitiesKHR const &capabilities) const noexcept -> VkExtent2D {
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    return VkExtent2D{
            .width = std::clamp(requested_extent_.width, capabilities.minImageExtent.width,
                                capabilities.maxImageExtent.width),
            .height = std::clamp(requested_extent_.height, capabilities.minImageExtent.height,
                                 capabilities.maxImageExtent.height),
    };
}
