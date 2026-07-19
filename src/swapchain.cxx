#include "swapchain.hxx"

#include "logger.hxx"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include "config.hxx"

namespace {

    auto report_vk_error(const char *operation, VkResult result) noexcept -> void {
        error("{} failed with VkResult {}", operation, static_cast<int>(result));
    }

} // namespace

Swapchain::~Swapchain() { destroy(); }

auto Swapchain::initialize(const SwapchainCreateInfo &create_info) noexcept -> bool {
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

    if (requested_extent_.width == 0 || requested_extent_.height == 0) {
        error("Initial framebuffer extent must be non-zero");
        destroy();
        return false;
    }

    frames_.resize(frames_in_flight);

    if (!create_swapchain(VK_NULL_HANDLE) || !create_image_views() || !create_command_resources() ||
        !create_synchronization()) {
        destroy();
        return false;
    }

    info("Swapchain created: {}x{}, {} images, format {}", extent_.width, extent_.height, images_.size(),
         static_cast<int>(surface_format_.format));

    return true;
}

auto Swapchain::begin_frame() noexcept -> std::expected<SwapchainFrame, SwapchainBeginFrameError> {
    if (recreate_requested_) {
        recreate_requested_ = false;

        if (!recreate()) {
            return std::unexpected(SwapchainBeginFrameError::fatal_error);
        }

        return std::unexpected(SwapchainBeginFrameError::recreated);
    }

    if (frames_.empty() || swapchain_ == VK_NULL_HANDLE) {
        return std::unexpected(SwapchainBeginFrameError::fatal_error);
    }

    auto &frame = frames_[current_frame_];

    auto result = vkWaitForFences(device_, 1, &frame.in_flight, VK_TRUE, std::numeric_limits<std::uint64_t>::max());

    if (result == VK_ERROR_DEVICE_LOST) {
        return std::unexpected(SwapchainBeginFrameError::device_lost);
    }

    if (result != VK_SUCCESS) {
        report_vk_error("vkWaitForFences", result);

        return std::unexpected(SwapchainBeginFrameError::fatal_error);
    }

    std::uint32_t image_index = 0;

    result = vkAcquireNextImageKHR(device_, swapchain_, std::numeric_limits<std::uint64_t>::max(),
                                   frame.image_available, VK_NULL_HANDLE, &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        if (!recreate()) {
            return std::unexpected(SwapchainBeginFrameError::fatal_error);
        }

        return std::unexpected(SwapchainBeginFrameError::recreated);
    }

    if (result == VK_ERROR_DEVICE_LOST) {
        return std::unexpected(SwapchainBeginFrameError::device_lost);
    }

    auto const acquire_suboptimal = result == VK_SUBOPTIMAL_KHR;

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        report_vk_error("vkAcquireNextImageKHR", result);

        return std::unexpected(SwapchainBeginFrameError::fatal_error);
    }

    if (image_index >= images_.size() || image_index >= image_views_.size()) {
        error("Swapchain returned invalid image index {}", image_index);

        return std::unexpected(SwapchainBeginFrameError::fatal_error);
    }

    result = vkResetCommandBuffer(frame.command_buffer, 0);

    if (result != VK_SUCCESS) {
        report_vk_error("vkResetCommandBuffer", result);

        return std::unexpected(SwapchainBeginFrameError::fatal_error);
    }

    VkCommandBufferBeginInfo const begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
    };

    result = vkBeginCommandBuffer(frame.command_buffer, &begin_info);

    if (result != VK_SUCCESS) {
        report_vk_error("vkBeginCommandBuffer", result);

        return std::unexpected(SwapchainBeginFrameError::fatal_error);
    }

    return SwapchainFrame{
            .command_buffer = frame.command_buffer,
            .image = images_[image_index],
            .image_view = image_views_[image_index],
            .extent = extent_,
            .format = surface_format_.format,
            .image_index = image_index,
            .frame_index = current_frame_,
            .acquire_suboptimal = acquire_suboptimal,
    };
}

auto Swapchain::end_frame(SwapchainFrame const &active_frame) noexcept -> SwapchainFrameResult {
    if (active_frame.frame_index >= frames_.size() || active_frame.image_index >= render_finished_semaphores_.size() ||
        active_frame.command_buffer == VK_NULL_HANDLE || active_frame.frame_index != current_frame_) {
        error("Invalid swapchain frame passed to end_frame");

        return SwapchainFrameResult::fatal_error;
    }

    auto &frame = frames_[active_frame.frame_index];

    if (frame.command_buffer != active_frame.command_buffer) {
        error("Swapchain frame command buffer does not "
              "match its frame resource");

        return SwapchainFrameResult::fatal_error;
    }

    auto result = vkEndCommandBuffer(active_frame.command_buffer);

    if (result != VK_SUCCESS) {
        report_vk_error("vkEndCommandBuffer", result);

        return SwapchainFrameResult::fatal_error;
    }

    result = vkResetFences(device_, 1, &frame.in_flight);

    if (result != VK_SUCCESS) {
        report_vk_error("vkResetFences", result);

        return SwapchainFrameResult::fatal_error;
    }

    VkSemaphoreSubmitInfo const wait_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = frame.image_available,
            .value = 0,

            .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,

            .deviceIndex = 0,
    };

    VkCommandBufferSubmitInfo const command_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = nullptr,
            .commandBuffer = active_frame.command_buffer,
            .deviceMask = 0,
    };

    auto const render_finished = render_finished_semaphores_[active_frame.image_index];

    VkSemaphoreSubmitInfo const signal_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = render_finished,
            .value = 0,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0,
    };

    VkSubmitInfo2 const submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = nullptr,
            .flags = 0,
            .waitSemaphoreInfoCount = 1,
            .pWaitSemaphoreInfos = &wait_info,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &command_info,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos = &signal_info,
    };

    result = vkQueueSubmit2(graphics_queue_, 1, &submit_info, frame.in_flight);

    if (result == VK_ERROR_DEVICE_LOST) {
        return SwapchainFrameResult::device_lost;
    }

    if (result != VK_SUCCESS) {
        report_vk_error("vkQueueSubmit2", result);

        return SwapchainFrameResult::fatal_error;
    }

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

    result = vkQueuePresentKHR(present_queue_, &present_info);

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
        const VkResult wait_result = vkDeviceWaitIdle(device_);

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

    constexpr VkImageUsageFlags required_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    if ((capabilities.supportedUsageFlags & required_usage) != required_usage) {
        error("Surface does not support required swapchain "
              "image usage flags");
        return false;
    }

    surface_format_ = choose_surface_format(formats);
    const VkPresentModeKHR present_mode = choose_present_mode(present_modes);
    extent_ = choose_extent(capabilities);

    constexpr auto requested = 5u;
    std::uint32_t image_count = std::max(requested, capabilities.minImageCount + 1);

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

auto Swapchain::create_command_resources() noexcept -> bool {
    const VkCommandPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = graphics_queue_family_,
    };

    VkResult result = vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);

    if (result != VK_SUCCESS) {
        report_vk_error("vkCreateCommandPool", result);
        return false;
    }

    std::vector<VkCommandBuffer> command_buffers(frames_.size());

    const VkCommandBufferAllocateInfo allocate_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = command_pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<std::uint32_t>(command_buffers.size()),
    };

    result = vkAllocateCommandBuffers(device_, &allocate_info, command_buffers.data());

    if (result != VK_SUCCESS) {
        report_vk_error("vkAllocateCommandBuffers", result);
        return false;
    }

    for (std::size_t index = 0; index < frames_.size(); ++index) {
        frames_[index].command_buffer = command_buffers[index];
    }

    return true;
}

auto Swapchain::create_synchronization() noexcept -> bool {
    const VkSemaphoreCreateInfo semaphore_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
    };

    const VkFenceCreateInfo fence_info{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    for (FrameResources &frame: frames_) {
        VkResult result = vkCreateSemaphore(device_, &semaphore_info, nullptr, &frame.image_available);

        if (result != VK_SUCCESS) {
            report_vk_error("vkCreateSemaphore(image available)", result);
            return false;
        }

        result = vkCreateFence(device_, &fence_info, nullptr, &frame.in_flight);

        if (result != VK_SUCCESS) {
            report_vk_error("vkCreateFence", result);
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

    const VkResult wait_result = vkDeviceWaitIdle(device_);

    if (wait_result != VK_SUCCESS) {
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
        if (frame.in_flight != VK_NULL_HANDLE) {
            vkDestroyFence(device_, frame.in_flight, nullptr);
            frame.in_flight = VK_NULL_HANDLE;
        }

        if (frame.image_available != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, frame.image_available, nullptr);
            frame.image_available = VK_NULL_HANDLE;
        }

        frame.command_buffer = VK_NULL_HANDLE;
    }

    frames_.clear();

    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }
}

auto Swapchain::choose_surface_format(const std::vector<VkSurfaceFormatKHR> &formats) const noexcept
        -> VkSurfaceFormatKHR {
    constexpr std::array preferred_formats{
            VK_FORMAT_B8G8R8A8_SRGB,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_FORMAT_R8G8B8A8_UNORM,
    };

    for (VkFormat preferred_format: preferred_formats) {
        const auto iterator = std::ranges::find_if(formats, [preferred_format](const VkSurfaceFormatKHR &format) {
            return format.format == preferred_format && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });

        if (iterator != formats.end()) {
            return *iterator;
        }
    }

    return formats.front();
}

auto Swapchain::choose_present_mode(const std::vector<VkPresentModeKHR> &present_modes) const noexcept
        -> VkPresentModeKHR {
    if (!vsync_) {
        const auto mailbox = std::ranges::find(present_modes, VK_PRESENT_MODE_MAILBOX_KHR);

        if (mailbox != present_modes.end()) {
            return *mailbox;
        }

        const auto immediate = std::ranges::find(present_modes, VK_PRESENT_MODE_IMMEDIATE_KHR);

        if (immediate != present_modes.end()) {
            return *immediate;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

auto Swapchain::choose_extent(const VkSurfaceCapabilitiesKHR &capabilities) const noexcept -> VkExtent2D {
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
