#pragma once

#include <volk.h>

#include <chrono>
#include <future>
#include <memory>
#include <string_view>
#include <thread>

#include "core/logger.hxx"

[[nodiscard]]
inline auto wait_idle_bounded(VkDevice device, std::string_view label,
                              std::chrono::seconds timeout = std::chrono::seconds{3}) noexcept -> VkResult {
    auto promise = std::make_shared<std::promise<VkResult>>();
    auto future = promise->get_future();

    std::thread{[device, promise] { promise->set_value(vkDeviceWaitIdle(device)); }}.detach();

    if (future.wait_for(timeout) != std::future_status::ready) {
        error("{}: vkDeviceWaitIdle did not return within {} -- the GPU is not responding.", label, timeout);

        return VK_TIMEOUT;
    }

    return future.get();
}

[[nodiscard]]
constexpr auto is_device_failure(VkResult result) noexcept -> bool {
    return result == VK_ERROR_DEVICE_LOST || result == VK_TIMEOUT;
}
