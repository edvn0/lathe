#pragma once

#include <volk.h>

#include <chrono>
#include <future>
#include <memory>
#include <string_view>
#include <thread>

#include "core/logger.hxx"

// vkDeviceWaitIdle has no timeout, and a hung GPU never returns from it, which freezes the whole application with
// no message. This waits on a helper thread and gives up after `timeout`, returning VK_TIMEOUT.
//
// After VK_TIMEOUT the device must be treated as lost: don't call into it again except to exit. The helper thread is
// detached (a future from std::async would block in its destructor), so it may still be inside the driver when the
// caller moves on.
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

// True for results meaning the device can no longer be used: lost outright, or hung past a bounded wait.
[[nodiscard]]
constexpr auto is_device_failure(VkResult result) noexcept -> bool {
    return result == VK_ERROR_DEVICE_LOST || result == VK_TIMEOUT;
}
