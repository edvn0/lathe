#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include "app/fatal_dialog.hxx"
#include "gpu/device_wait.hxx"

namespace {

    std::atomic_bool release_hung_wait{false};

    VKAPI_ATTR auto VKAPI_CALL idle_returns_success(VkDevice) -> VkResult { return VK_SUCCESS; }

    VKAPI_ATTR auto VKAPI_CALL idle_reports_loss(VkDevice) -> VkResult { return VK_ERROR_DEVICE_LOST; }

    // A GPU that hangs: never returns until the test lets it.
    VKAPI_ATTR auto VKAPI_CALL idle_hangs(VkDevice) -> VkResult {
        while (!release_hung_wait.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
        }

        return VK_SUCCESS;
    }

    // vkDeviceWaitIdle is volk's function pointer, so a test can stand in for the driver.
    struct ScopedDeviceWaitIdle {
        PFN_vkDeviceWaitIdle saved = vkDeviceWaitIdle;

        explicit ScopedDeviceWaitIdle(PFN_vkDeviceWaitIdle replacement) { vkDeviceWaitIdle = replacement; }

        ~ScopedDeviceWaitIdle() { vkDeviceWaitIdle = saved; }

        ScopedDeviceWaitIdle(ScopedDeviceWaitIdle const &) = delete;
        auto operator=(ScopedDeviceWaitIdle const &) -> ScopedDeviceWaitIdle & = delete;
    };

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("wait_idle_bounded passes a healthy or failed wait through") {
        {
            ScopedDeviceWaitIdle const fake{idle_returns_success};
            CHECK(wait_idle_bounded(VK_NULL_HANDLE, "test") == VK_SUCCESS);
        }

        {
            ScopedDeviceWaitIdle const fake{idle_reports_loss};
            CHECK(wait_idle_bounded(VK_NULL_HANDLE, "test") == VK_ERROR_DEVICE_LOST);
        }
    }

    TEST_CASE("wait_idle_bounded gives up on a hung GPU instead of freezing the caller") {
        release_hung_wait.store(false, std::memory_order_release);

        ScopedDeviceWaitIdle const fake{idle_hangs};
        auto const start = std::chrono::steady_clock::now();

        CHECK(wait_idle_bounded(VK_NULL_HANDLE, "test", std::chrono::seconds{1}) == VK_TIMEOUT);
        CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds{3});

        // The detached helper is still inside the "driver"; let it finish before the fake goes out of scope.
        release_hung_wait.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }

    TEST_CASE("Device failures are the lost and hung results only") {
        CHECK(is_device_failure(VK_ERROR_DEVICE_LOST));
        CHECK(is_device_failure(VK_TIMEOUT));
        CHECK_FALSE(is_device_failure(VK_SUCCESS));
        CHECK_FALSE(is_device_failure(VK_ERROR_OUT_OF_DATE_KHR));
        CHECK_FALSE(is_device_failure(VK_ERROR_OUT_OF_DEVICE_MEMORY));
    }

#if !defined(_WIN32)
    TEST_CASE("The fatal dialog declines to run without a display") {
        auto const *display_value = std::getenv("DISPLAY");
        auto const *wayland_value = std::getenv("WAYLAND_DISPLAY");
        auto const display = display_value != nullptr ? std::string{display_value} : std::string{};
        auto const wayland = wayland_value != nullptr ? std::string{wayland_value} : std::string{};

        unsetenv("DISPLAY");
        unsetenv("WAYLAND_DISPLAY");

        CHECK_FALSE(show_fatal_dialog("title", "message"));

        if (!display.empty()) {
            setenv("DISPLAY", display.c_str(), 1);
        }
        if (!wayland.empty()) {
            setenv("WAYLAND_DISPLAY", wayland.c_str(), 1);
        }
    }
#endif
}
