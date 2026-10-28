#include "core/thread_pool.hxx"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

#include "core/logger.hxx"

namespace {
    std::atomic<bool> shutting_down{false};
}

auto request_thread_pool_shutdown() noexcept -> void { shutting_down.store(true, std::memory_order_release); }

auto thread_pool_shutting_down() noexcept -> bool { return shutting_down.load(std::memory_order_acquire); }

auto thread_pool_busy() noexcept -> bool { return thread_pool().get_tasks_total() != 0; }

auto exit_promptly(int code) noexcept -> int {
    if (!thread_pool_busy()) {
        return code;
    }

    info("Not waiting for {} unfinished background tasks", thread_pool().get_tasks_total());
    logger::Logger::the().flush();
    std::fflush(nullptr);
    std::_Exit(code);
}

auto thread_pool() noexcept -> BS::priority_thread_pool & {
    static auto const set_worker_idle_priority = [](std::size_t) noexcept {
        if (!BS::this_thread::set_os_thread_priority(BS::os_thread_priority::idle)) {
            warn("thread_pool: failed to lower a worker thread's OS priority to idle");
        }
    };

    static auto thread_pool_ =
            std::make_unique<BS::priority_thread_pool>(std::thread::hardware_concurrency(), set_worker_idle_priority);

    return *thread_pool_;
}
