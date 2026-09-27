#include "core/thread_pool.hxx"

#include <memory>
#include <thread>

#include "core/logger.hxx"

auto thread_pool() noexcept -> BS::priority_thread_pool & {
    // Workers run at SCHED_IDLE so background work never starves the render thread. Undersizing the pool to
    // reserve a core barely helped.
    static auto const set_worker_idle_priority = [](std::size_t) noexcept {
        if (!BS::this_thread::set_os_thread_priority(BS::os_thread_priority::idle)) {
            warn("thread_pool: failed to lower a worker thread's OS priority to idle");
        }
    };

    static auto thread_pool_ =
            std::make_unique<BS::priority_thread_pool>(std::thread::hardware_concurrency(), set_worker_idle_priority);

    return *thread_pool_;
}
