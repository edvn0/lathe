#include "core/thread_pool.hxx"

#include <memory>
#include <thread>

auto thread_pool() noexcept -> BS::priority_thread_pool & {
    static auto thread_pool_ = std::make_unique<BS::priority_thread_pool>(std::thread::hardware_concurrency());
    return *thread_pool_;
}
