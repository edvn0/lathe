#include "core/thread_pool.hxx"

#include <memory>
#include <thread>

#include "core/logger.hxx"

auto thread_pool() noexcept -> BS::priority_thread_pool & {
    // Every worker runs at SCHED_IDLE (see BS::this_thread::
    // set_os_thread_priority, enabled via BS_THREAD_POOL_NATIVE_EXTENSIONS
    // -- see src/core/CMakeLists.txt), so background work here (glTF
    // parsing, texture encode/transcode, ...) only ever runs on CPU time
    // nothing else wants. Reserving a core by undersizing the pool instead
    // was tried first and measured to barely help: a fixed core count can't
    // account for this process's own other threads sharing that reserved
    // core, or for unrelated processes on the same machine also competing
    // for it, whereas SCHED_IDLE is honored by the kernel scheduler
    // unconditionally -- profiling a cold-cache Sponza load with the pool
    // still sized to every hardware thread showed the render thread so
    // starved that a single per-frame compress_vertices() call (normally
    // sub-millisecond) measured well over a second of wall-clock time,
    // purely from waiting for a timeslice, not real work.
    static auto const set_worker_idle_priority = [](std::size_t) noexcept {
        if (!BS::this_thread::set_os_thread_priority(BS::os_thread_priority::idle)) {
            warn("thread_pool: failed to lower a worker thread's OS priority to idle");
        }
    };

    static auto thread_pool_ =
            std::make_unique<BS::priority_thread_pool>(std::thread::hardware_concurrency(), set_worker_idle_priority);

    return *thread_pool_;
}
