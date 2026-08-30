#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

struct MemoryStats {
    std::uint64_t live_bytes;
    std::uint64_t peak_bytes;

    std::uint64_t total_allocated_bytes;
    std::uint64_t total_freed_bytes;

    std::uint64_t live_allocations;
    std::uint64_t total_allocations;
    std::uint64_t total_frees;
};

class MemoryTracker {
public:
    static auto on_allocate(std::size_t size) noexcept -> void;
    static auto on_free(std::size_t size) noexcept -> void;

    [[nodiscard]]
    static auto stats() noexcept -> MemoryStats;

    // Allocations made on this thread while an UntrackedScope is alive are left
    // out of the counters above (though still visible to Tracy) -- for allocator
    // bookkeeping whose churn isn't itself meaningful engine memory usage.
    // BS::thread_pool's internal std::promise/shared_ptr machinery is the
    // motivating case: it heap-allocates per parallelFor submission, which has
    // nothing to do with the game's actual memory footprint and just adds noise
    // to the per-frame allocation count.
    class UntrackedScope {
    public:
        UntrackedScope() noexcept;
        ~UntrackedScope() noexcept;

        UntrackedScope(UntrackedScope const &) = delete;
        auto operator=(UntrackedScope const &) -> UntrackedScope & = delete;

    private:
        bool previous_;
    };

    [[nodiscard]]
    static auto is_untracked() noexcept -> bool;

private:
    static inline thread_local bool untracked_{false};

    static inline std::atomic<std::uint64_t> live_bytes_{};
    static inline std::atomic<std::uint64_t> peak_bytes_{};

    static inline std::atomic<std::uint64_t> total_allocated_bytes_{};
    static inline std::atomic<std::uint64_t> total_freed_bytes_{};

    static inline std::atomic<std::uint64_t> live_allocations_{};
    static inline std::atomic<std::uint64_t> total_allocations_{};

    static inline std::atomic<std::uint64_t> total_frees_{};
};
