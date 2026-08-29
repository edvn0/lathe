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

private:
    static inline std::atomic<std::uint64_t> live_bytes_{};
    static inline std::atomic<std::uint64_t> peak_bytes_{};

    static inline std::atomic<std::uint64_t> total_allocated_bytes_{};
    static inline std::atomic<std::uint64_t> total_freed_bytes_{};

    static inline std::atomic<std::uint64_t> live_allocations_{};
    static inline std::atomic<std::uint64_t> total_allocations_{};

    static inline std::atomic<std::uint64_t> total_frees_{};
};
