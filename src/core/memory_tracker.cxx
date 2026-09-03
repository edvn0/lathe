#include "core/memory_tracker.hxx"

auto MemoryTracker::on_allocate(std::size_t const size) noexcept -> void {
    auto const bytes = live_bytes_.fetch_add(size, std::memory_order_relaxed) + size;

    total_allocated_bytes_.fetch_add(size, std::memory_order_relaxed);
    live_allocations_.fetch_add(1, std::memory_order_relaxed);
    total_allocations_.fetch_add(1, std::memory_order_relaxed);

    auto peak = peak_bytes_.load(std::memory_order_relaxed);

    while (bytes > peak &&
           !peak_bytes_.compare_exchange_weak(peak, bytes, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}


auto MemoryTracker::on_free(std::size_t const size) noexcept -> void {
    live_bytes_.fetch_sub(size, std::memory_order_relaxed);
    total_freed_bytes_.fetch_add(size, std::memory_order_relaxed);
    live_allocations_.fetch_sub(1, std::memory_order_relaxed);
    total_frees_.fetch_add(1, std::memory_order_relaxed);
}

auto MemoryTracker::is_untracked() noexcept -> bool { return untracked_; }

MemoryTracker::UntrackedScope::UntrackedScope() noexcept : previous_{untracked_} { untracked_ = true; }

MemoryTracker::UntrackedScope::~UntrackedScope() noexcept { untracked_ = previous_; }

auto MemoryTracker::stats() noexcept -> MemoryStats {
    return {
            .live_bytes = live_bytes_.load(std::memory_order_relaxed),
            .peak_bytes = peak_bytes_.load(std::memory_order_relaxed),
            .total_allocated_bytes = total_allocated_bytes_.load(std::memory_order_relaxed),
            .total_freed_bytes = total_freed_bytes_.load(std::memory_order_relaxed),
            .live_allocations = live_allocations_.load(std::memory_order_relaxed),
            .total_allocations = total_allocations_.load(std::memory_order_relaxed),
            .total_frees = total_frees_.load(std::memory_order_relaxed),
    };
}
