#include <tracy/Tracy.hpp>

#include <cstdio>
#include <cstdlib>

#include "memory_tracker.hxx"

namespace {

    [[noreturn]] auto abort_out_of_memory(std::size_t size) -> void {
        std::fprintf(stderr, "out of memory allocating %zu bytes\n", size);
        std::abort();
    }

} // namespace

auto operator new(std::size_t size) -> void * {
    void *ptr = std::malloc(size);
    if (ptr == nullptr) {
        abort_out_of_memory(size);
    }

    MemoryTracker::on_allocate(size);
    TracyAlloc(ptr, size);

    return ptr;
}

auto operator new[](std::size_t size) -> void * {
    void *ptr = std::malloc(size);
    if (ptr == nullptr) {
        abort_out_of_memory(size);
    }

    MemoryTracker::on_allocate(size);
    TracyAlloc(ptr, size);

    return ptr;
}

auto operator delete(void *ptr) noexcept -> void {
    TracyFree(ptr);
    std::free(ptr);
}

auto operator delete[](void *ptr) noexcept -> void {
    TracyFree(ptr);
    std::free(ptr);
}

auto operator delete(void *ptr, std::size_t free_size) noexcept -> void {
    TracyFree(ptr);
    std::free(ptr);
    MemoryTracker::on_free(free_size);
}

auto operator delete[](void *ptr, std::size_t free_size) noexcept -> void {
    TracyFree(ptr);
    std::free(ptr);
    MemoryTracker::on_free(free_size);
}
