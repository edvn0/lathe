#include <tracy/Tracy.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "memory_tracker.hxx"

namespace {

    [[noreturn]] auto abort_out_of_memory(std::size_t size) -> void {
        std::fprintf(stderr, "out of memory allocating %zu bytes\n", size);
        std::abort();
    }

    // Every allocation carries its own header recording its real size and whether
    // it was tracked. That makes freeing correct regardless of which delete
    // overload the compiler ends up choosing (deleting through a polymorphic base
    // pointer can't pass a size, so the *un*sized overload runs instead) and
    // regardless of which thread frees it (relevant for MemoryTracker::UntrackedScope,
    // whose thread_local flag is read once here at allocation time -- never at
    // free time, since the freeing thread may not be the allocating one).
    struct AllocHeader {
        std::size_t size;
        bool tracked;
    };

    constexpr std::size_t header_size = [] {
        constexpr std::size_t align = alignof(std::max_align_t);
        return ((sizeof(AllocHeader) + align - 1) / align) * align;
    }();

    auto allocate(std::size_t size) -> void * {
        void *raw = std::malloc(size + header_size);
        if (raw == nullptr) {
            abort_out_of_memory(size);
        }

        auto const tracked = !MemoryTracker::is_untracked();

        ::new (raw) AllocHeader{.size = size, .tracked = tracked};

        void *ptr = static_cast<std::byte *>(raw) + header_size;

        if (tracked) {
            MemoryTracker::on_allocate(size);
        }
        TracyAlloc(ptr, size);

        return ptr;
    }

    auto deallocate(void *ptr) noexcept -> void {
        if (ptr == nullptr) {
            return;
        }

        TracyFree(ptr);

        auto *raw = static_cast<std::byte *>(ptr) - header_size;
        auto const *header = reinterpret_cast<AllocHeader const *>(raw);

        if (header->tracked) {
            MemoryTracker::on_free(header->size);
        }

        std::free(raw);
    }

} // namespace

auto operator new(std::size_t size) -> void * { return allocate(size); }

auto operator new[](std::size_t size) -> void * { return allocate(size); }

auto operator delete(void *ptr) noexcept -> void { deallocate(ptr); }

auto operator delete[](void *ptr) noexcept -> void { deallocate(ptr); }

auto operator delete(void *ptr, std::size_t /*free_size*/) noexcept -> void { deallocate(ptr); }

auto operator delete[](void *ptr, std::size_t /*free_size*/) noexcept -> void { deallocate(ptr); }
