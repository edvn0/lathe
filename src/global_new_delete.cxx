// Global operator new/delete overrides, feeding every heap allocation in the
// process (this engine plus every statically-linked third-party library) to
// Tracy's memory profiler. TracyAlloc/TracyFree are no-ops when TRACY_ENABLE
// isn't defined, so this file behaves as a plain malloc/free passthrough in
// that configuration.
//
// MINGW_VULKAN_ENABLE_EXCEPTIONS is off project-wide, so allocation failure
// aborts instead of throwing std::bad_alloc.

#include <tracy/Tracy.hpp>

#include <cstdio>
#include <cstdlib>

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

    TracyAlloc(ptr, size);

    return ptr;
}

auto operator new[](std::size_t size) -> void * {
    void *ptr = std::malloc(size);
    if (ptr == nullptr) {
        abort_out_of_memory(size);
    }

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

auto operator delete(void *ptr, std::size_t) noexcept -> void {
    TracyFree(ptr);
    std::free(ptr);
}

auto operator delete[](void *ptr, std::size_t) noexcept -> void {
    TracyFree(ptr);
    std::free(ptr);
}
