#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

// Bump allocator that never runs destructors: owners must destroy every object explicitly before the arena
// goes away. Chunks are freed in bulk.
class ArenaAllocator {
public:
    explicit ArenaAllocator(std::size_t chunk_size = std::size_t{64} * 1024) : chunk_size_(chunk_size) {
        allocate_chunk(chunk_size_);
    }

    ArenaAllocator(ArenaAllocator const &) = delete;
    auto operator=(ArenaAllocator const &) -> ArenaAllocator & = delete;

    template<typename T, typename... Args>
    auto construct(Args &&...args) -> T * {
        void *ptr = allocate(sizeof(T), alignof(T));
        return ::new (ptr) T(std::forward<Args>(args)...);
    }

    template<typename T, typename Base, typename... Args>
        requires std::is_base_of_v<Base, T>
    auto construct_with_base(Args &&...args) -> Base * {
        void *ptr = allocate(sizeof(T), alignof(T));
        return ::new (ptr) T(std::forward<Args>(args)...);
    }

private:
    auto allocate(std::size_t size, std::size_t alignment) -> void * {
        auto const current_ptr = reinterpret_cast<std::uintptr_t>(current_chunk_ + offset_);
        auto const aligned_ptr = (current_ptr + alignment - 1) & ~(alignment - 1);
        auto const padding = aligned_ptr - current_ptr;

        if (offset_ + padding + size > current_chunk_capacity_) {
            // Oversized requests get a dedicated chunk; chunk_size_ is unchanged.
            allocate_chunk(std::max(chunk_size_, size + alignment));
            return allocate(size, alignment);
        }

        void *result = current_chunk_ + offset_ + padding;
        offset_ += padding + size;
        return result;
    }

    auto allocate_chunk(std::size_t size) -> void {
        current_chunk_ = chunks_.emplace_back(size).data();
        offset_ = 0;
        current_chunk_capacity_ = size;
    }

    std::size_t chunk_size_;
    std::size_t offset_ = 0;
    std::size_t current_chunk_capacity_ = 0;
    std::byte *current_chunk_ = nullptr;
    // Chunk buffers never move: growing the outer vector moves the inner vectors, not their storage.
    std::vector<std::vector<std::byte>> chunks_;
};
