#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

//
// A bump-pointer allocator: `construct<T>()` never fails to return live
// memory and never runs a destructor for you. Every object constructed here
// MUST have its destructor called explicitly by the owner before the arena
// itself is destroyed (or the object is otherwise abandoned) -- the arena
// only ever grows chunks and frees them in bulk in its own destructor, it
// keeps no per-object bookkeeping to invoke destructors automatically.
// PhysicsWorld::Impl is the intended usage pattern: it placement-destroys
// every object it constructed here, in `remove_body()` for early teardown
// and in `~Impl()` for the rest, before the arena's chunks are freed.
//
class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t chunk_size = 64 * 1024) : chunk_size_(chunk_size) {
        allocate_chunk(chunk_size_);
    }

    ~ArenaAllocator() = default;

    ArenaAllocator(const ArenaAllocator &) = delete;
    ArenaAllocator &operator=(const ArenaAllocator &) = delete;

    template<typename T, typename... Args>
    T *construct(Args &&...args) {
        void *ptr = allocate(sizeof(T), alignof(T));
        return ::new (ptr) T(std::forward<Args>(args)...);
    }

    template<typename T, typename Base, typename... Args>
        requires std::is_base_of_v<Base, T>
    Base *construct_with_base(Args &&...args) {
        void *ptr = allocate(sizeof(T), alignof(T));
        Base *allocated = ::new (ptr) T(std::forward<Args>(args)...);
        return allocated;
    }

private:
    void *allocate(size_t size, size_t alignment) {
        // Align offset
        size_t current_ptr = reinterpret_cast<size_t>(current_chunk_ + offset_);
        size_t aligned_ptr = (current_ptr + alignment - 1) & ~(alignment - 1);
        size_t padding = aligned_ptr - current_ptr;

        if (offset_ + padding + size > current_chunk_capacity_) {
            // Allocate a new chunk if this one is full. A request bigger
            // than the configured default chunk size gets a dedicated chunk
            // sized to fit it (chunk_size_ itself is left alone, so later
            // normal-sized allocations don't inherit this one-off size).
            allocate_chunk(std::max(chunk_size_, size + alignment));
            return allocate(size, alignment);
        }

        offset_ += padding + size;
        return reinterpret_cast<void *>(aligned_ptr);
    }

    void allocate_chunk(size_t size) {
        chunks_.push_back(std::make_unique<uint8_t[]>(size));
        current_chunk_ = chunks_.back().get();
        offset_ = 0;
        current_chunk_capacity_ = size;
    }

    size_t chunk_size_;
    size_t offset_ = 0;
    size_t current_chunk_capacity_ = 0;
    uint8_t *current_chunk_ = nullptr;
    std::vector<std::unique_ptr<uint8_t[]>> chunks_;
};
