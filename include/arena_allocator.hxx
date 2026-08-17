#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t chunk_size = 64 * 1024) : chunk_size_(chunk_size) { allocate_chunk(); }

    ~ArenaAllocator() = default;

    ArenaAllocator(const ArenaAllocator &) = delete;
    ArenaAllocator &operator=(const ArenaAllocator &) = delete;

    template<typename T, typename... Args>
    T *construct(Args &&...args) {
        void *ptr = allocate(sizeof(T), alignof(T));
        return ::new (ptr) T(std::forward<Args>(args)...);
    }

    void clear() {
        chunks_.clear();
        current_chunk_ = nullptr;
        offset_ = 0;
        allocate_chunk();
    }

private:
    void *allocate(size_t size, size_t alignment) {
        // Align offset
        size_t current_ptr = reinterpret_cast<size_t>(current_chunk_ + offset_);
        size_t aligned_ptr = (current_ptr + alignment - 1) & ~(alignment - 1);
        size_t padding = aligned_ptr - current_ptr;

        if (offset_ + padding + size > chunk_size_) {
            // Allocate new chunk if this one is full
            allocate_chunk();
            return allocate(size, alignment);
        }

        offset_ += padding + size;
        return reinterpret_cast<void *>(aligned_ptr);
    }

    void allocate_chunk() {
        chunks_.push_back(std::make_unique<uint8_t[]>(chunk_size_));
        current_chunk_ = chunks_.back().get();
        offset_ = 0;
    }

    size_t chunk_size_;
    size_t offset_ = 0;
    uint8_t *current_chunk_ = nullptr;
    std::vector<std::unique_ptr<uint8_t[]>> chunks_;
};
