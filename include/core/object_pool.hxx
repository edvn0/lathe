#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "core/handle.hxx"
#include "core/holder.hxx"

// Fixed-capacity, generation-checked free-list pool behind every *Storage class that hands out generational
// handles.
//
// Knows nothing about Vulkan: release() moves the payload out so the wrapper can destroy it. Thread safety,
// reserved slots and dirty tracking are also left to the wrapper.
template<typename T, std::uint32_t Sentinel = std::numeric_limits<std::uint32_t>::max()>
class ObjectPool {
public:
    using HandleT = Handle<T, Sentinel>;

    ObjectPool() = default;

    ObjectPool(ObjectPool const &) = delete;
    auto operator=(ObjectPool const &) -> ObjectPool & = delete;

    ObjectPool(ObjectPool &&other) noexcept :
        slots_(std::move(other.slots_)), free_head_(std::exchange(other.free_head_, 0)),
        size_(std::exchange(other.size_, 0)) {}

    auto operator=(ObjectPool &&other) noexcept -> ObjectPool & {
        if (this == &other) {
            return *this;
        }

        slots_ = std::move(other.slots_);
        free_head_ = std::exchange(other.free_head_, 0);
        size_ = std::exchange(other.size_, 0);

        return *this;
    }

    [[nodiscard]]
    static auto create(std::uint32_t capacity) -> ObjectPool {
        ObjectPool pool;

        pool.slots_.resize(capacity);

        for (std::uint32_t index = 0; index < capacity; ++index) {
            pool.slots_[index].next_free = index + 1 < capacity ? index + 1 : capacity;
        }

        pool.free_head_ = 0;

        return pool;
    }

    // Reserves a slot and returns its handle and value. The value is T{} on a slot's first use, otherwise
    // whatever release() left behind, so the caller must overwrite every field it needs. nullopt when full.
    [[nodiscard]]
    auto allocate() -> std::optional<std::pair<HandleT, T &>> {
        if (free_head_ >= slots_.size()) {
            return std::nullopt;
        }

        auto const index = free_head_;
        auto &slot = slots_[index];

        free_head_ = slot.next_free;

        slot.next_free = static_cast<std::uint32_t>(slots_.size());
        slot.occupied = true;

        ++size_;

        return std::pair<HandleT, T &>{
                HandleT{.index = index, .generation = slot.generation},
                slot.value,
        };
    }

    // Moves the payload out, bumps the generation (never back to 0, which means "never allocated") and frees the
    // slot. nullopt for a stale or out-of-range handle.
    //
    // The moved-from value stays in the slot rather than being reset, so fields like a descriptor revision counter
    // survive reuse.
    [[nodiscard]]
    auto release(HandleT handle) -> std::optional<T> {
        auto *slot = slot_for(handle);

        if (slot == nullptr) {
            return std::nullopt;
        }

        auto value = std::move(slot->value);

        slot->occupied = false;

        ++slot->generation;

        if (slot->generation == 0) {
            slot->generation = 1;
        }

        slot->next_free = free_head_;
        free_head_ = handle.index;

        --size_;

        return value;
    }

    // Owns a slot; destroying or resetting it calls release() and destroys the value.
    using HolderT = Holder<ObjectPool, HandleT, &ObjectPool::release>;

    // allocate(), with the slot owned by the returned Holder.
    [[nodiscard]]
    auto acquire() -> std::optional<HolderT> {
        auto allocation = allocate();

        if (!allocation) {
            return std::nullopt;
        }

        // Built in place, so no temporary Holder is moved into the optional.
        return std::optional<HolderT>{
                std::in_place,
                *this,
                allocation->first,
        };
    }

    [[nodiscard]]
    auto get(HandleT handle) noexcept -> T * {
        auto *slot = slot_for(handle);
        return slot != nullptr ? &slot->value : nullptr;
    }

    [[nodiscard]]
    auto get(HandleT handle) const noexcept -> T const * {
        auto const *slot = slot_for(handle);
        return slot != nullptr ? &slot->value : nullptr;
    }

    [[nodiscard]]
    auto contains(HandleT handle) const noexcept -> bool {
        return get(handle) != nullptr;
    }

    // Raw-index access for code that iterates every slot by GPU index.
    [[nodiscard]]
    auto get_at(std::uint32_t index) noexcept -> T * {
        return index < slots_.size() ? &slots_[index].value : nullptr;
    }

    [[nodiscard]]
    auto get_at(std::uint32_t index) const noexcept -> T const * {
        return index < slots_.size() ? &slots_[index].value : nullptr;
    }

    [[nodiscard]]
    auto occupied_at(std::uint32_t index) const noexcept -> bool {
        return index < slots_.size() && slots_[index].occupied;
    }

    [[nodiscard]]
    auto generation_at(std::uint32_t index) const noexcept -> std::uint32_t {
        return index < slots_.size() ? slots_[index].generation : 0;
    }

    [[nodiscard]]
    auto size() const noexcept -> std::uint32_t {
        return size_;
    }

    [[nodiscard]]
    auto capacity() const noexcept -> std::uint32_t {
        return static_cast<std::uint32_t>(slots_.size());
    }

private:
    struct Slot {
        T value{};

        std::uint32_t generation = 1;
        std::uint32_t next_free = 0;

        bool occupied = false;
    };

    [[nodiscard]]
    auto slot_for(HandleT handle) noexcept -> Slot * {
        if (handle.index >= slots_.size()) {
            return nullptr;
        }

        auto &slot = slots_[handle.index];

        if (!slot.occupied || slot.generation != handle.generation) {
            return nullptr;
        }

        return &slot;
    }

    [[nodiscard]]
    auto slot_for(HandleT handle) const noexcept -> Slot const * {
        if (handle.index >= slots_.size()) {
            return nullptr;
        }

        auto const &slot = slots_[handle.index];

        if (!slot.occupied || slot.generation != handle.generation) {
            return nullptr;
        }

        return &slot;
    }

    std::vector<Slot> slots_;
    std::uint32_t free_head_ = 0;
    std::uint32_t size_ = 0;
};
