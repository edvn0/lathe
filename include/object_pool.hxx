#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "handle.hxx"
#include "holder.hxx"

/*
 * Generic fixed-capacity, generation-checked free-list pool ("bindless
 * arena") shared by every *Storage class that hands out a generational
 * handle to a slot it owns (ImageStorage, MaterialStorage, SamplerStorage,
 * PipelineStorage, ShaderObjectStorage, MeshStorage, ModelStorage).
 *
 * Deliberately knows nothing about Vulkan or domain-specific cleanup:
 * release() moves the payload back out to the caller instead of destroying
 * it in place, so a wrapper storage can run vkDestroySampler, Pipeline's own
 * destroy(), etc. on the returned value. This keeps ObjectPool a plain data
 * structure
 * that's fully unit-testable without a VulkanContext -- see
 * test/object_pool_test.cxx.
 *
 * Thread safety, "protected"/permanently-reserved slots, and dirty/revision
 * bits are likewise left to the wrapper: T carries whatever extra
 * bookkeeping a given domain needs, and a wrapper that mutates the pool from
 * multiple threads (PipelineStorage, ShaderObjectStorage) supplies its own
 * mutex around allocate()/release().
 */
template<typename T, std::uint32_t Sentinel = std::numeric_limits<std::uint32_t>::max()>
class ObjectPool {
public:
    using HandleT = Handle<T, Sentinel>;
    using HolderT = Holder<T, Sentinel>;

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

    // Reserves a slot and returns a handle plus a reference to the slot's
    // value for the caller to fill in. The value starts at T{} the first
    // time a given slot is ever used, and otherwise holds whatever a prior
    // release() left behind (see release()'s comment) -- the caller is
    // responsible for overwriting every field a fresh occupant needs.
    // std::nullopt means the free list is exhausted -- translating that into
    // a domain-specific capacity_exceeded error is the caller's job.
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

    [[nodiscard]]
    auto acquire() -> std::optional<HolderT> {
        auto allocation = allocate();

        if (!allocation) {
            return std::nullopt;
        }

        return HolderT{
                *this,
                allocation->first,
        };
    }

    // Validates handle, moves its payload out, bumps the slot's generation
    // (skipping the 0 wraparound so generation 0 always stays "never
    // allocated"), and returns the slot to the free list. std::nullopt for a
    // stale or out-of-range handle -- the pool never double-frees.
    //
    // Deliberately leaves whatever std::move() left behind in the slot's
    // value untouched (rather than resetting it to T{}) -- some payloads
    // carry fields that must survive a release/reuse cycle unmodified (e.g.
    // a monotonically-bumped descriptor revision counter that must never
    // collide with a value some in-flight frame already cached). It is the
    // wrapper storage's job to overwrite every field a fresh occupant needs
    // on its next allocate(), exactly as it already does today.
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

    // Raw-index accessors for call sites that iterate every slot by GPU
    // index rather than through a validated handle (e.g.
    // ImageStorage::descriptor_record / SamplerStorage::descriptor_record).
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
