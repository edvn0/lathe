#pragma once

#include <cstdint>
#include <limits>
#include <utility>

#include "core/handle.hxx"

template<typename T, std::uint32_t Sentinel>
class ObjectPool;

// Move-only owner of an ObjectPool allocation. The pool stores the T; destroying or resetting the Holder
// releases the slot and destroys the value. Handle<T> stays the copyable, non-owning reference.
//
// The ObjectPool must outlive every Holder referring to it.
template<typename T, std::uint32_t Sentinel = std::numeric_limits<std::uint32_t>::max()>
class Holder {
public:
    using HandleT = Handle<T, Sentinel>;
    using PoolT = ObjectPool<T, Sentinel>;

    Holder() = default;

    ~Holder() { reset(); }

    Holder(Holder const &) = delete;
    auto operator=(Holder const &) -> Holder & = delete;

    Holder(Holder &&other) noexcept :
        pool_(std::exchange(other.pool_, nullptr)), handle_(std::exchange(other.handle_, {})) {}

    auto operator=(Holder &&other) noexcept -> Holder & {
        if (this == &other) {
            return *this;
        }

        reset();

        pool_ = std::exchange(other.pool_, nullptr);
        handle_ = std::exchange(other.handle_, {});

        return *this;
    }

    [[nodiscard]]
    auto get() noexcept -> T * {
        if (pool_ == nullptr) {
            return nullptr;
        }

        return pool_->get(handle_);
    }

    [[nodiscard]]
    auto get() const noexcept -> T const * {
        if (pool_ == nullptr) {
            return nullptr;
        }

        return pool_->get(handle_);
    }

    [[nodiscard]]
    auto operator*() noexcept -> T & {
        return *get();
    }

    [[nodiscard]]
    auto operator*() const noexcept -> T const & {
        return *get();
    }

    [[nodiscard]]
    auto operator->() noexcept -> T * {
        return get();
    }

    [[nodiscard]]
    auto operator->() const noexcept -> T const * {
        return get();
    }

    [[nodiscard]]
    auto handle() const noexcept -> HandleT {
        return handle_;
    }

    [[nodiscard]]
    explicit operator bool() const noexcept {
        return pool_ != nullptr && pool_->contains(handle_);
    }

    // Releases the slot and destroys the value.
    auto reset() noexcept -> void {
        if (pool_ == nullptr) {
            return;
        }

        (void) pool_->release(handle_);

        pool_ = nullptr;
        handle_ = {};
    }

    // Gives up ownership without releasing the slot; the caller must release it. Like std::unique_ptr::release().
    [[nodiscard]]
    auto detach() noexcept -> HandleT {
        pool_ = nullptr;

        return std::exchange(handle_, {});
    }

private:
    friend class ObjectPool<T, Sentinel>;

    Holder(PoolT &pool, HandleT handle) noexcept : pool_(&pool), handle_(handle) {}

    PoolT *pool_ = nullptr;
    HandleT handle_{};
};
