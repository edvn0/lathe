#pragma once

#include <cstdint>
#include <limits>
#include <utility>

#include "handle.hxx"

template<typename T, std::uint32_t Sentinel>
class ObjectPool;

/*
 * Move-only RAII owner for an allocation in ObjectPool<T, Sentinel>.
 *
 * Holder does not physically store T. ObjectPool owns the storage while
 * Holder owns the lifetime of the allocation.
 *
 * Destroying or resetting a Holder:
 *
 *   1. calls ObjectPool::release(handle)
 *   2. release() moves T out of the pool slot
 *   3. the returned T is destroyed
 *   4. the slot becomes available for reuse
 *
 * Handle<T> remains the cheap, copyable, non-owning reference to the
 * allocation. Holder<T> is the unique owner of that allocation's lifetime.
 *
 * The ObjectPool must outlive every Holder referring to it.
 */
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

    /*
     * Releases and destroys the owned T.
     *
     * ObjectPool::release() returns optional<T>. The temporary returned here
     * owns the moved-out value and is destroyed at the end of this statement,
     * invoking T::~T().
     */
    auto reset() noexcept -> void {
        if (pool_ == nullptr) {
            return;
        }

        (void) pool_->release(handle_);

        pool_ = nullptr;
        handle_ = {};
    }

    /*
     * Relinquishes ownership without releasing the ObjectPool allocation.
     *
     * The returned HandleT remains valid and whoever receives it becomes
     * responsible for eventually calling ObjectPool::release().
     *
     * Analogous to std::unique_ptr::release().
     */
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
