#pragma once

#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

// Owner has a get(HandleT) returning a pointer to the held value, so a Holder can dereference.
template<typename Owner, typename HandleT>
concept HolderOwnerWithGet =
        requires(Owner &owner, HandleT handle) { requires std::is_pointer_v<decltype(owner.get(handle))>; };

template<typename Owner, typename HandleT>
concept HolderOwnerWithConstGet =
        requires(Owner const &owner, HandleT handle) { requires std::is_pointer_v<decltype(owner.get(handle))>; };

template<typename Owner, typename HandleT>
concept HolderOwnerWithContains = requires(Owner const &owner, HandleT handle) {
    { owner.contains(handle) } -> std::convertible_to<bool>;
};

// Move-only owner of a handle. Destroying or resetting the Holder hands the handle back through `Release`, a member
// of Owner such as &ImageStorage::destroy_image, so the owner's full teardown runs rather than a bare slot free.
// HandleT stays the copyable, non-owning reference.
//
// get(), operator* and operator-> exist when Owner has a get(HandleT) returning a pointer.
//
// Release runs immediately, with no deferral: for a GPU resource, drop the Holder only once the GPU is done with it.
// The owner must outlive, and not move under, every Holder referring to it.
template<typename Owner, typename HandleT, auto Release>
class Holder {
public:
    Holder() = default;

    // Takes ownership of `handle`, which must belong to `owner`. Like std::unique_ptr's pointer constructor.
    Holder(Owner &owner, HandleT handle) noexcept : owner_(&owner), handle_(handle) {}

    ~Holder() { reset(); }

    Holder(Holder const &) = delete;
    auto operator=(Holder const &) -> Holder & = delete;

    Holder(Holder &&other) noexcept :
        owner_(std::exchange(other.owner_, nullptr)), handle_(std::exchange(other.handle_, {})) {}

    auto operator=(Holder &&other) noexcept -> Holder & {
        if (this == &other) {
            return *this;
        }

        reset();

        owner_ = std::exchange(other.owner_, nullptr);
        handle_ = std::exchange(other.handle_, {});

        return *this;
    }

    [[nodiscard]]
    auto get() noexcept
        requires HolderOwnerWithGet<Owner, HandleT>
    {
        using PointerT = decltype(owner_->get(handle_));

        if (owner_ == nullptr) {
            return PointerT{nullptr};
        }

        return owner_->get(handle_);
    }

    [[nodiscard]]
    auto get() const noexcept
        requires HolderOwnerWithConstGet<Owner, HandleT>
    {
        auto const *owner = owner_;

        using PointerT = decltype(owner->get(handle_));

        if (owner == nullptr) {
            return PointerT{nullptr};
        }

        return owner->get(handle_);
    }

    [[nodiscard]]
    auto operator*() noexcept -> auto &
        requires HolderOwnerWithGet<Owner, HandleT>
    {
        return *get();
    }

    [[nodiscard]]
    auto operator*() const noexcept -> auto const &
        requires HolderOwnerWithConstGet<Owner, HandleT>
    {
        return *get();
    }

    [[nodiscard]]
    auto operator->() noexcept
        requires HolderOwnerWithGet<Owner, HandleT>
    {
        return get();
    }

    [[nodiscard]]
    auto operator->() const noexcept
        requires HolderOwnerWithConstGet<Owner, HandleT>
    {
        return get();
    }

    [[nodiscard]]
    auto handle() const noexcept -> HandleT {
        return handle_;
    }

    // True while the handle is still live in its owner, as far as the owner can tell.
    [[nodiscard]]
    explicit operator bool() const noexcept {
        if (owner_ == nullptr) {
            return false;
        }

        if constexpr (HolderOwnerWithContains<Owner, HandleT>) {
            return static_cast<Owner const *>(owner_)->contains(handle_);
        } else if constexpr (HolderOwnerWithGet<Owner, HandleT>) {
            return owner_->get(handle_) != nullptr;
        } else {
            return handle_.valid();
        }
    }

    // Releases the handle through the owner.
    auto reset() noexcept -> void {
        if (owner_ == nullptr) {
            return;
        }

        static_cast<void>(std::invoke(Release, *owner_, handle_));

        owner_ = nullptr;
        handle_ = {};
    }

    // Gives up ownership without releasing; the caller must release it. Like std::unique_ptr::release().
    [[nodiscard]]
    auto detach() noexcept -> HandleT {
        owner_ = nullptr;

        return std::exchange(handle_, {});
    }

private:
    Owner *owner_ = nullptr;
    HandleT handle_{};
};
