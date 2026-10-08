#pragma once

#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

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

template<typename Owner, typename HandleT, auto Release>
class Holder {
public:
    Holder() = default;

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

    auto reset() noexcept -> void {
        if (owner_ == nullptr) {
            return;
        }

        static_cast<void>(std::invoke(Release, *owner_, handle_));

        owner_ = nullptr;
        handle_ = {};
    }

    [[nodiscard]]
    auto detach() noexcept -> HandleT {
        owner_ = nullptr;

        return std::exchange(handle_, {});
    }

private:
    Owner *owner_ = nullptr;
    HandleT handle_{};
};
