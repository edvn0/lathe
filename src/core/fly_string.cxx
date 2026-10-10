#include "core/fly_string.hxx"

#include <cstdio>

FlyString::FlyString() noexcept = default;

FlyString::FlyString(std::string_view value) : value_(&pool().intern(value)) {}

FlyString::FlyString(std::string &&value) : value_(&pool().intern(std::move(value))) {}

[[nodiscard]]
auto FlyString::view() const noexcept -> std::string_view {
    return value_ != nullptr ? std::string_view{*value_} : std::string_view{};
}

[[nodiscard]]
auto FlyString::empty() const noexcept -> bool {
    return value_ == nullptr || value_->empty();
}

[[nodiscard]]
auto FlyString::c_str() const noexcept -> char const * {
    return value_ != nullptr ? value_->c_str() : "";
}

auto FlyString::operator==(FlyString rhs) const noexcept -> bool { return value_ == rhs.value_; }

auto FlyString::Pool::intern(std::string_view value) -> std::string const & {
    std::scoped_lock lock{mutex_};
    auto const [iterator, inserted] = strings_.emplace(value);

    if (inserted) {
        characters_ += iterator->size();
    }

    return *iterator;
}

auto FlyString::Pool::intern(std::string &&value) -> std::string const & {
    std::scoped_lock lock{mutex_};
    auto const [iterator, inserted] = strings_.emplace(std::move(value));

    if (inserted) {
        characters_ += iterator->size();
    }

    return *iterator;
}

auto FlyString::Pool::stats() const -> PoolStats {
    std::scoped_lock lock{mutex_};
    return PoolStats{.strings = strings_.size(), .characters = characters_};
}

auto FlyString::pool_stats() -> PoolStats { return pool().stats(); }

auto FlyString::pool() -> Pool & {
    static auto *instance = []() {
        auto *p = new Pool{};
        return p;
    }();
    return *instance;
}
