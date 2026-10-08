#pragma once

#include <cstddef>
#include <cstdlib>
#include <format>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>

class FlyString {
public:
    FlyString() noexcept;
    explicit FlyString(std::string_view);

    [[nodiscard]]
    auto view() const noexcept -> std::string_view;
    [[nodiscard]]
    auto empty() const noexcept -> bool;
    [[nodiscard]]
    auto c_str() const noexcept -> char const *;

    auto operator==(FlyString rhs) const noexcept -> bool;

    struct PoolStats {
        std::size_t strings = 0;
        std::size_t characters = 0;
    };

    [[nodiscard]]
    static auto pool_stats() -> PoolStats;
    auto operator==(std::string_view rhs) const noexcept -> bool { return view() == rhs; }

    [[nodiscard]]
    auto identity() const noexcept -> std::size_t {
        return std::hash<std::string const *>{}(value_);
    }

private:
    class Pool {
    public:
        auto intern(std::string_view value) -> std::string const &;
        [[nodiscard]]
        auto stats() const -> PoolStats;

    private:
        mutable std::mutex mutex_;
        std::unordered_set<std::string> strings_;
        std::size_t characters_ = 0;
    };

    static auto pool() -> Pool &;
    std::string const *value_ = nullptr;
};

template<>
struct std::hash<FlyString> {
    [[nodiscard]]
    auto operator()(FlyString const &value) const noexcept -> std::size_t {
        return value.identity();
    }
};

template<>
struct std::formatter<FlyString> : std::formatter<std::string_view> {
    auto format(FlyString const &value, std::format_context &context) const {
        return std::formatter<std::string_view>::format(value.view(), context);
    }
};
