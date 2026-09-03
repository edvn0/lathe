#pragma once

#include <cstddef>
#include <cstdlib>
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

private:
    class Pool {
    public:
        auto intern(std::string_view value) -> std::string const &;

        void print_stats() const;

    private:
        mutable std::mutex mutex_;
        std::unordered_set<std::string> strings_;
        std::size_t total_requests_ = 0;
        std::size_t total_bytes_allocated_ = 0;
    };

    static auto pool() -> Pool &;
    std::string const *value_ = nullptr;
};
