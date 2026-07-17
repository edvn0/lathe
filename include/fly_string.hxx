#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>

class FlyString {
public:
    FlyString() noexcept = default;

    explicit FlyString(std::string_view value) : value_(&pool().intern(value)) {}

    [[nodiscard]]
    auto view() const noexcept -> std::string_view {
        return value_ != nullptr ? std::string_view{*value_} : std::string_view{};
    }

    [[nodiscard]]
    auto empty() const noexcept -> bool {
        return value_ == nullptr || value_->empty();
    }

    [[nodiscard]]
    auto c_str() const noexcept -> char const * {
        return value_ != nullptr ? value_->c_str() : "";
    }

    friend auto operator==(FlyString lhs, FlyString rhs) noexcept -> bool { return lhs.value_ == rhs.value_; }

private:
    class Pool {
    public:
        auto intern(std::string_view value) -> std::string const & {
            std::scoped_lock lock{mutex_};

            auto const [iterator, inserted] = strings_.emplace(value);

            return *iterator;
        }

    private:
        std::mutex mutex_;
        std::unordered_set<std::string> strings_;
    };

    static auto pool() -> Pool & {
        // Deliberately retained until process termination. This avoids
        // static-destruction ordering issues for global FlyString values.
        static auto *instance = new Pool{};
        return *instance;
    }

    std::string const *value_ = nullptr;
};

template<>
struct std::hash<FlyString> {
    auto operator()(FlyString value) const noexcept -> std::size_t {
        return std::hash<std::string_view>{}(value.view());
    }
};
