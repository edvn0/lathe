#pragma once

#include <charconv>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

class CommandLine;

class [[nodiscard]] CommandLineGroup {
public:
    using Setter = std::function<std::expected<void, std::string>(std::string_view)>;

    template <typename T>
    struct Range {
        T min = std::numeric_limits<T>::lowest();
        T max = std::numeric_limits<T>::max();
    };

    auto flag(std::string_view name, std::string_view help, bool &target) -> void;

    auto option(std::string_view name, std::string_view hint, std::string_view help, Setter setter,
                bool repeatable = false) -> void;

    auto toggle(std::string_view name, std::string_view help, std::optional<bool> &target) -> void;

    template <typename T>
    auto value(std::string_view name, std::string_view hint, std::string_view help, T &target,
               Range<T> range = {}) -> void {
        option(name, hint, help, [&target, range](std::string_view text) -> std::expected<void, std::string> {
            auto parsed = parse_value<T>(text, range);
            if (!parsed) {
                return std::unexpected(std::move(parsed.error()));
            }
            target = std::move(*parsed);
            return {};
        });
    }

    template <typename T>
    auto value(std::string_view name, std::string_view hint, std::string_view help, std::optional<T> &target,
               Range<T> range = {}) -> void {
        option(name, hint, help, [&target, range](std::string_view text) -> std::expected<void, std::string> {
            auto parsed = parse_value<T>(text, range);
            if (!parsed) {
                return std::unexpected(std::move(parsed.error()));
            }
            target = std::move(*parsed);
            return {};
        });
    }

    template <typename T>
    auto choice(std::string_view name, std::string_view help, std::span<std::pair<std::string_view, T> const> choices,
                std::optional<T> &target) -> void {
        std::string hint;
        for (auto const &entry: choices) {
            hint += hint.empty() ? "" : "|";
            hint += entry.first;
        }
        option(name, hint, help,
               [&target, choices, hint](std::string_view text) -> std::expected<void, std::string> {
                   for (auto const &entry: choices) {
                       if (entry.first == text) {
                           target = entry.second;
                           return {};
                       }
                   }
                   return std::unexpected("'" + std::string{text} + "' (expected " + hint + ")");
               });
    }

private:
    friend class CommandLine;

    CommandLineGroup(CommandLine &owner, std::size_t index) : owner_(&owner), index_(index) {}

    template <typename T>
    [[nodiscard]]
    static auto parse_value(std::string_view text, Range<T> const &range) -> std::expected<T, std::string> {
        if constexpr (std::is_same_v<T, std::string>) {
            return std::string{text};
        } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
            if (text.empty()) {
                return std::unexpected(std::string{"needs a path"});
            }
            return std::filesystem::path{text};
        } else {
            static_assert(std::is_arithmetic_v<T>, "unsupported option value type");

            T result{};
            auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);

            if (error != std::errc{} || end != text.data() + text.size()) {
                return std::unexpected("'" + std::string{text} + "' is not a number");
            }
            if (result < range.min || result > range.max) {
                return std::unexpected("'" + std::string{text} + "' is out of range (" + std::to_string(range.min) +
                                       " to " + std::to_string(range.max) + ")");
            }
            return result;
        }
    }

    CommandLine *owner_;
    std::size_t index_;
};

class CommandLine {
public:
    enum class Outcome : std::uint8_t {
        run,
        help,
    };

    CommandLine(std::string program, std::string description);
    ~CommandLine();

    CommandLine(CommandLine const &) = delete;
    auto operator=(CommandLine const &) -> CommandLine & = delete;
    CommandLine(CommandLine &&) = delete;
    auto operator=(CommandLine &&) -> CommandLine & = delete;

    [[nodiscard]]
    auto group(std::string_view title) -> CommandLineGroup;

    auto positional(std::string_view name, std::string_view help, std::string &target) -> void;

    [[nodiscard]]
    auto parse(std::span<char const *const> args) -> std::expected<Outcome, std::string>;

    [[nodiscard]]
    auto parse(int argc, char const *const *argv) -> std::expected<Outcome, std::string> {
        return parse(std::span<char const *const>{argv + 1, static_cast<std::size_t>(argc > 0 ? argc - 1 : 0)});
    }

    [[nodiscard]]
    auto help_text() const -> std::string;

    [[nodiscard]]
    static auto split(std::string_view text, char separator) -> std::vector<std::string_view>;

private:
    friend class CommandLineGroup;

    auto add_flag(std::size_t group, std::string_view name, std::string_view help, bool &target) -> void;
    auto add_option(std::size_t group, std::string_view name, std::string_view hint, std::string_view help,
                    CommandLineGroup::Setter setter, bool repeatable) -> void;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
