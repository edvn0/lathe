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

// Declarative command line on top of Lyra (exception-free, so it builds with -fno-exceptions). Options are registered
// under named groups, which --help prints as sections:
//
//   CommandLine cli{"lathe", "Vulkan renderer"};
//   auto presentation = cli.group("Presentation");
//   presentation.toggle("--vsync", "Cap the frame rate at the refresh rate", vsync);
//   presentation.value("--swapchain-images", "N", "Images to ask for", images, {.min = 2, .max = 8});
//   auto const outcome = cli.parse(args);
//
// Both `--name=value` and `--name value` are accepted. Unknown options are an error. Registered targets are written
// through references, so they must outlive parse(). A CommandLineGroup refers to its CommandLine and must not outlive it.
class CommandLine;

// One section of --help, handed out by CommandLine::group(). Cheap to copy; registers options on its CommandLine.
class [[nodiscard]] CommandLineGroup {
public:
    // Validates and stores one option value; the error becomes the parse error, prefixed with the option name.
    using Setter = std::function<std::expected<void, std::string>(std::string_view)>;

    template <typename T>
    struct Range {
        T min = std::numeric_limits<T>::lowest();
        T max = std::numeric_limits<T>::max();
    };

    // --name sets the target to true.
    auto flag(std::string_view name, std::string_view help, bool &target) -> void;

    // --name <hint> with a custom Setter. `repeatable` lets it appear more than once (the Setter sees each).
    auto option(std::string_view name, std::string_view hint, std::string_view help, Setter setter,
                bool repeatable = false) -> void;

    // --name on|off into an optional, left empty when the option is absent.
    auto toggle(std::string_view name, std::string_view help, std::optional<bool> &target) -> void;

    // --name <hint> parsed as a number, a string or a path, and range-checked for numbers.
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

    // --name a|b|c into an optional, picking from `choices` (which must outlive the CommandLine); the hint lists them.
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
        run, // arguments parsed; carry on
        help, // --help was given: print help_text() and exit successfully
    };

    CommandLine(std::string program, std::string description);
    ~CommandLine();

    CommandLine(CommandLine const &) = delete;
    auto operator=(CommandLine const &) -> CommandLine & = delete;
    CommandLine(CommandLine &&) = delete;
    auto operator=(CommandLine &&) -> CommandLine & = delete;

    // The --help section called `title`, created on first use. Sections print in creation order.
    [[nodiscard]]
    auto group(std::string_view title) -> CommandLineGroup;

    // A required positional argument, in registration order.
    auto positional(std::string_view name, std::string_view help, std::string &target) -> void;

    // `args` excludes the program name. Errors read like "--swapchain-images: 'x' is not a number".
    [[nodiscard]]
    auto parse(std::span<char const *const> args) -> std::expected<Outcome, std::string>;

    // parse() for main(): skips argv[0].
    [[nodiscard]]
    auto parse(int argc, char const *const *argv) -> std::expected<Outcome, std::string> {
        return parse(std::span<char const *const>{argv + 1, static_cast<std::size_t>(argc > 0 ? argc - 1 : 0)});
    }

    [[nodiscard]]
    auto help_text() const -> std::string;

    // Splits "a,b,c" on `separator`; a helper for list-valued Setters.
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
