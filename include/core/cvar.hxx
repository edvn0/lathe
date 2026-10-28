#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

class CommandLine;

// Where a CVar's value came from, weakest first: a later source overrides an earlier one.
enum class CVarSource : std::uint8_t {
    fallback,
    environment,
    command_line,
    runtime,
};

[[nodiscard]] auto to_string(CVarSource source) -> std::string_view;

// A configuration variable: one typed value that can be set from the environment (LATHE_<NAME>), the command line
// (--<name>, with '_' written as '-') or at run time, and read from anywhere. Declare one as a namespace-scope object;
// it registers itself with CVarRegistry::global() and shows up in --help.
class CVarBase {
public:
    CVarBase(CVarBase const &) = delete;
    auto operator=(CVarBase const &) -> CVarBase & = delete;
    CVarBase(CVarBase &&) = delete;
    auto operator=(CVarBase &&) -> CVarBase & = delete;

    virtual ~CVarBase();

    [[nodiscard]] auto name() const noexcept -> std::string_view { return name_; }
    [[nodiscard]] auto group() const noexcept -> std::string_view { return group_; }
    [[nodiscard]] auto cli_name() const -> std::string;
    [[nodiscard]] auto env_name() const -> std::string;
    [[nodiscard]] auto cli_only() const noexcept -> bool { return cli_only_; }
    [[nodiscard]] auto help() const -> std::string { return help_fn_ ? help_fn_() : std::string{help_}; }

    [[nodiscard]] auto source() const -> CVarSource;
    // True once something set it (anything past the default).
    [[nodiscard]] auto has_value() const -> bool { return source() != CVarSource::fallback; }

    // The type's name for a UI or help: on|off, N, PATH, a|b|c.
    [[nodiscard]] virtual auto hint() const -> std::string = 0;
    [[nodiscard]] virtual auto text() const -> std::string = 0;
    [[nodiscard]] virtual auto default_text() const -> std::string = 0;
    // A switch given with no value on the command line (--max-quality); others take one.
    [[nodiscard]] virtual auto is_flag() const -> bool { return false; }
    // Each use adds to the value (--benchmark-sweep, repeated).
    [[nodiscard]] virtual auto is_list() const -> bool { return false; }

    [[nodiscard]] virtual auto set_text(std::string_view text, CVarSource from) -> std::expected<void, std::string> = 0;
    virtual auto reset() -> void = 0;

protected:
    struct Meta {
        std::string_view name;
        std::string_view group;
        std::string_view help;
        std::function<std::string()> help_fn;
        bool cli_only = false;
    };

    explicit CVarBase(Meta meta);

    mutable std::mutex mutex_;
    CVarSource source_ = CVarSource::fallback;

private:
    std::string_view name_;
    std::string_view group_;
    std::string_view help_;
    std::function<std::string()> help_fn_;
    bool cli_only_;
};

template<typename T>
struct CVarInfo {
    std::string_view name;
    std::string_view group;
    std::string_view help;
    // Replaces `help` when the text is only known at run time (a list of the games, say).
    std::function<std::string()> help_fn{};
    // Overrides the hint derived from the type.
    std::string_view hint{};
    T default_value{};
    // Arithmetic values must lie in [min, max].
    std::optional<T> min{};
    std::optional<T> max{};
    // An enum's names.
    std::span<std::pair<std::string_view, T> const> choices{};
    // A string's (or list item's) extra check; the text is stored as given.
    std::function<std::expected<void, std::string>(std::string_view)> validator{};
    // A bool that takes on|off instead of being a bare switch: it can be turned off, and left unset.
    bool toggle = false;
    // What to show for the default when it depends on the run ("auto"), instead of the type's zero value.
    std::string_view default_label{};
    // Not read from the environment: an action to run, not a setting.
    bool cli_only = false;
};

template<typename T>
class CVar final : public CVarBase {
    static constexpr bool is_bool = std::is_same_v<T, bool>;
    static constexpr bool is_arithmetic = std::is_arithmetic_v<T> && !is_bool;
    static constexpr bool is_enum = std::is_enum_v<T>;
    static constexpr bool is_path = std::is_same_v<T, std::filesystem::path>;
    static constexpr bool is_text = std::is_same_v<T, std::string>;
    static constexpr bool is_strings = std::is_same_v<T, std::vector<std::string>>;

    static_assert(is_bool || is_arithmetic || is_enum || is_path || is_text || is_strings, "unsupported CVar type");

public:
    explicit CVar(CVarInfo<T> info);

    // The value, or the default while nothing has set it.
    [[nodiscard]] auto get() const -> T {
        std::scoped_lock const lock{mutex_};
        return value_;
    }

    // The value if something set it, else `fallback`: for settings whose default depends on the run.
    [[nodiscard]] auto value_or(T fallback) const -> T {
        std::scoped_lock const lock{mutex_};
        return source_ == CVarSource::fallback ? std::move(fallback) : value_;
    }

    [[nodiscard]] auto value() const -> std::optional<T> {
        std::scoped_lock const lock{mutex_};
        return source_ == CVarSource::fallback ? std::nullopt : std::optional<T>{value_};
    }

    auto set(T new_value, CVarSource from = CVarSource::runtime) -> void {
        std::scoped_lock const lock{mutex_};
        value_ = std::move(new_value);
        source_ = from;
    }

    [[nodiscard]] auto hint() const -> std::string override;
    // The value as text; while nothing has set it and the default depends on the run, that default's label.
    [[nodiscard]] auto text() const -> std::string override {
        std::scoped_lock const lock{mutex_};
        return source_ == CVarSource::fallback && !info_.default_label.empty() ? std::string{info_.default_label}
                                                                               : format(value_);
    }
    [[nodiscard]] auto default_text() const -> std::string override {
        return info_.default_label.empty() ? format(info_.default_value) : std::string{info_.default_label};
    }
    [[nodiscard]] auto is_flag() const -> bool override { return is_bool && !info_.toggle; }
    [[nodiscard]] auto is_list() const -> bool override { return is_strings; }

    [[nodiscard]] auto set_text(std::string_view text, CVarSource from) -> std::expected<void, std::string> override;

    auto reset() -> void override {
        std::scoped_lock const lock{mutex_};
        value_ = info_.default_value;
        source_ = CVarSource::fallback;
    }

private:
    [[nodiscard]] auto parse(std::string_view text, [[maybe_unused]] T const &current) const -> std::expected<T, std::string>;
    [[nodiscard]] auto format(T const &value) const -> std::string;

    CVarInfo<T> info_;
    T value_;
};

class CVarRegistry {
public:
    [[nodiscard]] static auto global() -> CVarRegistry &;

    auto add(CVarBase &cvar) -> void;
    auto remove(CVarBase &cvar) -> void;

    [[nodiscard]] auto find(std::string_view name) const -> CVarBase *;
    // Every registered variable in registration order.
    [[nodiscard]] auto all() const -> std::vector<CVarBase *>;

    // Reads LATHE_<NAME> for every variable that allows it. Returns what could not be applied, one message each.
    [[nodiscard]] auto apply_environment() -> std::vector<std::string>;

    // Adds every variable to `cli` as an option, in its group; a value given there overrides the environment.
    auto attach(CommandLine &cli) -> void;

    // Sets one by name, as a change made at run time (a console, a panel).
    [[nodiscard]] auto set(std::string_view name, std::string_view text) -> std::expected<void, std::string>;

    auto reset_all() -> void;

private:
    CVarRegistry() = default;

    mutable std::mutex mutex_;
    std::vector<CVarBase *> cvars_;
};

template<typename T>
CVar<T>::CVar(CVarInfo<T> info)
    : CVarBase(Meta{.name = info.name,
                    .group = info.group,
                    .help = info.help,
                    .help_fn = info.help_fn,
                    .cli_only = info.cli_only}),
      info_(std::move(info)), value_(info_.default_value) {
    CVarRegistry::global().add(*this);
}

template<typename T>
auto CVar<T>::hint() const -> std::string {
    if (!info_.hint.empty()) {
        return std::string{info_.hint};
    }

    if constexpr (is_bool) {
        return "on|off";
    } else if constexpr (is_arithmetic) {
        return "N";
    } else if constexpr (is_enum) {
        std::string hint;
        for (auto const &choice: info_.choices) {
            hint += hint.empty() ? "" : "|";
            hint += choice.first;
        }
        return hint;
    } else if constexpr (is_path) {
        return "PATH";
    } else {
        return "TEXT";
    }
}

template<typename T>
auto CVar<T>::format(T const &value) const -> std::string {
    if constexpr (is_bool) {
        return value ? "on" : "off";
    } else if constexpr (is_arithmetic) {
        return std::format("{}", value);
    } else if constexpr (is_enum) {
        for (auto const &choice: info_.choices) {
            if (choice.second == value) {
                return std::string{choice.first};
            }
        }
        return std::to_string(static_cast<std::underlying_type_t<T>>(value));
    } else if constexpr (is_path) {
        return value.generic_string();
    } else if constexpr (is_text) {
        return value;
    } else {
        std::string joined;
        for (auto const &item: value) {
            joined += joined.empty() ? "" : ",";
            joined += item;
        }
        return joined;
    }
}

template<typename T>
auto CVar<T>::parse(std::string_view text, [[maybe_unused]] T const &current) const -> std::expected<T, std::string> {
    if constexpr (is_bool) {
        if (text == "on" || text == "true" || text == "1" || text == "yes") {
            return true;
        }
        if (text == "off" || text == "false" || text == "0" || text == "no") {
            return false;
        }
        return std::unexpected(std::format("'{}' (expected on or off)", text));
    } else if constexpr (is_arithmetic) {
        T result{};
        auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);

        if (error != std::errc{} || end != text.data() + text.size()) {
            return std::unexpected(std::format("'{}' is not a number", text));
        }
        if constexpr (std::is_floating_point_v<T>) {
            if (!std::isfinite(result)) {
                return std::unexpected(std::format("'{}' is not a number", text));
            }
        }
        if ((info_.min && result < *info_.min) || (info_.max && result > *info_.max)) {
            return std::unexpected(std::format("'{}' is out of range ({} to {})", text,
                                               info_.min.value_or(std::numeric_limits<T>::lowest()),
                                               info_.max.value_or(std::numeric_limits<T>::max())));
        }
        return result;
    } else if constexpr (is_enum) {
        for (auto const &choice: info_.choices) {
            if (choice.first == text) {
                return choice.second;
            }
        }
        return std::unexpected(std::format("'{}' (expected {})", text, hint()));
    } else if constexpr (is_path) {
        if (text.empty()) {
            return std::unexpected(std::string{"needs a path"});
        }
        return std::filesystem::path{text};
    } else if constexpr (is_text) {
        if (info_.validator) {
            if (auto const checked = info_.validator(text); !checked) {
                return std::unexpected(checked.error());
            }
        }
        return std::string{text};
    } else {
        if (info_.validator) {
            if (auto const checked = info_.validator(text); !checked) {
                return std::unexpected(checked.error());
            }
        }
        auto items = current;
        items.emplace_back(text);
        return items;
    }
}

template<typename T>
auto CVar<T>::set_text(std::string_view text, CVarSource from) -> std::expected<void, std::string> {
    std::scoped_lock const lock{mutex_};

    // A list collects what the command line gives it, but starts over for a source that has not touched it yet.
    auto const &base = is_strings && source_ != from ? info_.default_value : value_;
    auto parsed = parse(text, base);

    if (!parsed) {
        return std::unexpected(std::move(parsed.error()));
    }

    value_ = std::move(*parsed);
    source_ = from;
    return {};
}
