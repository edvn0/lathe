#include "scripting/script_error.hxx"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <format>
#include <system_error>
#include <vector>

namespace {
    // Globals a script might reach for that the sandbox leaves out on purpose.
    constexpr std::array<std::string_view, 13> blocked_names{
            "os",      "io",        "require",   "load",           "loadstring", "dofile", "loadfile",
            "debug",   "package",   "coroutine", "collectgarbage", "utf8",       "_G",
    };

    constexpr std::array<std::string_view, 2> nil_value_prefixes{
            "attempt to call a nil value (",
            "attempt to index a nil value (",
    };

    constexpr std::array<std::string_view, 5> variable_kinds{"global", "field", "method", "local", "upvalue"};

    // Optimal string alignment: Levenshtein plus adjacent transpositions, no substring edited twice.
    [[nodiscard]] auto osa_distance(std::string_view a, std::string_view b) -> std::size_t {
        auto const columns = b.size() + 1;
        std::vector<std::size_t> before_previous(columns, 0);
        std::vector<std::size_t> previous(columns, 0);
        std::vector<std::size_t> current(columns, 0);

        for (std::size_t j = 0; j < columns; ++j) {
            previous[j] = j;
        }

        for (std::size_t i = 1; i <= a.size(); ++i) {
            current[0] = i;
            for (std::size_t j = 1; j < columns; ++j) {
                auto const cost = a[i - 1] == b[j - 1] ? std::size_t{0} : std::size_t{1};
                current[j] = std::min({previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost});
                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) {
                    current[j] = std::min(current[j], before_previous[j - 2] + 1);
                }
            }
            std::swap(before_previous, previous);
            std::swap(previous, current);
        }

        return previous[b.size()];
    }

    // NAME from "attempt to (call|index) a nil value (<kind> 'NAME')", if the message has that shape.
    [[nodiscard]] auto find_nil_value_name(std::string_view message) noexcept -> std::optional<std::string_view> {
        for (auto const prefix: nil_value_prefixes) {
            auto const start = message.find(prefix);
            if (start == std::string_view::npos) {
                continue;
            }

            auto rest = message.substr(start + prefix.size());
            auto const space = rest.find(' ');
            if (space == std::string_view::npos) {
                return std::nullopt;
            }

            auto const kind = rest.substr(0, space);
            if (std::ranges::find(variable_kinds, kind) == variable_kinds.end()) {
                return std::nullopt;
            }

            rest = rest.substr(space + 1);
            if (!rest.starts_with('\'')) {
                return std::nullopt;
            }
            rest.remove_prefix(1);

            auto const end = rest.find("')");
            if (end == std::string_view::npos || end == 0) {
                return std::nullopt;
            }

            return rest.substr(0, end);
        }

        return std::nullopt;
    }
} // namespace

auto script_error_kind_name(ScriptErrorKind kind) noexcept -> std::string_view {
    switch (kind) {
        case ScriptErrorKind::syntax:
            return "syntax error";
        case ScriptErrorKind::runtime:
            return "runtime error";
        case ScriptErrorKind::timeout:
            return "timeout";
        case ScriptErrorKind::memory:
            return "out of memory";
        case ScriptErrorKind::invalid_entity:
            return "invalid entity";
        case ScriptErrorKind::busy:
            return "busy";
    }

    return "error";
}

auto describe(ScriptError const &error) -> std::string {
    if (error.line.has_value()) {
        return std::format("{} (line {}): {}", script_error_kind_name(error.kind), *error.line, error.message);
    }

    return std::format("{}: {}", script_error_kind_name(error.kind), error.message);
}

auto split_lua_error_location(std::string_view message, std::string_view chunk_name) noexcept -> LuaErrorLocation {
    LuaErrorLocation const unlocated{.line = std::nullopt, .text = message};

    if (!message.starts_with(chunk_name)) {
        return unlocated;
    }

    auto rest = message.substr(chunk_name.size());
    if (!rest.starts_with(':')) {
        return unlocated;
    }
    rest.remove_prefix(1);

    auto const digits_end = rest.find_first_not_of("0123456789");
    if (digits_end == 0 || digits_end == std::string_view::npos) {
        return unlocated;
    }

    auto const after_digits = rest.substr(digits_end);
    if (!after_digits.starts_with(": ")) {
        return unlocated;
    }

    std::int32_t line = 0;
    auto const *const first = rest.data();
    auto const *const last = rest.data() + digits_end;
    auto const [parsed_end, parse_error] = std::from_chars(first, last, line);
    if (parse_error != std::errc{} || parsed_end != last) {
        return unlocated;
    }

    return LuaErrorLocation{.line = line, .text = after_digits.substr(2)};
}

auto closest_name(std::string_view name, std::span<std::string const> candidates) -> std::optional<std::string_view> {
    if (name.empty()) {
        return std::nullopt;
    }

    auto const max_distance = std::max<std::size_t>(1, name.size() / 3);
    std::optional<std::string_view> best;
    auto best_distance = max_distance + 1;

    for (auto const &candidate: candidates) {
        if (candidate == name) {
            continue;
        }

        auto const size_difference =
                candidate.size() > name.size() ? candidate.size() - name.size() : name.size() - candidate.size();
        if (size_difference > max_distance) {
            continue;
        }

        auto const distance = osa_distance(name, candidate);
        if (distance < best_distance) {
            best_distance = distance;
            best = candidate;
        }
    }

    return best;
}

auto with_name_suggestion(std::string_view message, std::span<std::string const> known_names) -> std::string {
    auto const name = find_nil_value_name(message);
    if (!name.has_value()) {
        return std::string{message};
    }

    if (std::ranges::find(blocked_names, *name) != blocked_names.end()) {
        return std::format("{} ('{}' is not available in scripts)", message, *name);
    }

    if (auto const suggestion = closest_name(*name, known_names)) {
        return std::format("{} (did you mean '{}'?)", message, *suggestion);
    }

    return std::string{message};
}
