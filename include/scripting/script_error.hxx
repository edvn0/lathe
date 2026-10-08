#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

enum class ScriptErrorKind : std::uint8_t { syntax, runtime, timeout, memory, invalid_entity, busy };

struct ScriptError {
    ScriptErrorKind kind = ScriptErrorKind::runtime;
    std::string message;
    std::optional<std::int32_t> line;
};

[[nodiscard]] auto script_error_kind_name(ScriptErrorKind kind) noexcept -> std::string_view;

[[nodiscard]] auto describe(ScriptError const &error) -> std::string;

struct LuaErrorLocation {
    std::optional<std::int32_t> line;
    std::string_view text;
};

[[nodiscard]] auto split_lua_error_location(std::string_view message,
                                            std::string_view chunk_name) noexcept -> LuaErrorLocation;

[[nodiscard]] auto closest_name(std::string_view name,
                                std::span<std::string const> candidates) -> std::optional<std::string_view>;

[[nodiscard]] auto with_name_suggestion(std::string_view message,
                                        std::span<std::string const> known_names) -> std::string;
