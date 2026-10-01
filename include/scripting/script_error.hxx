#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

enum class ScriptErrorKind : std::uint8_t { syntax, runtime, timeout, memory, invalid_entity, busy };

struct ScriptError {
    ScriptErrorKind kind = ScriptErrorKind::runtime;
    // Without the "script:<line>: " prefix.
    std::string message;
    // 1-based, when Lua reported a position.
    std::optional<std::int32_t> line;
};

// "syntax error", "runtime error", "timeout", ...
[[nodiscard]] auto script_error_kind_name(ScriptErrorKind kind) noexcept -> std::string_view;

// "runtime error (line 3): attempt to call a nil value (field 'get_childs') (did you mean 'get_children_or_empty'?)"
[[nodiscard]] auto describe(ScriptError const &error) -> std::string;

struct LuaErrorLocation {
    std::optional<std::int32_t> line;
    std::string_view text;
};

// Splits "<chunk_name>:<digits>: <text>". Without that prefix, returns {nullopt, message}. Digits parsed with
// std::from_chars (bugprone-unchecked-string-to-number-conversion).
[[nodiscard]] auto split_lua_error_location(std::string_view message,
                                            std::string_view chunk_name) noexcept -> LuaErrorLocation;

// Optimal-string-alignment distance; accepts a candidate within max(1, name.size() / 3) edits, never an exact match.
[[nodiscard]] auto closest_name(std::string_view name,
                                std::span<std::string const> candidates) -> std::optional<std::string_view>;

// For "attempt to (call|index) a nil value ((global|field|method|local|upvalue) 'NAME')": if NAME is a blocked
// sandbox name (os, io, require, load, loadstring, dofile, loadfile, debug, package, coroutine, collectgarbage,
// utf8, _G), appends " ('NAME' is not available in scripts)"; else appends " (did you mean 'X'?)" from
// closest_name(). Other messages are returned unchanged.
[[nodiscard]] auto with_name_suggestion(std::string_view message,
                                        std::span<std::string const> known_names) -> std::string;
