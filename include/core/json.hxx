#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// A small JSON reader and writer for the engine's own result files (benchmark output), so tools that read them back
// can live in the engine instead of in scripts. Not a general-purpose library: numbers are doubles, objects keep
// their key order and allow duplicate keys (lookups return the first), and there are no exceptions.

class JsonValue;

struct JsonMember;

using JsonArray = std::vector<JsonValue>;
using JsonObject = std::vector<JsonMember>;

class JsonValue {
public:
    using Storage = std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject>;

    JsonValue() noexcept;
    explicit JsonValue(Storage storage) noexcept;

    [[nodiscard]] auto is_null() const noexcept -> bool { return std::holds_alternative<std::nullptr_t>(storage_); }
    [[nodiscard]] auto is_bool() const noexcept -> bool { return std::holds_alternative<bool>(storage_); }
    [[nodiscard]] auto is_number() const noexcept -> bool { return std::holds_alternative<double>(storage_); }
    [[nodiscard]] auto is_string() const noexcept -> bool { return std::holds_alternative<std::string>(storage_); }
    [[nodiscard]] auto is_array() const noexcept -> bool { return std::holds_alternative<JsonArray>(storage_); }
    [[nodiscard]] auto is_object() const noexcept -> bool { return std::holds_alternative<JsonObject>(storage_); }

    // Typed reads with a fallback for a missing or differently typed value, so readers can be lenient about files
    // written by older builds.
    [[nodiscard]] auto as_bool(bool fallback = false) const noexcept -> bool;
    [[nodiscard]] auto as_number(double fallback = 0.0) const noexcept -> double;
    [[nodiscard]] auto as_string(std::string_view fallback = {}) const -> std::string;

    // Empty when not an array / object.
    [[nodiscard]] auto items() const noexcept -> std::span<JsonValue const>;
    [[nodiscard]] auto members() const noexcept -> std::span<JsonMember const>;

    // Object member by key, or nullptr when absent or not an object.
    [[nodiscard]] auto find(std::string_view key) const noexcept -> JsonValue const *;

    // find(key), or a shared null value: chains like value["a"]["b"].as_number() never dereference null.
    [[nodiscard]] auto operator[](std::string_view key) const noexcept -> JsonValue const &;

private:
    Storage storage_;
};

struct JsonMember {
    std::string key;
    JsonValue value;
};

struct JsonParseError {
    std::size_t offset = 0;
    std::string message;
};

[[nodiscard]] auto parse_json(std::string_view text) -> std::expected<JsonValue, JsonParseError>;

// Escapes `text` for use inside a JSON string literal (quotes not included).
[[nodiscard]] auto json_escape(std::string_view text) -> std::string;

// Streaming writer: two-space indented, with containers optionally kept on one line (`inline_container`), which keeps
// long per-frame arrays and small records readable. Keys are ignored inside arrays and required inside objects.
class JsonWriter {
public:
    auto begin_object(std::string_view key = {}, bool inline_container = false) -> JsonWriter &;
    auto end_object() -> JsonWriter &;
    auto begin_array(std::string_view key = {}, bool inline_container = false) -> JsonWriter &;
    auto end_array() -> JsonWriter &;

    auto value(std::string_view key, std::string_view text) -> JsonWriter &;
    auto value(std::string_view key, char const *text) -> JsonWriter & { return value(key, std::string_view{text}); }
    auto value(std::string_view key, bool flag) -> JsonWriter &;
    auto value(std::string_view key, std::int64_t number) -> JsonWriter &;
    auto value(std::string_view key, std::uint64_t number) -> JsonWriter &;
    auto value(std::string_view key, std::uint32_t number) -> JsonWriter & {
        return value(key, static_cast<std::uint64_t>(number));
    }
    auto value(std::string_view key, std::int32_t number) -> JsonWriter & {
        return value(key, static_cast<std::int64_t>(number));
    }
    // Fixed-point with `precision` decimals; NaN and infinities are written as null.
    auto value(std::string_view key, double number, int precision = 4) -> JsonWriter &;
    auto value(std::string_view key, float number, int precision = 4) -> JsonWriter & {
        return value(key, static_cast<double>(number), precision);
    }
    auto null(std::string_view key) -> JsonWriter &;

    // An inline array of numbers.
    auto numbers(std::string_view key, std::span<float const> values, int precision = 4) -> JsonWriter &;

    // The document so far, newline-terminated once the outermost container is closed.
    [[nodiscard]] auto str() const -> std::string const & { return out_; }

private:
    struct Level {
        bool is_object = false;
        bool inline_container = false;
        bool empty = true;
    };

    auto prefix(std::string_view key) -> void;
    auto close(char bracket) -> void;

    std::string out_;
    std::vector<Level> stack_;
};
