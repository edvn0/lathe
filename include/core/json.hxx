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

    [[nodiscard]] auto as_bool(bool fallback = false) const noexcept -> bool;
    [[nodiscard]] auto as_number(double fallback = 0.0) const noexcept -> double;
    [[nodiscard]] auto as_string(std::string_view fallback = {}) const -> std::string;

    [[nodiscard]] auto items() const noexcept -> std::span<JsonValue const>;
    [[nodiscard]] auto members() const noexcept -> std::span<JsonMember const>;

    [[nodiscard]] auto find(std::string_view key) const noexcept -> JsonValue const *;

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

[[nodiscard]] auto json_escape(std::string_view text) -> std::string;

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
    auto value(std::string_view key, double number, int precision = 4) -> JsonWriter &;
    auto value(std::string_view key, float number, int precision = 4) -> JsonWriter & {
        return value(key, static_cast<double>(number), precision);
    }
    auto null(std::string_view key) -> JsonWriter &;

    auto numbers(std::string_view key, std::span<float const> values, int precision = 4) -> JsonWriter &;

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
