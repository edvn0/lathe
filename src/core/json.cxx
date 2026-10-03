#include "core/json.hxx"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <format>
#include <system_error>

JsonValue::JsonValue() noexcept : storage_(nullptr) {}

JsonValue::JsonValue(Storage storage) noexcept : storage_(std::move(storage)) {}

auto JsonValue::as_bool(bool fallback) const noexcept -> bool {
    auto const *flag = std::get_if<bool>(&storage_);
    return flag != nullptr ? *flag : fallback;
}

auto JsonValue::as_number(double fallback) const noexcept -> double {
    auto const *number = std::get_if<double>(&storage_);
    return number != nullptr ? *number : fallback;
}

auto JsonValue::as_string(std::string_view fallback) const -> std::string {
    auto const *text = std::get_if<std::string>(&storage_);
    return text != nullptr ? *text : std::string{fallback};
}

auto JsonValue::items() const noexcept -> std::span<JsonValue const> {
    auto const *array = std::get_if<JsonArray>(&storage_);
    return array != nullptr ? std::span<JsonValue const>{*array} : std::span<JsonValue const>{};
}

auto JsonValue::members() const noexcept -> std::span<JsonMember const> {
    auto const *object = std::get_if<JsonObject>(&storage_);
    return object != nullptr ? std::span<JsonMember const>{*object} : std::span<JsonMember const>{};
}

auto JsonValue::find(std::string_view key) const noexcept -> JsonValue const * {
    for (auto const &member: members()) {
        if (member.key == key) {
            return &member.value;
        }
    }
    return nullptr;
}

auto JsonValue::operator[](std::string_view key) const noexcept -> JsonValue const & {
    static JsonValue const null_value{};
    auto const *found = find(key);
    return found != nullptr ? *found : null_value;
}

namespace {

    // Containers deeper than this are refused rather than recursed into.
    constexpr std::size_t maximum_depth = 256;

    class Parser {
    public:
        explicit Parser(std::string_view text) noexcept : text_(text) {}

        auto parse_document() -> std::expected<JsonValue, JsonParseError> {
            auto value = parse_value(0);
            if (!value) {
                return value;
            }
            skip_whitespace();
            if (position_ != text_.size()) {
                return fail("trailing characters after the document");
            }
            return value;
        }

    private:
        [[nodiscard]] auto fail(std::string_view message) const -> std::unexpected<JsonParseError> {
            return std::unexpected(JsonParseError{.offset = position_, .message = std::string{message}});
        }

        auto skip_whitespace() noexcept -> void {
            while (position_ < text_.size()) {
                auto const character = text_[position_];
                if (character != ' ' && character != '\t' && character != '\n' && character != '\r') {
                    return;
                }
                ++position_;
            }
        }

        [[nodiscard]] auto consume_literal(std::string_view literal) noexcept -> bool {
            if (text_.substr(position_, literal.size()) != literal) {
                return false;
            }
            position_ += literal.size();
            return true;
        }

        auto parse_value(std::size_t depth) -> std::expected<JsonValue, JsonParseError> {
            if (depth > maximum_depth) {
                return fail("nesting too deep");
            }

            skip_whitespace();
            if (position_ >= text_.size()) {
                return fail("unexpected end of input");
            }

            switch (text_[position_]) {
                case '{':
                    return parse_object(depth);
                case '[':
                    return parse_array(depth);
                case '"': {
                    auto text = parse_string();
                    if (!text) {
                        return std::unexpected(text.error());
                    }
                    return JsonValue{std::move(*text)};
                }
                case 't':
                    if (consume_literal("true")) {
                        return JsonValue{true};
                    }
                    return fail("invalid literal");
                case 'f':
                    if (consume_literal("false")) {
                        return JsonValue{false};
                    }
                    return fail("invalid literal");
                case 'n':
                    if (consume_literal("null")) {
                        return JsonValue{};
                    }
                    return fail("invalid literal");
                default:
                    return parse_number();
            }
        }

        auto parse_number() -> std::expected<JsonValue, JsonParseError> {
            auto const start = position_;
            while (position_ < text_.size()) {
                auto const character = text_[position_];
                auto const numeric = (character >= '0' && character <= '9') || character == '-' || character == '+' ||
                                     character == '.' || character == 'e' || character == 'E';
                if (!numeric) {
                    break;
                }
                ++position_;
            }

            if (start == position_) {
                return fail("unexpected character");
            }

            auto number = 0.0;
            auto const *first = text_.data() + start;
            auto const *last = text_.data() + position_;
            auto const [end, error] = std::from_chars(first, last, number);
            if (error != std::errc{} || end != last) {
                position_ = start;
                return fail("invalid number");
            }
            return JsonValue{number};
        }

        [[nodiscard]] static auto hex_digit(char character) noexcept -> int {
            if (character >= '0' && character <= '9') {
                return character - '0';
            }
            if (character >= 'a' && character <= 'f') {
                return character - 'a' + 10;
            }
            if (character >= 'A' && character <= 'F') {
                return character - 'A' + 10;
            }
            return -1;
        }

        auto parse_hex4() -> std::expected<std::uint32_t, JsonParseError> {
            if (position_ + 4 > text_.size()) {
                return fail("truncated \\u escape");
            }
            std::uint32_t code = 0;
            for (std::size_t i = 0; i < 4; ++i) {
                auto const digit = hex_digit(text_[position_ + i]);
                if (digit < 0) {
                    return fail("invalid \\u escape");
                }
                code = (code << 4U) | static_cast<std::uint32_t>(digit);
            }
            position_ += 4;
            return code;
        }

        static auto append_utf8(std::string &out, std::uint32_t code) -> void {
            if (code < 0x80U) {
                out.push_back(static_cast<char>(code));
            } else if (code < 0x800U) {
                out.push_back(static_cast<char>(0xC0U | (code >> 6U)));
                out.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
            } else if (code < 0x10000U) {
                out.push_back(static_cast<char>(0xE0U | (code >> 12U)));
                out.push_back(static_cast<char>(0x80U | ((code >> 6U) & 0x3FU)));
                out.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
            } else {
                out.push_back(static_cast<char>(0xF0U | (code >> 18U)));
                out.push_back(static_cast<char>(0x80U | ((code >> 12U) & 0x3FU)));
                out.push_back(static_cast<char>(0x80U | ((code >> 6U) & 0x3FU)));
                out.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
            }
        }

        auto parse_string() -> std::expected<std::string, JsonParseError> {
            ++position_; // opening quote
            std::string out;

            while (position_ < text_.size()) {
                auto const character = text_[position_++];

                if (character == '"') {
                    return out;
                }

                if (static_cast<unsigned char>(character) < 0x20U) {
                    --position_;
                    return fail("control character in string");
                }

                if (character != '\\') {
                    out.push_back(character);
                    continue;
                }

                if (position_ >= text_.size()) {
                    break;
                }

                switch (text_[position_++]) {
                    case '"':
                        out.push_back('"');
                        break;
                    case '\\':
                        out.push_back('\\');
                        break;
                    case '/':
                        out.push_back('/');
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    case 'u': {
                        auto code = parse_hex4();
                        if (!code) {
                            return std::unexpected(code.error());
                        }
                        // A high surrogate followed by an escaped low one is one code point.
                        if (*code >= 0xD800U && *code <= 0xDBFFU && text_.substr(position_, 2) == "\\u") {
                            position_ += 2;
                            auto low = parse_hex4();
                            if (!low) {
                                return std::unexpected(low.error());
                            }
                            if (*low >= 0xDC00U && *low <= 0xDFFFU) {
                                *code = 0x10000U + ((*code - 0xD800U) << 10U) + (*low - 0xDC00U);
                            } else {
                                append_utf8(out, *code);
                                *code = *low;
                            }
                        }
                        append_utf8(out, *code);
                        break;
                    }
                    default:
                        --position_;
                        return fail("invalid escape");
                }
            }

            return fail("unterminated string");
        }

        auto parse_array(std::size_t depth) -> std::expected<JsonValue, JsonParseError> {
            ++position_; // [
            JsonArray array;

            skip_whitespace();
            if (position_ < text_.size() && text_[position_] == ']') {
                ++position_;
                return JsonValue{std::move(array)};
            }

            while (true) {
                auto item = parse_value(depth + 1);
                if (!item) {
                    return item;
                }
                array.push_back(std::move(*item));

                skip_whitespace();
                if (position_ >= text_.size()) {
                    return fail("unterminated array");
                }
                auto const separator = text_[position_++];
                if (separator == ']') {
                    return JsonValue{std::move(array)};
                }
                if (separator != ',') {
                    --position_;
                    return fail("expected ',' or ']'");
                }
            }
        }

        auto parse_object(std::size_t depth) -> std::expected<JsonValue, JsonParseError> {
            ++position_; // {
            JsonObject object;

            skip_whitespace();
            if (position_ < text_.size() && text_[position_] == '}') {
                ++position_;
                return JsonValue{std::move(object)};
            }

            while (true) {
                skip_whitespace();
                if (position_ >= text_.size() || text_[position_] != '"') {
                    return fail("expected a key");
                }
                auto key = parse_string();
                if (!key) {
                    return std::unexpected(key.error());
                }

                skip_whitespace();
                if (position_ >= text_.size() || text_[position_] != ':') {
                    return fail("expected ':'");
                }
                ++position_;

                auto member_value = parse_value(depth + 1);
                if (!member_value) {
                    return member_value;
                }
                object.push_back(JsonMember{.key = std::move(*key), .value = std::move(*member_value)});

                skip_whitespace();
                if (position_ >= text_.size()) {
                    return fail("unterminated object");
                }
                auto const separator = text_[position_++];
                if (separator == '}') {
                    return JsonValue{std::move(object)};
                }
                if (separator != ',') {
                    --position_;
                    return fail("expected ',' or '}'");
                }
            }
        }

        std::string_view text_;
        std::size_t position_ = 0;
    };

} // namespace

auto parse_json(std::string_view text) -> std::expected<JsonValue, JsonParseError> {
    return Parser{text}.parse_document();
}

auto json_escape(std::string_view text) -> std::string {
    std::string escaped;
    escaped.reserve(text.size());

    for (auto const character: text) {
        switch (character) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(character) < 0x20U) {
                    escaped += std::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(character)));
                } else {
                    escaped.push_back(character);
                }
        }
    }

    return escaped;
}

auto JsonWriter::prefix(std::string_view key) -> void {
    if (stack_.empty()) {
        return;
    }

    auto &level = stack_.back();
    if (!level.empty) {
        out_ += level.inline_container ? ", " : ",";
    }
    if (!level.inline_container) {
        out_ += '\n';
        out_.append(stack_.size() * 2, ' ');
    }
    level.empty = false;

    if (level.is_object) {
        out_ += '"';
        out_ += json_escape(key);
        out_ += "\": ";
    }
}

auto JsonWriter::close(char bracket) -> void {
    if (stack_.empty()) {
        return;
    }

    auto const level = stack_.back();
    stack_.pop_back();

    if (!level.inline_container && !level.empty) {
        out_ += '\n';
        out_.append(stack_.size() * 2, ' ');
    }
    out_ += bracket;

    if (stack_.empty()) {
        out_ += '\n';
    }
}

auto JsonWriter::begin_object(std::string_view key, bool inline_container) -> JsonWriter & {
    prefix(key);
    out_ += '{';
    // Containers inside an inline one stay inline.
    auto const keep_inline = inline_container || (!stack_.empty() && stack_.back().inline_container);
    stack_.push_back(Level{.is_object = true, .inline_container = keep_inline});
    return *this;
}

auto JsonWriter::end_object() -> JsonWriter & {
    close('}');
    return *this;
}

auto JsonWriter::begin_array(std::string_view key, bool inline_container) -> JsonWriter & {
    prefix(key);
    out_ += '[';
    auto const keep_inline = inline_container || (!stack_.empty() && stack_.back().inline_container);
    stack_.push_back(Level{.is_object = false, .inline_container = keep_inline});
    return *this;
}

auto JsonWriter::end_array() -> JsonWriter & {
    close(']');
    return *this;
}

auto JsonWriter::value(std::string_view key, std::string_view text) -> JsonWriter & {
    prefix(key);
    out_ += '"';
    out_ += json_escape(text);
    out_ += '"';
    return *this;
}

auto JsonWriter::value(std::string_view key, bool flag) -> JsonWriter & {
    prefix(key);
    out_ += flag ? "true" : "false";
    return *this;
}

auto JsonWriter::value(std::string_view key, std::int64_t number) -> JsonWriter & {
    prefix(key);
    out_ += std::format("{}", number);
    return *this;
}

auto JsonWriter::value(std::string_view key, std::uint64_t number) -> JsonWriter & {
    prefix(key);
    out_ += std::format("{}", number);
    return *this;
}

auto JsonWriter::value(std::string_view key, double number, int precision) -> JsonWriter & {
    prefix(key);
    if (!std::isfinite(number)) {
        out_ += "null";
    } else {
        out_ += std::format("{:.{}f}", number, precision);
    }
    return *this;
}

auto JsonWriter::null(std::string_view key) -> JsonWriter & {
    prefix(key);
    out_ += "null";
    return *this;
}

auto JsonWriter::numbers(std::string_view key, std::span<float const> values, int precision) -> JsonWriter & {
    begin_array(key, true);
    for (auto const number: values) {
        value({}, static_cast<double>(number), precision);
    }
    return end_array();
}
