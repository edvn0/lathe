#include "scripting/lua_text.hxx"

#include <cmath>
#include <expected>
#include <format>
#include <string_view>

namespace {
    using Problem = std::string;

    auto number_field(lua_State *state, int value, std::string_view name) -> std::expected<double, Problem> {
        if (lua_isnumber(state, value) == 0 || lua_type(state, value) == LUA_TSTRING) {
            return std::unexpected(std::format("'{}' must be a number", name));
        }
        auto const result = lua_tonumber(state, value);
        if (!std::isfinite(result)) {
            return std::unexpected(std::format("'{}' must be finite", name));
        }
        return result;
    }

    auto read_colour(lua_State *state, int value, std::string_view name, glm::vec4 &colour) -> std::optional<Problem> {
        auto const shape = std::format("'{}' must be {{r, g, b [, a]}}, components 0 .. 1000 and alpha 0 .. 1", name);

        if (!lua_istable(state, value)) {
            return shape;
        }
        auto const length = lua_rawlen(state, value);
        if (length != 3 && length != 4) {
            return shape;
        }

        auto result = glm::vec4{0.0F, 0.0F, 0.0F, 1.0F};
        for (auto channel = std::size_t{0}; channel < length; ++channel) {
            lua_rawgeti(state, value, static_cast<lua_Integer>(channel + 1));
            auto const ok = lua_type(state, -1) == LUA_TNUMBER;
            auto const component = ok ? lua_tonumber(state, -1) : -1.0;
            lua_pop(state, 1);

            if (!ok || !std::isfinite(component) || component < 0.0 || component > 1.0e3 ||
                (channel == 3 && component > 1.0)) {
                return shape;
            }
            result[static_cast<glm::length_t>(channel)] = static_cast<float>(component);
        }

        colour = result;
        return std::nullopt;
    }

    auto apply_field(lua_State *state, std::string_view key, int value, Components::Text &text)
            -> std::optional<Problem> {
        auto const boolean = [&]() -> std::expected<bool, Problem> {
            if (!lua_isboolean(state, value)) {
                return std::unexpected(std::format("'{}' must be true or false", key));
            }
            return lua_toboolean(state, value) != 0;
        };

        if (key == "text") {
            if (lua_type(state, value) != LUA_TSTRING) {
                return "'text' must be a string";
            }
            auto length = std::size_t{0};
            auto const *const chars = lua_tolstring(state, value, &length);
            if (length > max_lua_text_bytes) {
                return std::format("'text' is longer than {} bytes", max_lua_text_bytes);
            }
            text.text.assign(chars, length);
            return std::nullopt;
        }
        if (key == "font") {
            auto const number = number_field(state, value, key);
            if (!number) {
                return number.error();
            }
            if (*number < 0.0 || *number > 1.0e6 || *number != std::floor(*number)) {
                return "'font' must be a font id";
            }
            text.font = static_cast<std::uint32_t>(*number);
            return std::nullopt;
        }
        if (key == "size") {
            auto const number = number_field(state, value, key);
            if (!number) {
                return number.error();
            }
            if (*number <= 0.0 || *number > 1.0e4) {
                return "'size' must be above 0 and at most 10000";
            }
            text.size = static_cast<float>(*number);
            return std::nullopt;
        }
        if (key == "emissive") {
            auto const number = number_field(state, value, key);
            if (!number) {
                return number.error();
            }
            if (*number < 0.0 || *number > 1.0e3) {
                return "'emissive' must be 0 .. 1000";
            }
            text.emissive = static_cast<float>(*number);
            return std::nullopt;
        }
        if (key == "color" || key == "colour") {
            return read_colour(state, value, key, text.colour);
        }
        if (key == "centred" || key == "centered") {
            auto const flag = boolean();
            if (!flag) {
                return flag.error();
            }
            text.centred = *flag;
            return std::nullopt;
        }
        if (key == "visible") {
            auto const flag = boolean();
            if (!flag) {
                return flag.error();
            }
            text.visible = *flag;
            return std::nullopt;
        }

        return std::format("unknown text field '{}'", key);
    }
} // namespace

auto read_text_table(lua_State *state, int table, Components::Text &text) -> std::optional<std::string> {
    table = lua_absindex(state, table);

    if (!lua_istable(state, table)) {
        return "expected a table of text fields";
    }

    auto const top = lua_gettop(state);
    auto updated = text;
    auto problem = std::optional<Problem>{};

    lua_pushnil(state);

    while (lua_next(state, table) != 0) {
        if (lua_type(state, -2) != LUA_TSTRING) {
            problem = "text field names must be strings";
            break;
        }

        problem = apply_field(state, lua_tostring(state, -2), lua_gettop(state), updated);

        if (problem) {
            break;
        }

        lua_pop(state, 1);
    }

    lua_settop(state, top);

    if (problem) {
        return problem;
    }

    text = std::move(updated);
    return std::nullopt;
}
