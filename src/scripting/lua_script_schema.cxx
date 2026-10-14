#include "scripting/lua_script_schema.hxx"

#include <algorithm>
#include <cmath>
#include <expected>
#include <format>
#include <limits>
#include <utility>
#include <vector>

namespace {
    constexpr std::string_view reserved_property = "entity";

    auto number_at(lua_State *state, int table, char const *name, lua_Integer position) -> std::optional<double> {
        lua_getfield(state, table, name);

        if (lua_isnil(state, -1)) {
            lua_pop(state, 1);
            lua_geti(state, table, position);
        }

        std::optional<double> result;

        if (lua_type(state, -1) == LUA_TNUMBER && std::isfinite(lua_tonumber(state, -1))) {
            result = lua_tonumber(state, -1);
        }

        lua_pop(state, 1);

        return result;
    }

    auto valid_identifier(std::string_view name) -> bool {
        if (name.empty() || (name.front() >= '0' && name.front() <= '9')) {
            return false;
        }

        return std::ranges::all_of(name, [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        });
    }

    auto optional_number(lua_State *state, int table, char const *key) -> std::optional<double> {
        lua_getfield(state, table, key);

        std::optional<double> result;

        if (lua_type(state, -1) == LUA_TNUMBER) {
            result = lua_tonumber(state, -1);
        }

        lua_pop(state, 1);

        return result;
    }

    auto optional_string(lua_State *state, int table, char const *key) -> std::string {
        lua_getfield(state, table, key);

        std::string result = lua_type(state, -1) == LUA_TSTRING ? lua_tostring(state, -1) : "";

        lua_pop(state, 1);

        return result;
    }

    // Reads one property's spec table at `table`. `name` is the key it was listed under, or empty for the array form.
    auto read_property(lua_State *state, int table, std::string name) -> std::expected<ScriptPropertyDesc, std::string> {
        if (name.empty()) {
            name = optional_string(state, table, "name");
        }

        if (!valid_identifier(name)) {
            return std::unexpected{std::format("'{}' is not a usable property name (letters, digits, underscore)", name)};
        }

        if (name == reserved_property) {
            return std::unexpected{"'entity' is reserved for the entity the script is on"};
        }

        ScriptPropertyDesc desc;
        desc.name = name;

        auto const type_text = optional_string(state, table, "type");

        if (!script_type_from_name(type_text, desc.type)) {
            return std::unexpected{std::format("property '{}': unknown type '{}'", name, type_text)};
        }

        desc.tooltip = optional_string(state, table, "tooltip");
        desc.default_value = script_type_zero(desc.type);

        if (desc.type == ScriptPropertyType::choice) {
            lua_getfield(state, table, "options");

            if (!lua_istable(state, -1)) {
                lua_pop(state, 1);

                return std::unexpected{std::format("property '{}': a choice needs an `options` list", name)};
            }

            for (lua_Integer position = 1;; ++position) {
                lua_geti(state, -1, position);

                if (lua_isnil(state, -1)) {
                    lua_pop(state, 1);

                    break;
                }

                if (lua_type(state, -1) != LUA_TSTRING) {
                    lua_pop(state, 2);

                    return std::unexpected{std::format("property '{}': options must be strings", name)};
                }

                desc.choices.emplace_back(lua_tostring(state, -1));
                lua_pop(state, 1);
            }

            lua_pop(state, 1);

            if (desc.choices.empty()) {
                return std::unexpected{std::format("property '{}': a choice needs at least one option", name)};
            }

            desc.default_value = desc.choices.front();
        }

        auto const minimum = optional_number(state, table, "min");
        auto const maximum = optional_number(state, table, "max");

        if (minimum || maximum) {
            if (desc.type != ScriptPropertyType::integer && desc.type != ScriptPropertyType::number &&
                desc.type != ScriptPropertyType::vec3) {
                return std::unexpected{std::format("property '{}': only numbers and vec3 take min/max", name)};
            }

            desc.has_range = true;
            desc.min = minimum.value_or(std::numeric_limits<double>::lowest());
            desc.max = maximum.value_or(std::numeric_limits<double>::max());

            if (!(desc.min <= desc.max)) {
                return std::unexpected{std::format("property '{}': min is greater than max", name)};
            }
        }

        desc.step = optional_number(state, table, "step").value_or(0.0);

        lua_getfield(state, table, "default");

        if (!lua_isnil(state, -1)) {
            if (desc.type == ScriptPropertyType::entity) {
                lua_pop(state, 1);

                return std::unexpected{std::format("property '{}': an entity property has no default", name)};
            }

            auto value = read_script_value(state, lua_gettop(state), desc.type);

            lua_pop(state, 1);

            if (!value) {
                return std::unexpected{std::format("property '{}': the default is not a {}", name, script_type_name(desc.type))};
            }

            if (desc.type == ScriptPropertyType::choice &&
                std::ranges::find(desc.choices, std::get<std::string>(*value)) == desc.choices.end()) {
                return std::unexpected{std::format("property '{}': the default is not one of the options", name)};
            }

            desc.default_value = script_value_sanitised(desc, std::move(*value));
        } else {
            lua_pop(state, 1);
        }

        return desc;
    }
}

auto read_script_value(lua_State *state, int index, ScriptPropertyType type) -> std::optional<ScriptValue> {
    index = lua_absindex(state, index);

    switch (type) {
        case ScriptPropertyType::boolean:
            if (lua_isboolean(state, index)) {
                return lua_toboolean(state, index) != 0;
            }

            return std::nullopt;
        case ScriptPropertyType::integer: {
            if (lua_type(state, index) != LUA_TNUMBER) {
                return std::nullopt;
            }

            auto const number = lua_tonumber(state, index);

            if (!std::isfinite(number) || std::floor(number) != number || std::fabs(number) > 9.0e15) {
                return std::nullopt;
            }

            return static_cast<std::int64_t>(number);
        }
        case ScriptPropertyType::number: {
            if (lua_type(state, index) != LUA_TNUMBER || !std::isfinite(lua_tonumber(state, index))) {
                return std::nullopt;
            }

            return lua_tonumber(state, index);
        }
        case ScriptPropertyType::string:
        case ScriptPropertyType::choice:
        case ScriptPropertyType::model:
        case ScriptPropertyType::material:
        case ScriptPropertyType::texture:
            if (lua_type(state, index) == LUA_TSTRING) {
                return std::string{lua_tostring(state, index)};
            }

            return std::nullopt;
        case ScriptPropertyType::vec3: {
            if (!lua_istable(state, index)) {
                return std::nullopt;
            }

            auto const x = number_at(state, index, "x", 1);
            auto const y = number_at(state, index, "y", 2);
            auto const z = number_at(state, index, "z", 3);

            if (!x || !y || !z) {
                return std::nullopt;
            }

            return glm::vec3{static_cast<float>(*x), static_cast<float>(*y), static_cast<float>(*z)};
        }
        case ScriptPropertyType::colour: {
            if (!lua_istable(state, index)) {
                return std::nullopt;
            }

            auto const red = number_at(state, index, "r", 1);
            auto const green = number_at(state, index, "g", 2);
            auto const blue = number_at(state, index, "b", 3);
            auto const alpha = number_at(state, index, "a", 4);

            if (!red || !green || !blue) {
                return std::nullopt;
            }

            return glm::vec4{static_cast<float>(*red), static_cast<float>(*green), static_cast<float>(*blue),
                             static_cast<float>(alpha.value_or(1.0))};
        }
        case ScriptPropertyType::entity:
            return std::nullopt;
    }

    return std::nullopt;
}

auto push_script_value(lua_State *state, ScriptValue const &value) -> void {
    struct Pusher {
        lua_State *state;

        void operator()(bool flag) const { lua_pushboolean(state, flag ? 1 : 0); }
        void operator()(std::int64_t integer) const { lua_pushinteger(state, integer); }
        void operator()(double number) const { lua_pushnumber(state, number); }
        void operator()(std::string const &text) const { lua_pushlstring(state, text.data(), text.size()); }
        void operator()(glm::vec3 const &vector) const {
            lua_createtable(state, 3, 3);

            constexpr char const *names[] = {"x", "y", "z"};

            for (int axis = 0; axis < 3; ++axis) {
                lua_pushnumber(state, static_cast<double>(vector[axis]));
                lua_pushvalue(state, -1);
                lua_seti(state, -3, axis + 1);
                lua_setfield(state, -2, names[axis]);
            }
        }
        void operator()(glm::vec4 const &colour) const {
            lua_createtable(state, 4, 4);

            constexpr char const *names[] = {"r", "g", "b", "a"};

            for (int channel = 0; channel < 4; ++channel) {
                lua_pushnumber(state, static_cast<double>(colour[channel]));
                lua_pushvalue(state, -1);
                lua_seti(state, -3, channel + 1);
                lua_setfield(state, -2, names[channel]);
            }
        }
        void operator()(entt::entity) const { lua_pushnil(state); }
    };

    std::visit(Pusher{state}, value);
}

auto read_script_schema(lua_State *state, int index, std::string name) -> ScriptSchema {
    index = lua_absindex(state, index);

    ScriptSchema schema;
    schema.name = std::move(name);

    auto const fail = [&](std::string message) {
        schema.properties.clear();
        schema.error = std::move(message);

        return schema;
    };

    if (!lua_checkstack(state, 12)) {
        return fail("out of Lua stack");
    }

    lua_getfield(state, index, "properties");

    if (lua_isnil(state, -1)) {
        lua_pop(state, 1);

        return schema;
    }

    if (!lua_istable(state, -1)) {
        lua_pop(state, 1);

        return fail("`properties` must be a table");
    }

    auto const properties = lua_gettop(state);


    std::vector<std::string> named;

    // The array form keeps its order.
    for (lua_Integer position = 1;; ++position) {
        lua_geti(state, properties, position);

        if (lua_isnil(state, -1)) {
            lua_pop(state, 1);

            break;
        }

        if (!lua_istable(state, -1)) {
            lua_pop(state, 2);

            return fail(std::format("properties[{}] must be a table", position));
        }

        auto desc = read_property(state, lua_gettop(state), {});

        lua_pop(state, 1);

        if (!desc) {
            lua_pop(state, 1);

            return fail(std::move(desc.error()));
        }

        if (schema.find(desc->name) != nullptr) {
            lua_pop(state, 1);

            return fail(std::format("property '{}' is listed twice", desc->name));
        }

        schema.properties.push_back(std::move(*desc));
    }

    // The keyed form follows, by name.
    lua_pushnil(state);

    while (lua_next(state, properties) != 0) {
        if (lua_type(state, -2) == LUA_TSTRING) {
            named.emplace_back(lua_tostring(state, -2));
        }

        lua_pop(state, 1);
    }

    std::ranges::sort(named);

    for (auto const &key: named) {
        lua_getfield(state, properties, key.c_str());

        if (!lua_istable(state, -1)) {
            lua_pop(state, 2);

            return fail(std::format("property '{}' must be a table", key));
        }

        auto desc = read_property(state, lua_gettop(state), key);

        lua_pop(state, 1);

        if (!desc) {
            lua_pop(state, 1);

            return fail(std::move(desc.error()));
        }

        if (schema.find(desc->name) != nullptr) {
            lua_pop(state, 1);

            return fail(std::format("property '{}' is listed twice", desc->name));
        }

        schema.properties.push_back(std::move(*desc));
    }

    lua_pop(state, 1);

    return schema;
}
