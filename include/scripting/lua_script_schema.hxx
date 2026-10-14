#pragma once

#include <optional>
#include <string>

#include "scene/script_property.hxx"
#include "scripting/lua_api.hxx"

// Reads the `properties` field of the behaviour class table at `index`.
//
//   properties = {
//       { name = "speed", type = "number", default = 3, min = 0, max = 10 },   -- listed in this order
//       colour = { type = "colour", default = {1, 0, 0} },                       -- then these, by name
//   }
//
// A script with a problem gets the message in `error` and no properties; nothing is raised.
[[nodiscard]] auto read_script_schema(lua_State *state, int index, std::string name) -> ScriptSchema;

// The value at `index` as a property of `type`, or nullopt when it is not one. Entity and asset properties are not
// read here: assets come as names (strings) and entities need the host.
[[nodiscard]] auto read_script_value(lua_State *state, int index, ScriptPropertyType type) -> std::optional<ScriptValue>;

// Pushes `value` as plain Lua data: booleans, numbers, strings, `{x, y, z}` and `{r, g, b, a}` tables (named and
// positional). Entity values are not pushed (nil).
auto push_script_value(lua_State *state, ScriptValue const &value) -> void;
