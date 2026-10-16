#pragma once

#include <optional>
#include <string>

#include "scene/components.hxx"
#include "scripting/lua_api.hxx"

// The most text one Text component holds, as the scene format allows.
inline constexpr std::size_t max_lua_text_bytes = 64 * 1024;

// Applies the fields of the Lua table at stack index `table` to `text`; fields that are left out keep their value.
// Every field is checked, so a script cannot hand the renderer a value it cannot draw: an unknown name, a value of the
// wrong type, a non-finite number or one out of range gives the problem as text and leaves `text` untouched. The stack
// is as it was on return.
//
//   text (string, up to 64 KiB)      font (integer id)         size (above 0, at most 10000)
//   color or colour = {r, g, b [, a]} (components 0 .. 1000, alpha 0 .. 1)
//   emissive (0 .. 1000)             centred or centered, visible (boolean)
[[nodiscard]] auto read_text_table(lua_State *state, int table, Components::Text &text) -> std::optional<std::string>;
