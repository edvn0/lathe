#pragma once

#include <optional>
#include <string>

#include "scene/components.hxx"
#include "scripting/lua_api.hxx"

// Applies the fields of the Lua table at stack index `table` to `emitter`; fields that are left out keep their value.
// Every field is checked, so a script cannot hand the renderer a value it cannot draw: an unknown name, a value of the
// wrong type, a non-finite number or one out of range gives the problem as text and leaves `emitter` untouched. The
// stack is as it was on return. `material` is accepted but not read: the caller owns the userdata type.
//
//   count (1 .. ParticleEmitter::max_count)   rate, lifetime, speed, speed_variance, shape_size, cone_degrees,
//   size_start, size_end                      shape ("point", "sphere", "box" or "cone")
//   gravity = {x, y, z}                       colour_start, colour_end = {r, g, b [, a]}
//   emitting (boolean)                        material
[[nodiscard]] auto read_particle_table(lua_State *state, int table, Components::ParticleEmitter &emitter)
        -> std::optional<std::string>;
