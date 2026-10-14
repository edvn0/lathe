#pragma once

#include <entt/entity/fwd.hpp>

#include "scene/script_property.hxx"
#include "scripting/lua_api.hxx"

class LuaRuntime;

auto open_lua_game_api(LuaRuntime &runtime) -> void;

struct LuaGameHost;

// Per-frame upkeep of what scripts set up through the physics library: adds mesh colliders whose triangles are ready
// to the active scene's physics world.
auto update_lua_game_physics(LuaGameHost &host) -> void;

// Pushes the Entity handle scripts hold for `entity`.
auto push_lua_entity(lua_State *state, entt::entity entity) -> void;

// Pushes `value` the way a behaviour sees property `desc`: plain Lua data, an Entity, or a Model, Material or
// Texture looked up by its registered name (nil when there is none).
auto push_script_property(lua_State *state, LuaGameHost &host, ScriptPropertyDesc const &desc, ScriptValue const &value)
        -> void;
