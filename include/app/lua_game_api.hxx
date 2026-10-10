#pragma once

class LuaRuntime;

auto open_lua_game_api(LuaRuntime &runtime) -> void;

struct LuaGameHost;

// Per-frame upkeep of what scripts set up through the physics library: adds mesh colliders whose triangles are ready
// to the active scene's physics world.
auto update_lua_game_physics(LuaGameHost &host) -> void;
