#pragma once

class LuaRuntime;

// Installs the game API into the runtime: the scene, assets, camera, ui and game tables, the Entity and Model
// types, and the key and mouse constants. The runtime's host must be a LuaGameHost.
auto open_lua_game_api(LuaRuntime &runtime) -> void;
