---@meta

---What a game script returns: a table of callbacks the engine calls (LuaGame in src/app/lua_game.cxx). All are
---optional. A callback that raises an error is reported once and disabled until the script is reloaded.
---@class LatheGame
---@field wants_cursor? boolean true to receive on_cursor events
local LatheGame = {}

---Once, after the script has loaded.
function LatheGame.on_populate() end
---When a scene becomes the active one (editor scene, then play scene).
function LatheGame.on_bind() end
---@param dt number seconds since the last frame
function LatheGame.on_update(dt) end
---@param key integer one of the key.* constants
---@param modifiers integer GLFW modifier bits
function LatheGame.on_key(key, modifiers) end
---@param button integer one of the mouse.* constants
function LatheGame.on_mouse_button(button) end
---Needs wants_cursor = true.
---@param ndc_x number
---@param ndc_y number
---@param inside number 1 when the cursor is over the game view, else 0
function LatheGame.on_cursor(ndc_x, ndc_y, inside) end
---Called while the UI is built; use the ui library.
function LatheGame.on_ui() end
---@param aspect number
---@return CameraResult
function LatheGame.camera(aspect) end

---A point as {x, y, z} or {x = , y = , z = }.
---@alias Vec3 number[]|{x: number, y: number, z: number}

---@class CameraResult
---@field eye? Vec3 default {0, 6, -8}
---@field target? Vec3 default {0, 0, 0}
---@field fov? number vertical field of view in degrees (default 60)
---@field near? number (default 0.1)
---@field far? number (default 1000)
