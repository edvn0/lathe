---@meta

---A game script defines its callbacks as global functions; the engine calls them (LuaGame in src/luagame/lua_game.cxx).
---All are optional. A callback that raises an error is reported once and disabled until the script is reloaded.
---
---```lua
---function on_populate()
---    scene.spawn("cube")
---end
---
---function on_update(dt) end
---```
---
---A script may instead return a table holding the same functions (the older style; `camera` there is `on_camera` here,
---because `camera` is also the name of the engine's library).

---Once, after the script has loaded.
function on_populate() end

---When a scene becomes the active one (editor scene, then play scene).
function on_bind() end

---@param dt number seconds since the last frame
function on_update(dt) end

---@param key integer one of the key.* constants
---@param modifiers integer GLFW modifier bits
function on_key(key, modifiers) end

---@param button integer one of the mouse.* constants
function on_mouse_button(button) end

---Needs `wants_cursor = true`.
---@param ndc_x number
---@param ndc_y number
---@param inside number 1 when the cursor is over the game view, else 0
function on_cursor(ndc_x, ndc_y, inside) end

---Called while the UI is built; use the ui library.
function on_ui() end

---Where the game camera is. Called with the view's aspect ratio.
---@param aspect number
---@return CameraResult
function on_camera(aspect) end

---Set to true to receive on_cursor events.
---@type boolean?
wants_cursor = nil

---A point as {x, y, z} or {x = , y = , z = }.
---@alias Vec3 number[]|{x: number, y: number, z: number}

---@class CameraResult
---@field eye? Vec3 default {0, 6, -8}
---@field target? Vec3 default {0, 0, 0}
---@field up? Vec3 default {0, 1, 0}; use another axis for a camera that looks straight down or up
---@field ortho_height? number if above 0, an orthographic view this many world units tall (width follows the aspect) instead of a perspective one
---@field fov? number vertical field of view in degrees (default 60)
---@field near? number (default 0.1)
---@field far? number (default 1000)

---The table a script may return instead of defining global functions.
---@class GameScript
---@field wants_cursor? boolean true to receive on_cursor events
---@field on_populate? fun()
---@field on_bind? fun()
---@field on_update? fun(dt: number)
---@field on_key? fun(key: integer, modifiers: integer)
---@field on_mouse_button? fun(button: integer)
---@field on_cursor? fun(ndc_x: number, ndc_y: number, inside: number)
---@field on_ui? fun()
---@field camera? fun(aspect: number): CameraResult
