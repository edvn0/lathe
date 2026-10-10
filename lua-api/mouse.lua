---@meta

---GLFW mouse buttons for on_mouse_button.
---@class mouse
mouse = {}
---@type integer
mouse.LEFT = 0
---@type integer
mouse.RIGHT = 1
---@type integer
mouse.MIDDLE = 2
---Mouse movement in pixels since the previous update (zero unless the mouse is captured by the game).
---@return number dx
---@return number dy
function mouse.delta() end
