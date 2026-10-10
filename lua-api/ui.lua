---@meta

---Immediate-mode UI (Dear ImGui). Call from on_ui.
---@class ui
ui = {}

---@class UiWindowOptions
---@field fullscreen? boolean cover the viewport
---@field centred? boolean centre the window
---@field x? number offset from the top-left of the work area (default 16)
---@field y? number (default 16)
---@field pivot_x? number (default 0)
---@field pivot_y? number (default 0)
---@field no_inputs? boolean
---@field bg_alpha? number background opacity (default 0.85)

---Runs `body` inside a window; errors in it propagate.
---@param id string
---@param options UiWindowOptions
---@param body fun()
function ui.window(id, options, body) end
---@param text string
---@param scale? number default 1
function ui.text(text, scale) end
---@param text string
function ui.text_disabled(text) end
---@param text string
---@param scale? number default 1
function ui.text_centred(text, scale) end
---A centred button; true on the frame it is pressed.
---@param label string
---@param width? number default 300
---@param height? number default 52
---@param scale? number default 1.6
---@return boolean
function ui.button(label, width, height, scale) end
---@param label string
---@return boolean
function ui.small_button(label) end
---Returns the edited text each frame (at most 255 bytes).
---@param label string
---@param text string
---@param width? number default 300
---@return string
function ui.input_text(label, text, width) end
---@param fraction number 0 .. 1
---@param width? number default 300
---@param height? number default 6
function ui.progress(fraction, width, height) end
---@param radius? number default 28
---@param thickness? number default 5
function ui.spinner(radius, thickness) end
---@param width number
---@param height number
function ui.dummy(width, height) end
function ui.same_line() end
function ui.separator() end
---@return number width
---@return number height
function ui.display_size() end
