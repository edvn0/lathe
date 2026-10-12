---@meta

---@class game
---@field player_mode boolean true when running as an installed game rather than in the editor
game = {}
game.player_mode = false

---Asks the application to exit.
function game.quit() end

---Seconds since start (ImGui's clock).
---@return number
function game.time() end
