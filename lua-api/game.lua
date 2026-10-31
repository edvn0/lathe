---@meta

---@class game
---@field player_mode boolean true when running as an installed game rather than in the editor
game = {}
game.player_mode = false

---Asks the application to exit.
function game.quit() end

---For testing the device-lost path: the next batch the GPU texture encoder compresses writes through a bad address, which
---loses the Vulkan device. Does nothing when textures are compressed on the CPU. Never call it in a real game.
function game.debug_lose_device() end

---Seconds since start (ImGui's clock).
---@return number
function game.time() end
