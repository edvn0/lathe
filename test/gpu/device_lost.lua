-- Driven by device_lost_recovery.sh: builds a scene with a textured model on a cold texture cache, with the GPU encoder
-- armed to fault, so the device is lost while the scene exists nowhere but in memory.

---@type GameScript
local M = {}

function M.on_populate()
    game.debug_lose_device()
    scene.spawn("device_lost_marker")
    assets.load_model("assets/models/damaged_helmet.glb")
end

function M.camera(aspect)
    return { eye = { 0, 1, 4 }, look_at = { 0, 0, 0 }, fov = 45 }
end

return M
