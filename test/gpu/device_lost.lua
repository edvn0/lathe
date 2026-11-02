-- Driven by device_lost_recovery.sh: builds a scene with a textured model on a cold texture cache, with the GPU encoder
-- armed to fault, so the device is lost while the scene exists nowhere but in memory.

function on_populate()
    game.debug_lose_device()
    scene.spawn("device_lost_marker")
    assets.load_model("assets/models/damaged_helmet.glb")
end

function on_camera(aspect)
    return { eye = { 0, 1, 4 }, look_at = { 0, 0, 0 }, fov = 45 }
end
