-- Intel's Sponza (Main) from a glb on disk, in a bright procedural sky.
-- Started with --game=lua --script=assets/scripts/sponza/main.lua [--player]; saved as assets/scenes/sponza.lbf.

---@type GameScript
local M = {}

local MODEL = "/home/edwin/Assets/Meshes/main_sponza.glb"

function M.on_populate()
    scene.set_environment({ sky = "procedural", skybox = true, ambient = 0.05, exposure_ev = 0, sun_intensity = 3,
                            sun_elevation = 55 })

    local sponza = scene.spawn("sponza")
    sponza:set_model(assets.load_model(MODEL, { collider = true }))
end

-- The relief (node sponza_ornament) is a thin slab centred at (-7.9, 13.3, 0) facing along x. Keys 1 and 2 look at it
-- from either side, 3 is a wide shot of the courtyard, 4 a close-up of the relief.
local SHOTS = {
    { eye = { -2.4, 13.3, 0 }, target = { -7.9, 13.3, 0 } },
    { eye = { -13.4, 13.3, 0 }, target = { -7.9, 13.3, 0 } },
    { eye = { 12, 8, -2 }, target = { -7.9, 11, 0 } },
    { eye = { -6.7, 13.9, 0.6 }, target = { -7.9, 13.6, 0.4 } },
}
local shot = 1

-- Starts on shot n; used by the wrapper scripts in this folder for scripted runs.
function M.set_shot(n)
    shot = n
end

function M.on_key(key)
    if key >= 49 and key <= 52 then
        shot = key - 48
    end
end

function M.camera(_aspect)
    local s = SHOTS[shot]
    return { eye = s.eye, target = s.target, fov = 60, near = 0.05, far = 500 }
end

return M
