-- Fixed camera shots of the Bistro exterior for GI / ray traced shadow comparisons (no player, no physics).
--   lathe --game=lua --script=assets/scripts/gi_test/bistro_shots.lua --player
-- Keys 1-4: camera shot. Edit MODEL to where BistroExterior.glb lives (see assets/scripts/bistro/main.lua).

local MODEL = "/home/edwin/Downloads/Bistro_v5_2/BistroExterior.glb"

---@type GameScript
local M = {}

local SHOTS = {
    { eye = { 0, 1.7, 0 }, look = { 6, 1.5, 4 }, fov = 70 },
    { eye = { -8, 1.7, 6 }, look = { 0, 1.5, 0 }, fov = 70 },
    { eye = { 10, 6, -10 }, look = { 0, 0, 0 }, fov = 60 },
    { eye = { 4, 1.7, -6 }, look = { -4, 2.5, 4 }, fov = 70 },
}
local shot = 1

function M.on_populate()
    scene.clear()
    local city = scene.spawn("bistro")
    city:set_model(assets.load_model(MODEL, { min_extent = 0.35, double_sided = true }))
end

function M.on_update(_) end

function M.on_key(code)
    if code >= key.NUM_1 and code <= key.NUM_4 then
        shot = code - key.NUM_1 + 1
    end
end

function M.camera(_)
    local s = SHOTS[shot]
    return { eye = s.eye, target = s.look, fov = s.fov, near = 0.05, far = 1500 }
end

return M
