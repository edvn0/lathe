-- Fixed camera shots of the Bistro exterior for GI / ray traced shadow comparisons (no player, no physics).
--   lathe --game=lua --script=assets/scripts/gi_test/bistro_shots.lua --player
-- Shots advance every SHOT_FRAMES (200) frames; screenshot at frames 190, 390, 590, 790 for shots 1 to 4.

local MODEL = "/home/edwin/Downloads/Bistro_v5_2/BistroExterior.glb"

local SHOTS = {
    { eye = { -2.83, 2.4, 0 }, look = { -2.83, 1.2, 12 }, fov = 75 },
    { eye = { -2.83, 45, -35 }, look = { -2.83, 0, 5 }, fov = 60 },
    { eye = { -2.83, 2.4, 5 }, look = { -2.83, 1.2, -8 }, fov = 75 },
    -- The first shot again, for runs where the model has finished streaming by now (screenshot at frame 790).
    { eye = { -2.83, 2.4, 0 }, look = { -2.83, 1.2, 12 }, fov = 75 },
}
local shot = 1

function on_populate()
    scene.clear()
    local city = scene.spawn("bistro")
    city:set_model(assets.load_model(MODEL, { min_extent = 0.35, double_sided = true }))
end

-- Key injection does not reach Lua games, so a headless run steps through the shots by frame count: shot k is on screen
-- from frame (k - 1) * SHOT_FRAMES, i.e. screenshot at frames 190, 390, 590 for shots 1, 2, 3 (the Bistro takes a while to stream in)
local SHOT_FRAMES = 200
local frames = 0

function on_update(_)
    frames = frames + 1
    shot = math.min(#SHOTS, math.floor(frames / SHOT_FRAMES) + 1)
end

function on_camera(_)
    local s = SHOTS[shot]
    return { eye = s.eye, target = s.look, fov = s.fov, near = 0.05, far = 1500 }
end
