-- "Cornell" room with a skylight for probe GI checks (phase 3: sun and sky only, so the light comes in through an opening):
-- colour bleeding from the red wall and the lit floor, and leaks into the closed back third behind the divider.
--   lathe --game=lua --script=assets/scripts/gi_test/cornell_skylight.lua --player
-- Shots advance every 60 frames (see SHOT_FRAMES), so screenshots at frames 30, 90, 150 ... show shots 1, 2, 3 ...
-- Red left wall, green right wall, white elsewhere, a ceiling that is open over the left front half, and a thin divider
-- wall that splits the back third off: irradiance behind it must stay ~0 for the leak test.

local SHOTS = {
    { eye = { 1.8, 2.2, 0.4 }, look = { -2, 1.0, 4.5 }, fov = 75 },
    { eye = { 2.4, 1.6, 4.5 }, look = { -2, 0.8, 2.5 }, fov = 70 },
    { eye = { 0, 1.6, 7.6 }, look = { 0, 1.4, 5.6 }, fov = 80 },
}
local shot = 1
local count = 0

local function material(colour, emissive, strength)
    count = count + 1
    return assets.material("gi_test_" .. count, colour[1], colour[2], colour[3], 0.9, 0,
        { emissive = emissive, emissive_strength = strength or 1, double_sided = true })
end

local function box(name, cube, mat, centre, half)
    local e = scene.spawn(name)
    e:set_position(centre[1], centre[2], centre[3])
    e:set_scale(half[1], half[2], half[3])
    e:set_model(cube)
    e:set_material(mat)
    return e
end

function on_populate()
    scene.clear()
    scene.set_environment({ sky = "procedural", skybox = false, ambient = 0.05, exposure_ev = 0, sun_intensity = 3,
        sun_elevation = 50 })

    local cube = assets.cube()
    local white = material({ 0.8, 0.8, 0.8 })
    local red = material({ 0.75, 0.1, 0.1 })
    local green = material({ 0.1, 0.65, 0.15 })

    -- Room interior 6 wide (x), 5 tall (y), 8 deep (z), centred on x = 0, floor at y = 0.
    box("floor", cube, white, { 0, -0.1, 4 }, { 3.2, 0.1, 4.2 })
    box("ceiling_back", cube, white, { 0, 5.1, 6.8 }, { 3.2, 0.1, 1.4 })
    box("ceiling_right", cube, white, { 1.6, 5.1, 2.7 }, { 1.6, 0.1, 2.7 })
    box("back", cube, white, { 0, 2.5, 8.1 }, { 3.2, 2.6, 0.1 })
    box("front", cube, white, { 0, 2.5, -0.1 }, { 3.2, 2.6, 0.1 })
    box("left", cube, red, { -3.1, 2.5, 4 }, { 0.1, 2.6, 4.2 })
    box("right", cube, green, { 3.1, 2.5, 4 }, { 0.1, 2.6, 4.2 })
    box("divider", cube, white, { 0, 2.5, 5.5 }, { 3.0, 2.5, 0.03 })
    box("block", cube, white, { -1.2, 0.75, 3 }, { 0.75, 0.75, 0.75 })
end

-- Key injection does not reach Lua games, so a headless run steps through the shots by frame count: shot k is on screen
-- from frame (k - 1) * SHOT_FRAMES, i.e. screenshot at frames 30, 90, 150 ... for shots 1, 2, 3 ...
local SHOT_FRAMES = 60
local frames = 0

function on_update(_)
    frames = frames + 1
    shot = math.min(#SHOTS, math.floor(frames / SHOT_FRAMES) + 1)
end

function on_camera(_)
    local s = SHOTS[shot]
    return { eye = s.eye, target = s.look, fov = s.fov, near = 0.05, far = 100 }
end
