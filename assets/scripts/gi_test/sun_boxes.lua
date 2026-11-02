-- Sun-lit boxes on a floor: a small scene for the ray traced sun shadows (a project with [render] ray_traced_shadows =
-- false shows the cascaded shadow map for comparison).
--   lathe --project samples/gi_sun_boxes --player
-- Shots advance every 60 frames (see SHOT_FRAMES), so screenshots at frames 30, 90, 150 ... show shots 1, 2, 3 ...

local SHOTS = {
    { eye = { -9, 5, -9 }, look = { 0, 1, 0 }, fov = 55 },
    { eye = { 0, 14, -3 }, look = { 0, 0, 1 }, fov = 55 },
}
local shot = 1
local count = 0

local function material(colour)
    count = count + 1
    return assets.material("gi_sun_" .. count, colour[1], colour[2], colour[3], 0.9, 0, { double_sided = true })
end

local function box(name, mesh, mat, centre, half)
    local e = scene.spawn(name)
    e:set_position(centre[1], centre[2], centre[3])
    e:set_scale(half[1], half[2], half[3])
    e:set_model(mesh)
    e:set_material(mat)
    return e
end

function on_populate()
    scene.clear()
    scene.set_environment({ sky = "procedural", skybox = true, ambient = 0.05, exposure_ev = 0, sun_intensity = 3,
                            sun_elevation = 35 })
    local cube, sphere = assets.cube(), assets.sphere()
    local grey = material({ 0.7, 0.7, 0.7 })
    box("floor", cube, grey, { 0, -0.25, 0 }, { 14, 0.25, 14 })
    box("tall", cube, material({ 0.8, 0.3, 0.2 }), { -2, 2, 1 }, { 0.8, 2, 0.8 })
    box("wide", cube, material({ 0.2, 0.5, 0.8 }), { 2.5, 0.75, -1 }, { 1.5, 0.75, 1 })
    box("ball", sphere, material({ 0.8, 0.8, 0.2 }), { 0, 1.2, -3.5 }, { 1.2, 1.2, 1.2 })
    box("thin", cube, material({ 0.3, 0.7, 0.3 }), { 4, 1.5, 3 }, { 0.05, 1.5, 1.5 })
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
    return { eye = s.eye, target = s.look, fov = s.fov, near = 0.05, far = 200 }
end
