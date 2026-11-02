-- Dynamic scene for probe GI checks (phase 5): a box sliding in front of a red wall, an orbiting point light and a sun
-- that sets over 3 seconds, so the probes have to follow moving geometry, lights and the sun.
--   lathe --project samples/gi_dynamic --player
-- Frames: 0..60 sun high, 60..240 sun sinks from 60 to 8 degrees, the box slides all the time and the light orbits.
-- Screenshots at 50 (before), 130 (mid sunset), 300 (after).

local count = 0
local function material(colour)
    count = count + 1
    return assets.material("gi_dyn_" .. count, colour[1], colour[2], colour[3], 0.9, 0, { double_sided = true })
end

local function box(name, mesh, mat, centre, half)
    local e = scene.spawn(name)
    e:set_position(centre[1], centre[2], centre[3])
    e:set_scale(half[1], half[2], half[3])
    e:set_model(mesh)
    e:set_material(mat)
    return e
end

local mover, lamp, marker

function on_populate()
    scene.clear()
    scene.set_environment({ sky = "procedural", skybox = true, ambient = 0.02, exposure_ev = 0, sun_intensity = 3,
                            sun_elevation = 60 })
    local cube, sphere = assets.cube(), assets.sphere()
    box("floor", cube, material({ 0.75, 0.75, 0.75 }), { 0, -0.25, 0 }, { 10, 0.25, 8 })
    box("red wall", cube, material({ 0.8, 0.12, 0.1 }), { -6, 2.5, 0 }, { 0.2, 2.5, 6 })
    box("white wall", cube, material({ 0.8, 0.8, 0.8 }), { 0, 2.5, 6 }, { 8, 2.5, 0.2 })
    box("green wall", cube, material({ 0.12, 0.7, 0.2 }), { 6, 2.5, 0 }, { 0.2, 2.5, 6 })
    mover = box("mover", cube, material({ 0.2, 0.4, 0.9 }), { -3, 1, 0 }, { 1, 1, 1 })
    lamp = scene.spawn("orbit light")
    lamp:set_position(0, 2, 0)
    lamp:add_light({ type = "point", colour = { 1, 0.7, 0.4 }, intensity = 40, range = 12 })
    marker = box("lamp marker", sphere, material({ 1, 0.8, 0.5 }), { 0, 2, 0 }, { 0.1, 0.1, 0.1 })
end

local frames = 0

function on_update(_)
    frames = frames + 1
    local t = frames / 60
    mover:set_position(-3 + 5 * (0.5 + 0.5 * math.sin(t * 0.8)), 1, 0)
    lamp:set_position(3 * math.cos(t * 1.2), 2, 3 * math.sin(t * 1.2))
    marker:set_position(3 * math.cos(t * 1.2), 2, 3 * math.sin(t * 1.2))
    if frames > 60 and frames < 240 then
        scene.set_environment({ sun_elevation = 60 - 52 * (frames - 60) / 180 })
    end
end

function on_camera(_)
    return { eye = { 0, 5, -9 }, target = { 0, 1.5, 1 }, fov = 60, near = 0.05, far = 100 }
end
