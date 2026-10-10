-- Particles from Lua. The engine simulates and draws them; a script only describes an emitter on an entity.
--   lathe --game lua --script assets/scripts/particles/main.lua

---@type LatheGame
local M = {}

local fountain, embers, orange, teal
local clock = 0.0

function M.on_populate()
    orange = assets.material("fountain_tint", 1.0, 0.55, 0.2)
    teal = assets.material("ember_tint", 0.2, 0.9, 0.8)

    -- A fountain: particles leave the entity's +Y axis within a 18 degree cone and fall back under gravity.
    fountain = scene.spawn("fountain")
    fountain:set_position(0.0, 0.0, 0.0)
    entity.add_particles(fountain, {
        count = 6000,
        rate = 2400,
        lifetime = 2.4,
        gravity = { 0.0, -9.81, 0.0 },
        shape = "cone",
        cone_degrees = 18,
        speed = 7.5,
        speed_variance = 0.2,
        size_start = 0.07,
        size_end = 0.02,
        colour_start = { 1.0, 0.9, 0.6, 1.0 },
        colour_end = { 1.0, 0.3, 0.1, 0.0 },
        material = orange,
    })

    -- Embers rising from a box, against gravity.
    embers = scene.spawn("embers")
    embers:set_position(4.0, 0.0, 0.0)
    entity.add_particles(embers, {
        count = 1500,
        rate = 300,
        lifetime = 3.0,
        gravity = { 0.0, 1.2, 0.0 },
        shape = "box",
        shape_size = 0.8,
        cone_degrees = 10,
        speed = 0.6,
        size_start = 0.05,
        size_end = 0.0,
        colour_start = { 0.6, 1.0, 1.0, 1.0 },
        colour_end = { 0.2, 0.6, 0.8, 0.0 },
        material = teal,
    })

    -- A bad value is a Lua error a script can catch; the emitter is left as it was.
    local ok, problem = pcall(function() entity.set_particles(fountain, { count = -5 }) end)
    print("rejected as expected: " .. tostring(problem), ok)
end

function M.on_update(dt)
    clock = clock + dt

    -- Changing fields keeps the particles that are alive: entity.set_particles only touches what it is given.
    entity.set_particles(fountain, { speed = 7.5 + 2.0 * math.sin(clock) })
end

function M.camera(aspect)
    return { eye = { 0.0, 3.5, -11.0 }, target = { 1.5, 3.0, 0.0 }, fov = 55 }
end

return M
