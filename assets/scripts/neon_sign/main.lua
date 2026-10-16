-- A neon sign: text in the world, emissive so that bloom makes it glow, with a particle emitter on the same entity that
-- sheds sparks from the outline of the letters. Run it with:
--   lathe --game=lua --script=assets/scripts/neon_sign/main.lua --player

---@type GameScript
local M = {}

local time = 0
local sign, open_sign

local function prop(name, model, x, y, z, sx, sy, sz, material)
    local e = scene.spawn(name)
    e:set_position(x, y, z)
    e:set_scale(sx, sy, sz)
    e:set_model(model)
    e:set_material(material)
    return e
end

function M.on_populate()
    scene.clear()
    scene.set_environment({ sky = "flat", skybox = false, ambient = 0.04, exposure_ev = 0, sun_intensity = 0 })
    render.set_bloom({ enabled = true, threshold = 1.0, knee = 0.5, intensity = 0.35 })

    local cube = assets.cube()
    local brick = assets.material("neon_wall", 0.09, 0.07, 0.08, 0.9)
    local floor = assets.material("neon_floor", 0.05, 0.05, 0.07, 0.35)

    prop("wall", cube, 0, 2.2, 0.4, 14, 7, 0.4, brick)
    prop("floor", cube, 0, -0.25, -2, 14, 0.5, 10, floor)

    -- The sign: pink, eight times brighter than white, so it is well past the bloom threshold.
    sign = scene.spawn("sign")
    sign:set_position(0, 2.6, 0)
    sign:add_text({ text = "NEON\nBAR", size = 1.3, color = { 1.0, 0.18, 0.55 }, emissive = 8, centred = true })
    -- Sparks leave the letters' outline, drift up, and cool from hot pink to a dim magenta.
    sign:add_particles({
        shape = "text",
        count = 24000,
        rate = 5000,
        lifetime = 1.8,
        speed = 0.35,
        speed_variance = 0.6,
        gravity = { 0, 0.5, 0 },
        size_start = 0.02,
        size_end = 0.0,
        colour_start = { 6.0, 1.2, 3.2, 1.0 },
        colour_end = { 2.0, 0.1, 0.8, 0.0 },
    })

    open_sign = scene.spawn("open")
    open_sign:set_position(0, 0.9, -0.1)
    open_sign:add_text({ text = "OPEN", size = 0.55, color = { 0.1, 0.9, 1.0 }, emissive = 6, centred = true })
    open_sign:add_particles({
        shape = "text",
        count = 12000,
        rate = 2500,
        lifetime = 1.2,
        speed = 0.2,
        speed_variance = 0.5,
        gravity = { 0, -0.4, 0 },
        size_start = 0.015,
        size_end = 0.0,
        colour_start = { 0.8, 4.0, 6.0, 1.0 },
        colour_end = { 0.1, 0.8, 2.0, 0.0 },
    })
end

function M.on_update(dt)
    time = time + dt

    -- A neon tube hums and now and then stutters.
    local hum = 0.9 + 0.1 * math.sin(time * 37.0)
    local stutter = (math.sin(time * 1.3) > 0.985) and 0.25 or 1.0
    sign:set_text({ emissive = 8 * hum * stutter })
    open_sign:set_text({ emissive = 6 * (0.85 + 0.15 * math.sin(time * 2.0)) })

    text.draw({ text = "text entity + emissive + text-shaped particles", x = 20, y = 16, size = 20, color = { 0.8, 0.8, 0.9, 1.0 } })
end

function M.camera(aspect)
    return { eye = { 0, 2.4, -6.5 }, target = { 0, 2.0, 0 }, fov = 50 }
end

return M
