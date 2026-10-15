-- Lights and bloom demo: orbiting emissive beads with point lights, ceiling spot lights, and a row of spheres across
-- roughness and metal. Started with --game=lights.
-- Keys: 1-3 camera shot, B bloom, -/= bloom intensity, T/G bloom threshold.

---@type GameScript
local M = {}

local TAU = 2 * math.pi
local SHOTS = 3

local time = 0
local shot = 0
local orbiters = {}
local eye = { 0, 3, 12 }
local look_at = { 0, 1.2, -3 }
local fov = 45

local function clamp(value, low, high)
    return math.max(low, math.min(high, value))
end

-- Hue in [0, 1) to an {r, g, b} colour.
local function hue(h)
    local function channel(offset)
        return clamp(math.abs((h * 6 + offset) % 6 - 3) - 1, 0, 1)
    end
    return { channel(0), channel(4), channel(2) }
end

local function scaled(colour, factor, offset)
    return { colour[1] * factor + offset, colour[2] * factor + offset, colour[3] * factor + offset }
end

local material_count = 0

local function material(colour, metallic, roughness, emissive, emissive_strength)
    material_count = material_count + 1
    return assets.material(
        "lights_material_" .. material_count,
        colour[1], colour[2], colour[3], roughness, metallic,
        { emissive = emissive, emissive_strength = emissive_strength, double_sided = true }
    )
end

local function prop(name, model, position, scale, mat)
    local e = scene.spawn(name)
    e:set_position(position[1], position[2], position[3])
    e:set_scale(scale[1], scale[2], scale[3])
    e:set_model(model)
    if mat then
        e:set_material(mat)
    end
    return e
end

local function spot(name, from, to, light)
    local e = scene.spawn(name)
    e:set_position(from[1], from[2], from[3])
    e:aim(to[1], to[2], to[3])
    light.type = "spot"
    e:add_light(light)
    return e
end

function M.on_populate()
    scene.clear()
    orbiters = {}
    time = 0

    scene.set_environment({ sky = "flat", skybox = false, ambient = 0.02, exposure_ev = 0, sun_intensity = 0 })

    local cube, sphere, capsule = assets.cube(), assets.sphere(), assets.capsule()

    -- Matte grey floor and back wall: low-contrast gradients under the spot cones are where banding shows.
    local matte = material({ 0.55, 0.55, 0.58 }, 0, 0.9)
    prop("floor", cube, { 0, -0.25, 0 }, { 24, 0.25, 24 }, matte)
    prop("back_wall", cube, { 0, 6, -9 }, { 24, 6, 0.25 }, matte)

    -- Sphere rows spanning roughness and metal/dielectric: specular glints are the firefly candidates.
    local row_count = 7
    for i = 0, row_count - 1 do
        local t = i / (row_count - 1)
        local x = (t - 0.5) * 12
        local roughness = math.max(0.04, t)
        prop("metal_" .. i, sphere, { x, 0.8, -3 }, { 0.8, 0.8, 0.8 }, material({ 0.95, 0.78, 0.45 }, 1, roughness))
        prop("plastic_" .. i, sphere, { x, 0.8, -5.5 }, { 0.8, 0.8, 0.8 },
            material(scaled(hue(t), 0.8, 0.1), 0, roughness))
    end

    local grey = { 0.8, 0.8, 0.8 }
    prop("pillar_left", cube, { -4.5, 1.5, 1.5 }, { 0.4, 1.5, 0.4 }, material(grey, 0, 0.5))
    prop("pillar_right", cube, { 4.5, 1.5, 1.5 }, { 0.4, 1.5, 0.4 }, material(grey, 0, 0.5))
    prop("capsule", capsule, { 0, 1, 1.5 }, { 1, 1, 1 }, material({ 0.9, 0.9, 0.9 }, 0, 0.3))

    -- Orbiting point lights, each with a small emissive bead; the bead's HDR value is what drives bloom.
    local orbiter_count = 14
    for i = 0, orbiter_count - 1 do
        local t = i / orbiter_count
        local colour = hue(t)
        local bead = prop("orbiter_" .. i, sphere, { 0, 1, 0 }, { 0.12, 0.12, 0.12 },
            material({ 0.02, 0.02, 0.02 }, 0, 1, colour, 6 + 6 * (i % 3)))
        bead:add_light({ type = "point", colour = colour, intensity = 5, range = 7 })

        orbiters[#orbiters + 1] = {
            entity = bead,
            radius = 3 + 0.45 * (i % 7),
            height = 0.6 + 0.35 * (i % 5),
            speed = (i % 2 == 0 and 0.35 or -0.25) * (1 + 0.1 * (i % 4)),
            phase = t * TAU,
        }
    end

    -- Spot lights from the ceiling: wide soft penumbras on the wall and floor.
    local spot_count = 5
    for i = 0, spot_count - 1 do
        local t = i / (spot_count - 1)
        local x = (t - 0.5) * 14
        spot("spot_" .. i, { x, 9, 3 }, { x * 0.8, 3, -9 }, {
            colour = i % 2 == 0 and { 1, 0.85, 0.65 } or { 0.6, 0.75, 1 },
            intensity = 100, range = 30, inner_cone = 4, outer_cone = 22,
        })
    end

    -- Two dim floor-level spots grazing the floor: long smooth gradients.
    for i = 0, 1 do
        local side = i == 0 and -1 or 1
        spot("graze_" .. i, { side * 10, 0.6, 6 }, { -side * 2, 0, -4 }, {
            colour = { 1, 0.9, 0.8 }, intensity = 25, range = 30, inner_cone = 2, outer_cone = 35,
        })
    end

    print("[lights] Keys: 1-3 shot, B bloom, -/= intensity, T/G threshold")
end

function M.on_update(dt)
    time = time + dt

    for _, o in ipairs(orbiters) do
        local angle = o.phase + time * o.speed
        o.entity:set_position(
            math.cos(angle) * o.radius,
            o.height + 0.2 * math.sin(time + o.phase),
            math.sin(angle) * o.radius - 2
        )
    end

    if shot == 0 then
        local yaw = math.sin(time * 0.15) * 0.35
        eye = { math.sin(yaw) * 12, 3, math.cos(yaw) * 12 }
        look_at = { 0, 1.2, -3 }
        fov = 45
    elseif shot == 1 then
        eye, look_at, fov = { -2.5, 1.4, 4 }, { 1, 1, -4 }, 40
    else -- wall gradient close-up for banding
        eye, look_at, fov = { 0, 4, 3 }, { 0, 4, -9 }, 55
    end
end

function M.on_key(code)
    if code >= key.NUM_1 and code < key.NUM_1 + SHOTS then
        shot = code - key.NUM_1
        return
    end

    local bloom = render.bloom()
    if code == key.B then
        render.set_bloom({ enabled = not bloom.enabled })
    elseif code == key.MINUS then
        render.set_bloom({ intensity = bloom.intensity - 0.02 })
    elseif code == key.EQUAL then
        render.set_bloom({ intensity = bloom.intensity + 0.02 })
    elseif code == key.T then
        render.set_bloom({ threshold = bloom.threshold + 0.25 })
    elseif code == key.G then
        render.set_bloom({ threshold = bloom.threshold - 0.25 })
    else
        return
    end

    bloom = render.bloom()
    print(string.format("[lights] bloom %s intensity %.2f threshold %.2f",
        bloom.enabled and "on" or "off", bloom.intensity, bloom.threshold))
end

function M.camera()
    return { eye = eye, target = look_at, fov = fov, near = 0.1, far = 200 }
end

return M
