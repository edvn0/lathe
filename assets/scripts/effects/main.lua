-- A post effect from Lua. The script names a shader's manifest, gives values for its params and chooses where in the
-- frame it runs; the engine builds and checks the passes. Edit assets/shaders/effects/tint.slang or tint.json while it
-- runs and the effect reloads.
--   lathe --game lua --script assets/scripts/effects/main.lua

---@type LatheGame
local M = {}

local tint
local clock = 0.0

function M.on_populate()
    local cube = assets.cube()
    local colours = {
        { "red", 0.9, 0.2, 0.15 }, { "green", 0.2, 0.8, 0.3 }, { "blue", 0.2, 0.35, 0.9 },
    }

    for index, entry in ipairs(colours) do
        local material = assets.material(entry[1], entry[2], entry[3], entry[4], 0.5, 0.0)
        local block = scene.spawn("block_" .. entry[1])
        block:set_model(cube)
        block:set_material(material)
        block:set_position((index - 2) * 2.5, 1.0, 0.0)
    end

    local shader = compute.load("assets/shaders/effects/tint.json")
    tint = compute.instance(shader, { strength = 0.6, tint = { 1.0, 0.65, 0.35 } })
    scene.add_effect(tint, "before_composite")

    -- Mistakes are Lua errors a script can catch, and never reach the renderer.
    print(select(2, pcall(function() tint:set("strength", 40) end)))
    print(select(2, pcall(function() tint:set("nonsense", 1) end)))
    print(select(2, pcall(function() scene.add_effect(tint, "after_depth") end)))
end

function M.on_update(dt)
    clock = clock + dt
    tint:set("strength", 0.5 + 0.5 * math.sin(clock))

    if tint:problem() then
        print("the effect was dropped: " .. tint:problem())
    end
end

function M.camera(aspect)
    return { eye = { 0.0, 3.0, -8.0 }, target = { 0.0, 1.0, 0.0 }, fov = 55 }
end

return M
