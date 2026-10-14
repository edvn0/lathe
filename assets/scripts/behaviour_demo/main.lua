-- Behaviour demo: cubes driven by assets/scripts/behaviours/spinner.lua and probe.lua.
--   lathe --game lua --script assets/scripts/behaviour_demo/main.lua
---@type GameScript
local M = {}

function M.on_populate()
    local cube = assets.cube()
    local colours = { { "red", 0.9, 0.2, 0.15 }, { "green", 0.2, 0.8, 0.3 }, { "blue", 0.2, 0.35, 0.9 } }
    local first
    for index, c in ipairs(colours) do
        local block = scene.spawn("block_" .. c[1])
        block:set_model(cube)
        block:set_material(assets.material(c[1], c[2], c[3], c[4], 0.5, 0.0))
        block:set_position((index - 2) * 2.5, 1.0, 0.0)
        first = first or block
        block:add_behaviour("spinner", { speed = 60 * index, axis = ({ "x", "y", "z" })[index], bob_height = index == 2 and 0.5 or 0 })
    end
    local probe = scene.spawn("probe")
    probe:set_position(0, 3, 0)
    probe:add_behaviour("probe", { target = first, tex = assets.texture("scene_colour"), label = "hello" })
end

function M.camera(aspect)
    return { eye = { 0.0, 3.0, -8.0 }, target = { 0.0, 1.0, 0.0 }, fov = 55 }
end

return M
