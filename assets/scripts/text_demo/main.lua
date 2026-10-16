-- Text drawn on the GPU: screen-space lines over the scene, and text placed in the world. Both come from one atlas of
-- distance fields, so any size is sharp. Run it with:  lathe --game=lua --script=assets/scripts/text_demo/main.lua --player

---@type GameScript
local M = {}

local time = 0

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
    scene.set_environment({ sky = "procedural", skybox = true, ambient = 1.0 })

    local cube = assets.cube()
    local floor = assets.material("text_floor", 0.12, 0.14, 0.18)
    local block = assets.material("text_block", 0.8, 0.35, 0.2)

    prop("floor", cube, 0, -0.5, 0, 24, 1, 24, floor)
    prop("block", cube, 0, 1, 2, 2, 2, 2, block)
end

function M.on_update(dt)
    time = time + dt

    -- Screen space: top-left anchored, in pixels. The size is free.
    text.draw({ text = "Lathe text", x = 24, y = 20, size = 72 })
    text.draw({
        text = "Signed distance fields from FreeType outlines,\nrasterised and packed into an atlas by compute shaders.",
        x = 28,
        y = 104,
        size = 22,
        color = { 0.75, 0.85, 1.0, 1.0 },
    })

    local sizes = { 12, 16, 24, 40 }
    local y = 190

    for _, size in ipairs(sizes) do
        text.draw({ text = string.format("%d px: The quick brown fox jumps over the lazy dog 0123456789", size), x = 28, y = y, size = size })
        y = y + size * 1.4
    end

    text.draw({ text = "AVAVA To Ty Wa", x = 28, y = y + 10, size = 48, color = { 1.0, 0.8, 0.2, 1.0 } })
    text.draw({ text = "Translucent", x = 28, y = y + 70, size = 48, color = { 1.0, 1.0, 1.0, 0.35 } })

    local width, height = text.measure("right aligned", 28)
    text.draw({ text = "right aligned", x = 1280 - width - 24, y = 20, size = 28, color = { 0.6, 1.0, 0.6, 1.0 } })

    -- World space: depth tested against the scene, so the block hides what is behind it.
    text.draw_world({ text = "On the floor", position = { -4, 0.02, 6 }, rotation = { 90, 0, 0 }, size = 1.2, color = { 1.0, 0.9, 0.5, 1.0 } })
    text.draw_world({ text = "Standing", position = { -5, 0.2, 3 }, size = 1.0, color = { 0.5, 1.0, 0.8, 1.0 } })
    text.draw_world({ text = "Spinning", position = { 3, 3.2, 2 }, rotation = { 0, time * 60, 0 }, size = 0.9, color = { 1.0, 0.5, 0.7, 1.0 } })
    text.draw_world({ text = "Behind the block", position = { -1.2, 1.0, 3.2 }, size = 0.5, color = { 1.0, 1.0, 1.0, 1.0 } })
end

function M.camera(aspect)
    return { eye = { 0, 5, -9 }, target = { 0, 1.2, 3 }, fov = 55 }
end

return M
