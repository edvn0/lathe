-- Snake, entirely in Lua. The rules are snake/logic.lua and the menu's choices are snake/config.lua; this file puts
-- them on screen: one board per map size, all built when the game starts, with a top-down orthographic camera that
-- glides to whichever board is chosen. Started with --game=snake.
-- Keys: arrows or WASD steer, Space/P pause, R restart, Enter starts, Esc pauses (or quits from the menu).

local Config = require("snake.config")
local Logic = require("snake.logic")

---@type GameScript
local M = {}

local PITCH = 100 -- distance between boards along x, so only one is ever in view
local HIDDEN_Y = -1000 -- where unused pieces wait; far below the camera's far plane
local CAMERA_HEIGHT = 60
local MARGIN = 3 -- cells of space around a board
local PIECE_Y = 0.45
local POOL = 64 -- snake segments built up front per board; a longer snake adds more as it grows
local MAX_DT = 0.25 -- a hitch longer than this does not make the snake lurch

local sel = { size = 1, speed = 2, spawn = 2 }
local screen = "menu" -- menu, playing, paused, over
local materials = {}
local maps = {}
local round, map
local ents, free = {}, {}
local food_ents, food_free = {}, {}
local best = {}
local over_time = 0

-- The camera glides here; the numbers are cells.
local cam = { x = 0, z = 0, w = 12 + 2 * MARGIN, h = 12 + 2 * MARGIN }

local function current_size() return Config.sizes[sel.size] end

local function best_key()
    return sel.size .. ":" .. sel.speed .. ":" .. sel.spawn
end

local function cell_position(m, cell)
    return m.ox + cell.x + 0.5, m.oz + cell.y + 0.5
end

local function hide(entity, m)
    entity:set_position(m.ox, HIDDEN_Y, 0)
end

local function make_material(name, r, g, b, glow)
    return assets.material(name, r, g, b, 0.8, 0.0, { emissive = { r, g, b }, emissive_strength = glow })
end

local function prop(name, model, x, y, z, sx, sy, sz, material)
    local e = scene.spawn(name)
    e:set_position(x, y, z)
    e:set_scale(sx, sy, sz)
    e:set_model(model)
    e:set_material(material)
    return e
end

-- Everything one board needs, made once: its floor and walls, a pool of segments and the most food the spawn
-- settings allow. Pieces not in use sit out of sight.
local function build_map(index, size)
    local cube, sphere = assets.cube(), assets.sphere()
    local m = { ox = (index - 1) * PITCH, oz = 0, width = size.width, height = size.height, segs = {}, foods = {} }
    local w, h = size.width, size.height
    local name = "snake_" .. size.name

    prop(name .. "_floor", cube, m.ox + w / 2, -0.25, m.oz + h / 2, w / 2, 0.25, h / 2, materials.floor)

    local t = 0.25 -- half thickness of a wall
    prop(name .. "_wall_s", cube, m.ox + w / 2, 0.25, m.oz - t, w / 2 + 2 * t, 0.5, t, materials.wall)
    prop(name .. "_wall_n", cube, m.ox + w / 2, 0.25, m.oz + h + t, w / 2 + 2 * t, 0.5, t, materials.wall)
    prop(name .. "_wall_w", cube, m.ox - t, 0.25, m.oz + h / 2, t, 0.5, h / 2, materials.wall)
    prop(name .. "_wall_e", cube, m.ox + w + t, 0.25, m.oz + h / 2, t, 0.5, h / 2, materials.wall)

    m.name = name
    m.cube = cube

    for i = 1, math.min(POOL, w * h) do
        m.segs[i] = prop(name .. "_seg_" .. i, cube, m.ox, HIDDEN_Y, 0, 0.42, 0.42, 0.42, materials.body)
    end

    local max_food = 0
    for _, spawn in ipairs(Config.spawns) do
        max_food = math.max(max_food, spawn.max_food)
    end

    for i = 1, max_food do
        m.foods[i] = prop(name .. "_food_" .. i, sphere, m.ox, HIDDEN_Y, 0, 0.32, 0.32, 0.32, materials.food)
    end

    return m
end

local function place_segment(entity, cell, material)
    local x, z = cell_position(map, cell)
    entity:set_position(x, PIECE_Y, z)
    entity:set_material(material)
end

-- A segment from the pool, or a new one when a long snake has used them all.
local function take_segment()
    local entity = table.remove(free)

    if entity then
        return entity
    end

    entity = prop(map.name .. "_seg_" .. (#map.segs + 1), map.cube, map.ox, HIDDEN_Y, 0, 0.42, 0.42, 0.42,
        materials.body)
    map.segs[#map.segs + 1] = entity

    return entity
end

local function sync_food()
    local seen = {}

    for _, cell in ipairs(round.food) do
        seen[cell] = true

        if not food_ents[cell] then
            local entity = table.remove(food_free)

            if entity then
                local x, z = cell_position(map, cell)
                entity:set_position(x, PIECE_Y, z)
                food_ents[cell] = entity
            end
        end
    end

    for cell, entity in pairs(food_ents) do
        if not seen[cell] then
            hide(entity, map)
            food_free[#food_free + 1] = entity
            food_ents[cell] = nil
        end
    end
end

-- Puts the pieces of the current round away, so another board can be shown.
local function clear_board()
    if not map then
        return
    end

    for _, entity in ipairs(ents) do
        hide(entity, map)
    end

    for cell, entity in pairs(food_ents) do
        hide(entity, map)
        food_ents[cell] = nil
    end

    ents, free, food_free = {}, {}, {}
end

local function on_step(_, result)
    local head = round.snake[1]

    if result == "moved" then
        local entity = table.remove(ents)
        ents[1]:set_material(materials.body)
        place_segment(entity, head, materials.head)
        table.insert(ents, 1, entity)
    elseif result == "ate" then
        ents[1]:set_material(materials.body)
        local entity = take_segment()
        place_segment(entity, head, materials.head)
        table.insert(ents, 1, entity)
    elseif result == "dead" then
        ents[1]:set_material(materials.dead)
    end

    sync_food()
end

-- A fresh round on the board the menu has selected. The round is only shown until Start is pressed.
local function new_round()
    clear_board()

    local size, speed, spawn = current_size(), Config.speeds[sel.speed], Config.spawns[sel.spawn]

    map = maps[sel.size]
    round = Logic.new({
        width = size.width,
        height = size.height,
        speed = speed.value,
        spawn_interval = spawn.interval,
        max_food = spawn.max_food,
        start_length = 3,
    }, function(n) return math.random(n) end)
    round.on_step = on_step

    for i = #map.segs, 1, -1 do
        free[#free + 1] = map.segs[i]
    end

    for i = #map.foods, 1, -1 do
        food_free[#food_free + 1] = map.foods[i]
    end

    for _, cell in ipairs(round.snake) do
        local entity = take_segment()
        place_segment(entity, cell, #ents == 0 and materials.head or materials.body)
        ents[#ents + 1] = entity
    end

    sync_food()
    over_time = 0
end

local function start_round()
    new_round()
    screen = "playing"
end

local function to_menu()
    screen = "menu"
    new_round()
end

function M.on_populate()
    scene.clear()
    math.randomseed(math.floor(game.time() * 1000) % 2147483647)

    scene.set_environment({ sky = "flat", skybox = false, ambient = 1.0, exposure_ev = 0, sun_intensity = 0 })
    render.set_bloom({ enabled = true, threshold = 1.0, intensity = 0.12 })

    materials.floor = make_material("snake_floor", 0.05, 0.07, 0.09, 1.0)
    materials.wall = make_material("snake_wall", 0.45, 0.5, 0.58, 0.7)
    materials.body = make_material("snake_body", 0.1, 0.65, 0.25, 1.0)
    materials.head = make_material("snake_head", 0.35, 1.0, 0.45, 1.4)
    materials.dead = make_material("snake_dead", 0.9, 0.2, 0.15, 1.2)
    materials.food = make_material("snake_food", 1.0, 0.2, 0.15, 4.0)

    for i, size in ipairs(Config.sizes) do
        maps[i] = build_map(i, size)
    end

    new_round()
    cam.x, cam.z = map.ox + map.width / 2, map.oz + map.height / 2
    cam.w, cam.h = map.width + 2 * MARGIN, map.height + 2 * MARGIN
end

local function steer(name)
    if screen == "playing" then
        Logic.turn(round, name)
    end
end

function M.on_key(code)
    if code == key.UP or code == key.W then
        steer("up")
    elseif code == key.DOWN or code == key.S then
        steer("down")
    elseif code == key.LEFT or code == key.A then
        steer("left")
    elseif code == key.RIGHT or code == key.D then
        steer("right")
    elseif code == key.SPACE or code == key.P then
        if screen == "playing" then
            screen = "paused"
        elseif screen == "paused" then
            screen = "playing"
        end
    elseif code == key.R then
        if screen ~= "menu" then
            start_round()
        end
    elseif code == key.ENTER then
        if screen == "menu" or screen == "over" then
            start_round()
        elseif screen == "paused" then
            screen = "playing"
        end
    elseif code == key.ESCAPE then
        if screen == "playing" then
            screen = "paused"
        elseif screen == "paused" then
            to_menu()
        elseif screen == "menu" and game.player_mode then
            game.quit()
        end
    end
end

function M.on_update(dt)
    if screen == "playing" then
        Logic.update(round, math.min(dt, MAX_DT))
        sync_food()

        if not round.alive then
            screen = "over"
            over_time = 0
            best[best_key()] = math.max(best[best_key()] or 0, round.score)
            print(string.format("[snake] game over: %d", round.score))
        end
    elseif screen == "over" then
        over_time = over_time + dt
    end

    -- Glide to the chosen board. Only the camera moves; every board already exists.
    local blend = 1 - math.exp(-8 * dt)
    local size = current_size()
    local target_x, target_z = maps[sel.size].ox + size.width / 2, maps[sel.size].oz + size.height / 2

    cam.x = cam.x + (target_x - cam.x) * blend
    cam.z = cam.z + (target_z - cam.z) * blend
    cam.w = cam.w + (size.width + 2 * MARGIN - cam.w) * blend
    cam.h = cam.h + (size.height + 2 * MARGIN - cam.h) * blend
end

function M.camera(aspect)
    local height = math.max(cam.h, cam.w / math.max(aspect, 0.1))

    return {
        eye = { cam.x, CAMERA_HEIGHT, cam.z },
        target = { cam.x, 0, cam.z },
        up = { 0, 0, 1 },
        ortho_height = height,
        near = 1,
        far = 200,
    }
end

-- A row of choices: the chosen one is bracketed. Returns the index pressed, or nil.
local function choice_row(label, list, selected)
    local pressed = nil

    ui.text(label)

    for i, item in ipairs(list) do
        ui.same_line()

        if ui.small_button(i == selected and ("[ " .. item.name .. " ]") or ("  " .. item.name .. "  ")) then
            pressed = i
        end
    end

    return pressed
end

local function menu_window()
    ui.window("##snake_menu", { x = 16, y = 16, bg_alpha = 0.88 }, function()
        ui.text("SNAKE", 2.5)
        ui.separator()

        local size = choice_row("Map size  ", Config.sizes, sel.size)
        local speed = choice_row("Speed     ", Config.speeds, sel.speed)
        local spawn = choice_row("Food      ", Config.spawns, sel.spawn)

        if size or speed or spawn then
            sel.size = size or sel.size
            sel.speed = speed or sel.speed
            sel.spawn = spawn or sel.spawn
            new_round()
        end

        ui.separator()

        local record = best[best_key()]

        if record then
            ui.text_disabled("Best on these settings: " .. record)
        end

        if ui.button("Start", 260, 48, 1.4) then
            start_round()
        end

        ui.text_disabled("Arrows or WASD to steer. Enter starts.")

        if game.player_mode and ui.button("Quit", 260, 40, 1.1) then
            game.quit()
        end
    end)
end

function M.on_ui()
    if screen == "menu" then
        menu_window()
        return
    end

    ui.window("##snake_hud", { x = 16, y = 16, no_inputs = true, bg_alpha = 0.6 }, function()
        ui.text(string.format("Score %d", round.score), 1.6)
        ui.text_disabled(string.format("Best %d", best[best_key()] or 0))
    end)

    if screen == "paused" then
        ui.window("##snake_pause", { centred = true, bg_alpha = 0.92 }, function()
            ui.text_centred("Paused", 2.0)

            if ui.button("Resume") then
                screen = "playing"
            end

            if ui.button("Restart") then
                start_round()
            end

            if ui.button("Menu") then
                to_menu()
            end
        end)
    elseif screen == "over" then
        ui.window("##snake_over", { centred = true, bg_alpha = 0.92 }, function()
            ui.text_centred("Game over", 2.0)
            ui.text_centred(string.format("Score %d", round.score), 1.4)

            if ui.button("Play again") then
                start_round()
            end

            if ui.button("Menu") then
                to_menu()
            end
        end)
    end
end

return M
