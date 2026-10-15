-- Snake, entirely in Lua. The rules are snake/logic.lua and the menu's choices are snake/config.lua; this file puts
-- them on screen: one board per map size, all built when the game starts, with a top-down orthographic camera that
-- glides to whichever board is chosen. The snake's head and body are small models in assets/models/snake (made by
-- tools/gen_snake_models.py); the food is a CC0 avocado fetched over http with a pinned hash, and plain spheres
-- until it arrives (or if it never does). Started with --game=snake.
-- Keys: arrows or WASD steer, Space/P pause, R restart, Enter starts, Esc pauses (or quits from the menu).

local Config = require("snake.config")
local Logic = require("snake.logic")

---@type GameScript
local M = {}

-- The food model: Khronos glTF-Sample-Assets (CC0), pinned to a commit and a hash. It is about 0.06 units tall, so
-- it is scaled up to fit a cell.
local FOOD_URL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/"
    .. "5bad5aaa0bbb5d0f9cdc934e626f27d0df1e79b8/Models/Avocado/glTF-Binary/Avocado.glb"
local FOOD_SHA256 = "85ae91e60ed519f53a7e2096508837fedd446dcc450f8e05678b7a204679bd12"
local AVOCADO_SCALE = 12
local MAX_FOOD = 8 -- the most food any spawn setting allows; set from the config below

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
local models = {}
local is_head = {} -- entity -> whether it currently wears the head model
local food_download = "idle" -- idle, fetching, ready, failed
local food_error = nil

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
        m.segs[i] = prop(name .. "_seg_" .. i, models.body, m.ox, HIDDEN_Y, 0, 1, 1, 1, materials.body)
    end

    local max_food = 0
    for _, spawn in ipairs(Config.spawns) do
        max_food = math.max(max_food, spawn.max_food)
    end

    MAX_FOOD = max_food
    m.sphere_foods = {}

    for i = 1, max_food do
        m.sphere_foods[i] = prop(name .. "_food_" .. i, sphere, m.ox, HIDDEN_Y, 0, 0.32, 0.32, 0.32, materials.food)
    end

    m.foods = m.sphere_foods

    -- Two eyes that sit on the head wherever it points.
    m.eyes = {}

    for i = 1, 2 do
        m.eyes[i] = prop(name .. "_eye_" .. i, sphere, m.ox, HIDDEN_Y, 0, 0.1, 0.1, 0.1, materials.eye)
    end

    return m
end

local function place_segment(entity, cell, material)
    local x, z = cell_position(map, cell)
    entity:set_position(x, PIECE_Y, z)
    entity:set_material(material)
end

-- Makes `entity` the head (its model, colour and facing) or an ordinary body piece. The model is only set when the
-- role changes, so a step costs a handful of calls.
local function set_role(entity, head, material)
    if is_head[entity] ~= head then
        entity:set_model(head and models.head or models.body)
        is_head[entity] = head
    end

    if not head then
        entity:set_euler(0, 0, 0)
    end

    entity:set_material(material)
end

-- Turns the head to face the way the snake is going and puts the eyes on it. The head model looks along +x.
local function face_head()
    local head = ents[1]
    local direction = round.direction
    local x, z = cell_position(map, round.snake[1])

    head:set_euler(0, math.atan(-direction[2], direction[1]), 0)

    local side = { -direction[2], direction[1] }

    for i, eye in ipairs(map.eyes) do
        local sign = i == 1 and 1 or -1
        eye:set_position(x + direction[1] * 0.3 + side[1] * 0.22 * sign, PIECE_Y + 0.17,
            z + direction[2] * 0.3 + side[2] * 0.22 * sign)
    end
end

-- A segment from the pool, or a new one when a long snake has used them all.
local function take_segment()
    local entity = table.remove(free)

    if entity then
        return entity
    end

    entity = prop(map.name .. "_seg_" .. (#map.segs + 1), models.body, map.ox, HIDDEN_Y, 0, 1, 1, 1, materials.body)
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
                entity:set_position(x, map.food_y, z)
                entity:set_euler(0, (cell.x * 7 + cell.y * 13) % 6, 0)
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

    for _, eye in ipairs(map.eyes) do
        hide(eye, map)
    end

    ents, free, food_free = {}, {}, {}
end

local function on_step(_, result)
    local head = round.snake[1]

    if result == "moved" then
        local entity = table.remove(ents)
        set_role(ents[1], false, materials.body)
        place_segment(entity, head, materials.head)
        set_role(entity, true, materials.head)
        table.insert(ents, 1, entity)
    elseif result == "ate" then
        set_role(ents[1], false, materials.body)
        local entity = take_segment()
        place_segment(entity, head, materials.head)
        set_role(entity, true, materials.head)
        table.insert(ents, 1, entity)
    elseif result == "dead" then
        ents[1]:set_material(materials.dead)
    end

    face_head()
    sync_food()
end

-- Food is the avocado once it has arrived (a pool of them per board, made the first time a board needs one), and
-- spheres before that or if it never does.
local function use_best_food(m)
    if food_download == "ready" and not m.avocados then
        m.avocados = {}

        for i = 1, MAX_FOOD do
            local entity = scene.spawn(m.name .. "_avocado_" .. i)

            entity:set_position(m.ox, HIDDEN_Y, 0)
            entity:set_scale(AVOCADO_SCALE, AVOCADO_SCALE, AVOCADO_SCALE)
            entity:set_model(models.avocado)
            m.avocados[i] = entity
        end

        for _, entity in ipairs(m.sphere_foods) do
            hide(entity, m)
        end
    end

    if m.avocados then
        m.foods, m.food_y = m.avocados, 0.02
    else
        m.foods, m.food_y = m.sphere_foods, PIECE_Y
    end
end

-- Starts fetching the avocado. The game plays with spheres until it is there.
local function fetch_food()
    if not http or food_download ~= "idle" then
        return
    end

    food_download = "fetching"

    http.download(FOOD_URL, {
        sha256 = FOOD_SHA256,
        name = "snake/avocado.glb",
        on_done = function(request)
            if request:ok() then
                models.avocado = assets.load_model(request)
                food_download = "ready"
                print("[snake] avocado ready")
            else
                food_download = "failed"
                food_error = request:error()
                print("[snake] no avocado, using spheres: " .. tostring(food_error))
            end
        end,
    })
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

    use_best_food(map)

    for i = #map.foods, 1, -1 do
        food_free[#food_free + 1] = map.foods[i]
    end

    for _, cell in ipairs(round.snake) do
        local entity = take_segment()
        local head = #ents == 0

        place_segment(entity, cell, head and materials.head or materials.body)
        set_role(entity, head, head and materials.head or materials.body)
        ents[#ents + 1] = entity
    end

    face_head()
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
    materials.eye = make_material("snake_eye", 0.02, 0.02, 0.03, 0.3)

    models.head = assets.load_model("assets/models/snake/head.glb")
    models.body = assets.load_model("assets/models/snake/body.glb")

    for i, size in ipairs(Config.sizes) do
        maps[i] = build_map(i, size)
    end

    fetch_food()
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

        if food_download == "fetching" then
            ui.text_disabled("Fetching the food model...")
        elseif food_download == "failed" then
            ui.text_disabled("Food model unavailable (" .. tostring(food_error) .. "); using spheres")
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
