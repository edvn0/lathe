-- Chess. The rules are the native engine's (require "native.chess"); the board, camera, input, menus and flow
-- (loading, menu, play, pause, game over) are all here.

local chess = require("native.chess")
local board = require("chess.board")

local PIECES = {
    "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
}

local PROMOTION_KEYS = {
    [key.Q] = "queen",
    [key.T] = "rook",
    [key.B] = "bishop",
    [key.N] = "knight",
}

local TARGET_MARKERS = 32
local STOWED = { 0.0, -50.0, 0.0 }

local CAMERA_EYE = { 0.0, 10.5, -6.5 }
local CAMERA_TARGET = { 0.0, 0.0, 0.4 }
local CAMERA_FLIP_LIFT = 3.0
local CAMERA_FLIP_RATE = 4.0

local GAME_OVER_DELAY = 1.6
local WARMUP_FRAMES = 30
local ORBIT_SPEED = 0.12

local RESULT_TEXT = {
    checkmate = { "Checkmate", nil },
    stalemate = { "Draw", "Stalemate" },
    draw_fifty_move = { "Draw", "Fifty-move rule" },
    draw_insufficient_material = { "Draw", "Insufficient material" },
    draw_repetition = { "Draw", "Threefold repetition" },
}

local engine = chess.new()

local models = {}
local board_model = nil
local piece_entities = {}
local target_entities = {}
local cursor_entity, selection_entity

-- Flow: an installed game starts on a loading screen; in the editor the game just plays.
local screen = game.player_mode and "loading" or "playing"
local loading_peak = 0
local loading_frames = 0
local game_over_timer = 0.0

local camera_angle = 0.0
local camera_target_angle = 0.0
local camera_side = "white"

local cursor = board.square(4, 1)
local hovered = nil
local selected = nil
local selected_moves = {}
local target_squares = {}
local pending_promotion = nil
local outlined = nil
local status = ""

local cursor_step = { 0, 0 }
local pending_activate = false
local pending_restart = false

local function stow(entity)
    entity:set_position(STOWED[1], STOWED[2], STOWED[3])
end

local function clear_selection()
    selected = nil
    selected_moves = {}
    target_squares = {}
    pending_promotion = nil
end

local function is_black(piece)
    return piece:sub(1, 5) == "black"
end

-- Projects the engine's state onto the 32 piece entities; captured pieces are stowed away.
local function sync_pieces()
    local visible = {}

    for _, piece in ipairs(engine:pieces()) do
        local entity = piece_entities[piece.id]

        visible[piece.id] = true

        if entity then
            entity:set_position(board.position(piece.square))
            entity:set_euler(0.0, is_black(piece.piece) and math.pi or 0.0, 0.0)
            entity:set_model(models[piece.piece])
        end
    end

    for id, entity in pairs(piece_entities) do
        if not visible[id] then
            stow(entity)
        end
    end
end

local function restart()
    engine:reset()

    camera_angle = 0.0
    camera_target_angle = 0.0
    camera_side = engine:side_to_move()

    cursor = board.square(4, 1)
    hovered = nil
    status = ""
    game_over_timer = 0.0

    clear_selection()
    sync_pieces()
end

local function start_new_game()
    pending_restart = true
    game_over_timer = 0.0
    screen = "playing"
end

local function execute(from, to, promotion)
    local result = engine:move(from, to, promotion)

    if not result.moved then
        status = "That move is not legal."
        return
    end

    status = board.name(from) .. (result.capture and "x" or "-") .. board.name(to)

    if result.state == "checkmate" then
        status = status .. " checkmate."
    elseif result.state == "stalemate" then
        status = status .. " stalemate."
    elseif result.state ~= "playing" then
        status = status .. " draw (" .. (RESULT_TEXT[result.state][2]:lower()) .. ")."
    elseif result.check then
        status = status .. " check."
    end

    clear_selection()
    sync_pieces()

    if game.player_mode and result.state ~= "playing" then
        game_over_timer = GAME_OVER_DELAY
    end
end

local function select_square(square)
    local piece = engine:piece_at(square)
    local mine = piece and is_black(piece) == (engine:side_to_move() == "black")

    if not mine then
        clear_selection()
        status = ""
        return
    end

    local moves = engine:moves(square)

    if #moves == 0 then
        clear_selection()
        status = "That piece has no legal moves."
        return
    end

    selected = square
    selected_moves = moves
    target_squares = {}
    pending_promotion = nil
    status = ""

    local seen = {}

    for _, move in ipairs(moves) do
        if not seen[move.to] then
            seen[move.to] = true
            table.insert(target_squares, move.to)
        end
    end
end

local function activate()
    if pending_promotion then
        status = "Choose a promotion piece."
        return
    end

    local square = cursor

    if selected then
        for _, move in ipairs(selected_moves) do
            if move.to == square then
                if move.promotion then
                    pending_promotion = { from = selected, to = square }
                    status = "Choose a promotion piece."
                else
                    execute(selected, square)
                end

                return
            end
        end
    end

    select_square(square)
end

local function update_camera(dt)
    local side = engine:side_to_move()

    if side ~= camera_side then
        camera_side = side
        camera_target_angle = camera_target_angle + math.pi
    end

    local remaining = camera_target_angle - camera_angle

    if math.abs(remaining) < 1e-3 then
        camera_angle = camera_target_angle
        return
    end

    camera_angle = camera_angle + remaining * (1.0 - math.exp(-CAMERA_FLIP_RATE * math.min(dt, 0.1)))
end

local function place(entity, square, height)
    if square then
        local x, y, z = board.position(square)
        entity:set_position(x, y + height, z)
    else
        stow(entity)
    end
end

local function place_markers()
    place(cursor_entity, cursor, 0.02)
    place(selection_entity, selected, 0.015)

    for index, entity in ipairs(target_entities) do
        place(entity, target_squares[index], 0.01)
    end

    -- Outline the piece under the cursor.
    local wanted = nil

    for _, piece in ipairs(engine:pieces()) do
        if piece.square == (hovered or cursor) then
            wanted = piece_entities[piece.id]
        end
    end

    if wanted ~= outlined then
        if outlined and outlined:valid() then
            outlined:set_outlined(false)
        end

        outlined = wanted

        if outlined then
            outlined:set_outlined(true)
        end
    end
end

-- Scene setup ------------------------------------------------------------------------------------------------------

local M = {}

M.wants_cursor = true

function M.on_populate()
    engine:reset()

    for _, name in ipairs(PIECES) do
        models[name] = assets.load_model("assets/models/chess/" .. name .. ".glb")
    end

    board_model = assets.load_model("assets/models/chess/chess_board.glb")

    scene.spawn("board"):set_model(board_model)

    for _, piece in ipairs(engine:pieces()) do
        local entity = scene.spawn("piece_" .. piece.id)

        entity:set_model(models[piece.piece])
        piece_entities[piece.id] = entity
    end

    local function marker(name, colour, footprint)
        local material = assets.material("chess." .. name, colour[1], colour[2], colour[3])
        local entity = scene.spawn(name)
        local size = board.square_size * footprint

        entity:set_model(assets.cube())
        entity:set_scale(size, 0.01, size)
        entity:set_material(material)
        stow(entity)

        -- The entity holds its own reference now.
        assets.release_material(material)

        return entity
    end

    cursor_entity = marker("cursor", { 0.95, 0.8, 0.1 }, 0.9)
    selection_entity = marker("selection", { 0.15, 0.45, 0.9 }, 0.96)

    for index = 1, TARGET_MARKERS do
        target_entities[index] = marker("target_" .. (index - 1), { 0.2, 0.75, 0.3 }, 0.7)
    end

    screen = game.player_mode and "loading" or "playing"
    loading_peak = 0
    loading_frames = 0
    pending_restart = true

    -- Pieces stand on their squares behind the loading screen and the menu, before any game starts.
    sync_pieces()

    print("Set up the board")
end

-- The active scene changed (editor scene -> runtime scene): find our entities again.
function M.on_bind()
    for id = 0, 31 do
        piece_entities[id] = scene.find("piece_" .. id)
    end

    cursor_entity = scene.find("cursor")
    selection_entity = scene.find("selection")
    outlined = nil

    for index = 1, TARGET_MARKERS do
        target_entities[index] = scene.find("target_" .. (index - 1))
    end

    pending_restart = true
    sync_pieces()
end

-- Per frame ---------------------------------------------------------------------------------------------------------

function M.on_update(dt)
    if game.player_mode and screen ~= "playing" then
        -- Behind the menus the board idles: it turns slowly on the menu and holds still elsewhere.
        if screen == "loading" or screen == "menu" then
            camera_angle = camera_angle + ORBIT_SPEED * math.min(dt, 0.1)
            camera_target_angle = camera_angle
        end

        return
    end

    if pending_restart then
        pending_restart = false
        restart()
    end

    if game_over_timer > 0.0 then
        game_over_timer = game_over_timer - dt

        if game_over_timer <= 0.0 then
            game_over_timer = 0.0
            screen = "game_over"
            clear_selection()
        end
    end

    update_camera(dt)

    -- Keys move the cursor as seen on screen, so mirror them from black's side.
    local dx, dy = cursor_step[1], cursor_step[2]

    cursor_step[1], cursor_step[2] = 0, 0

    if math.cos(camera_target_angle) < 0.0 then
        dx, dy = -dx, -dy
    end

    local file = math.max(0, math.min(7, board.file(cursor) + dx))
    local rank = math.max(0, math.min(7, board.rank(cursor) + dy))

    cursor = board.square(file, rank)

    if pending_activate then
        pending_activate = false
        activate()
    end

    place_markers()
end

function M.camera(aspect)
    local sin, cos = math.sin(camera_angle), math.cos(camera_angle)

    -- Orbit the board centre by `camera_angle` around the up axis, rising while it swings.
    local function spin(v)
        return { v[1] * cos + v[3] * sin, v[2], -v[1] * sin + v[3] * cos }
    end

    local eye = spin(CAMERA_EYE)

    eye[2] = eye[2] + CAMERA_FLIP_LIFT * math.abs(sin)

    return { eye = eye, target = spin(CAMERA_TARGET), fov = 40.0, near = 0.1, far = 200.0 }
end

-- Input --------------------------------------------------------------------------------------------------------------

function M.on_cursor(ndc_x, ndc_y, inside)
    hovered = nil

    if inside ~= 0 then
        local x, z = camera.pick_plane(ndc_x, ndc_y, board.top_y)

        if x then
            hovered = board.from_world(x, z)
        end
    end
end

function M.on_mouse_button(button)
    if game.player_mode and screen ~= "playing" then
        return
    end

    if button == mouse.LEFT and hovered then
        cursor = hovered
        pending_activate = true
    elseif button == mouse.RIGHT then
        clear_selection()
        status = ""
    end
end

function M.on_key(k, modifiers)
    if game.player_mode then
        if screen == "loading" then
            return
        elseif screen == "menu" or screen == "game_over" then
            if k == key.ENTER or k == key.SPACE then
                start_new_game()
            end

            return
        elseif screen == "paused" then
            if k == key.ESCAPE then
                screen = "playing"
            end

            return
        elseif k == key.ESCAPE then
            screen = "paused"
            return
        end
    end

    if k == key.UP or k == key.W then
        cursor_step[2] = cursor_step[2] + 1
    elseif k == key.DOWN or k == key.S then
        cursor_step[2] = cursor_step[2] - 1
    elseif k == key.LEFT or k == key.A then
        cursor_step[1] = cursor_step[1] - 1
    elseif k == key.RIGHT or k == key.D then
        cursor_step[1] = cursor_step[1] + 1
    elseif k == key.ENTER or k == key.SPACE then
        pending_activate = true
    elseif k == key.BACKSPACE then
        clear_selection()
        status = ""
    elseif pending_promotion and PROMOTION_KEYS[k] then
        execute(pending_promotion.from, pending_promotion.to, PROMOTION_KEYS[k])
    elseif k == key.R and modifiers == 0 then
        pending_restart = true
    end
end

-- UI -----------------------------------------------------------------------------------------------------------------

local function result_lines()
    local state = engine:state()
    local text = RESULT_TEXT[state]

    if state == "checkmate" then
        -- The side to move is the one that has been mated.
        return "Checkmate", engine:side_to_move() == "white" and "Black wins" or "White wins"
    end

    return text and text[1] or "", text and text[2] or ""
end

local function draw_loading()
    local pending = assets.pending()

    loading_peak = math.max(loading_peak, pending)
    loading_frames = loading_frames + 1

    local streaming = loading_peak == 0 and 0.0 or 1.0 - pending / loading_peak
    local warm = math.min(1.0, loading_frames / WARMUP_FRAMES)

    -- Models first; the last frames are the pipelines and the driver settling on the finished board.
    local progress = pending == 0 and 0.5 + 0.5 * warm or 0.5 * streaming

    if pending == 0 and warm >= 1.0 then
        screen = "menu"
        return
    end

    ui.window("##loading", { fullscreen = true, bg_alpha = 1.0, no_inputs = true }, function()
        local _, height = ui.display_size()

        ui.dummy(0, height * 0.5 - 60)
        ui.spinner(28, 5)
        ui.progress(progress, 300, 6)
        ui.text_centred("Loading...", 1.4)
    end)
end

local function draw_menu()
    ui.window("##menu", { centred = true, bg_alpha = 0.8 }, function()
        ui.dummy(340, 8)
        ui.text_centred("Chess", 4.0)
        ui.dummy(0, 18)

        if ui.button("Play") then
            start_new_game()
        end

        ui.dummy(0, 4)

        if ui.button("Quit") then
            game.quit()
        end

        ui.dummy(0, 8)
    end)
end

local function draw_pause()
    ui.window("##pause", { centred = true, bg_alpha = 0.85 }, function()
        ui.dummy(340, 8)
        ui.text_centred("Paused", 3.0)
        ui.dummy(0, 18)

        if ui.button("Resume") then
            screen = "playing"
        end

        ui.dummy(0, 4)

        if ui.button("Restart") then
            start_new_game()
        end

        ui.dummy(0, 4)

        if ui.button("Main menu") then
            screen = "menu"
        end

        ui.dummy(0, 4)

        if ui.button("Quit") then
            game.quit()
        end

        ui.dummy(0, 8)
    end)
end

local function draw_game_over()
    local title, reason = result_lines()

    ui.window("##game_over", { centred = true, bg_alpha = 0.88 }, function()
        ui.dummy(340, 8)
        ui.text_centred(title, 3.5)
        ui.text_centred(reason, 2.0)
        ui.dummy(0, 18)

        if ui.button("Play again") then
            start_new_game()
        end

        ui.dummy(0, 4)

        if ui.button("Main menu") then
            screen = "menu"
        end

        ui.dummy(0, 4)

        if ui.button("Quit") then
            game.quit()
        end

        ui.dummy(0, 8)
    end)
end

local function draw_hud()
    ui.window("##hud", { x = 16, y = 16, bg_alpha = 0.45 }, function()
        ui.text(engine:side_to_move() == "white" and "White to move" or "Black to move", 1.6)

        if status ~= "" then
            ui.text(status, 1.6)
        end

        if pending_promotion then
            ui.separator()
            ui.text("Promote pawn to:")

            for _, choice in ipairs({ "queen", "rook", "bishop", "knight" }) do
                if ui.small_button(choice:sub(1, 1):upper() .. choice:sub(2)) then
                    execute(pending_promotion.from, pending_promotion.to, choice)
                end

                ui.same_line()
            end
        end

        ui.text_disabled("Esc: menu")
    end)
end

function M.on_ui()
    if not game.player_mode or screen == "playing" then
        draw_hud()
    elseif screen == "loading" then
        draw_loading()
    elseif screen == "menu" then
        draw_menu()
    elseif screen == "paused" then
        draw_pause()
    elseif screen == "game_over" then
        draw_game_over()
    end
end

return M
