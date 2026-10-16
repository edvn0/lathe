-- Snake rules with no engine in them: a grid, a snake, food, and a clock. Everything the player can change (speed,
-- how often food appears, how much there can be) is a field of the config, so main.lua can build one state per round.

local Logic = {}

local DIRECTIONS = {
    up = { 0, 1 },
    down = { 0, -1 },
    left = { -1, 0 },
    right = { 1, 0 },
}

Logic.DIRECTIONS = DIRECTIONS

-- cfg: width, height, speed (cells per second), spawn_interval (seconds between new food), max_food, start_length
-- random(n): an integer in 1..n (math.random in the game, a fixed sequence in tests)
function Logic.new(cfg, random)
    local state = {
        width = cfg.width,
        height = cfg.height,
        speed = cfg.speed,
        spawn_interval = cfg.spawn_interval,
        max_food = cfg.max_food,
        random = random,
        snake = {},
        occupied = {},
        food = {},
        food_at = {},
        direction = DIRECTIONS.right,
        queued = {},
        score = 0,
        alive = true,
        step_clock = 0,
        spawn_clock = 0,
        steps = 0,
    }

    local length = math.min(cfg.start_length or 3, cfg.width - 1)
    local y = cfg.height // 2

    for i = 0, length - 1 do
        local cell = { x = length - 1 - i + 1, y = y }
        state.snake[#state.snake + 1] = cell
        state.occupied[cell.y * cfg.width + cell.x] = true
    end

    Logic.spawn_food(state)

    return state
end

local function key(state, x, y)
    return y * state.width + x
end

function Logic.head(state)
    return state.snake[1]
end

-- Puts one food on a random free cell; false when the board is full.
function Logic.spawn_food(state)
    local free = state.width * state.height - #state.snake - #state.food

    if free <= 0 then
        return false
    end

    local pick = state.random(free)

    for y = 0, state.height - 1 do
        for x = 0, state.width - 1 do
            local k = key(state, x, y)

            if not state.occupied[k] and not state.food_at[k] then
                pick = pick - 1

                if pick == 0 then
                    local cell = { x = x, y = y }
                    state.food[#state.food + 1] = cell
                    state.food_at[k] = cell

                    return true
                end
            end
        end
    end

    return false
end

-- Queues a turn. Up to four are kept, so quick key presses between steps are not lost, and a turn straight back
-- into the snake is ignored.
function Logic.turn(state, name)
    local wanted = DIRECTIONS[name]

    if not wanted or not state.alive then
        return false
    end

    local last = state.queued[#state.queued] or state.direction

    if wanted[1] == -last[1] and wanted[2] == -last[2] then
        return false
    end

    if wanted[1] == last[1] and wanted[2] == last[2] then
        return false
    end

    if #state.queued >= 4 then
        return false
    end

    state.queued[#state.queued + 1] = wanted

    return true
end

-- Moves the snake one cell. Returns "ate", "moved" or "dead"; state.on_step(state, result), if set, is told too.
function Logic.step(state)
    local result = Logic.advance(state)

    if state.on_step then
        state.on_step(state, result)
    end

    return result
end

function Logic.advance(state)
    if not state.alive then
        return "dead"
    end

    if #state.queued > 0 then
        state.direction = table.remove(state.queued, 1)
    end

    local head = state.snake[1]
    local x = head.x + state.direction[1]
    local y = head.y + state.direction[2]

    if x < 0 or y < 0 or x >= state.width or y >= state.height then
        state.alive = false

        return "dead"
    end

    local k = key(state, x, y)
    local food = state.food_at[k]
    local tail = state.snake[#state.snake]

    -- The tail cell is free this step unless the snake is eating (it then keeps its tail).
    if state.occupied[k] and not (not food and tail.x == x and tail.y == y) then
        state.alive = false

        return "dead"
    end

    table.insert(state.snake, 1, { x = x, y = y })
    state.occupied[k] = true
    state.steps = state.steps + 1

    if food then
        state.food_at[k] = nil

        for i, candidate in ipairs(state.food) do
            if candidate == food then
                table.remove(state.food, i)
                break
            end
        end

        state.score = state.score + 1

        -- Eating clears the board of food: always keep one on it so the round cannot stall.
        if #state.food == 0 then
            Logic.spawn_food(state)
        end

        return "ate"
    end

    local last = table.remove(state.snake)
    state.occupied[key(state, last.x, last.y)] = nil

    return "moved"
end

-- Advances the clock by dt seconds: steps the snake at its speed and adds food at the spawn interval. Returns how
-- many steps were taken and whether the board changed enough to redraw.
function Logic.update(state, dt)
    if not state.alive then
        return 0
    end

    local taken = 0
    state.step_clock = state.step_clock + dt
    local period = 1 / state.speed

    while state.step_clock >= period and state.alive do
        state.step_clock = state.step_clock - period
        Logic.step(state)
        taken = taken + 1
    end

    if state.alive and state.spawn_interval > 0 then
        state.spawn_clock = state.spawn_clock + dt

        while state.spawn_clock >= state.spawn_interval do
            state.spawn_clock = state.spawn_clock - state.spawn_interval

            if #state.food < state.max_food then
                Logic.spawn_food(state)
            end
        end
    end

    return taken
end

return Logic
