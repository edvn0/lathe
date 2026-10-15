-- A stand-in for the engine's Lua API, enough to run assets/scripts/snake/main.lua in a test: entities that remember
-- where they were put, a UI whose buttons are pressed by setting PRESSED, and checks on what ended up on the board.

key = { ENTER = 257, SPACE = 32, ESCAPE = 256, UP = 265, DOWN = 264, LEFT = 263, RIGHT = 262 }
for code = 65, 90 do key[string.char(code)] = code end

WINDOWS = {}
TEXTS = {}
PRESSED = nil
LAST_WINDOW = nil
QUIT = false
ENTITIES = {}

local function entity(name)
    local e = { name = name, position = { 0, 0, 0 }, scale = { 1, 1, 1 }, material = nil }
    function e:valid() return true end
    function e:set_position(x, y, z) self.position = { x, y, z } end
    function e:set_scale(x, y, z) self.scale = { x, y or x, z or x } end
    function e:set_euler(x, y, z) self.euler = { x, y, z } end
    function e:destroy() self.destroyed = true end
    function e:set_model(model) self.model = model end
    function e:set_material(material) self.material = material end
    return e
end

scene = {
    spawn = function(name) local e = entity(name); ENTITIES[#ENTITIES + 1] = e; return e end,
    clear = function() ENTITIES = {} end,
    set_environment = function() end,
}

assets = {
    cube = function() return { path = "cube" } end,
    sphere = function() return { path = "sphere" } end,
    load_model = function(source)
        if type(source) == "string" then return { path = source } end
        assert(source:ok(), "load_model needs a finished download")
        return { path = "downloaded:" .. source.url }
    end,
    material = function(name, r, g, b, roughness, metallic, options)
        assert(type(options) == "table" and options.emissive, "materials are emissive")
        return { name = name }
    end,
}

render = { set_bloom = function() end, bloom = function() return {} end }
game = { player_mode = false, time = function() return 12.5 end, quit = function() QUIT = true end }

ui = {
    window = function(id, options, body) LAST_WINDOW = id; body() end,
    text = function(s) TEXTS[#TEXTS + 1] = s end,
    text_centred = function(s) TEXTS[#TEXTS + 1] = s end,
    text_disabled = function(s) TEXTS[#TEXTS + 1] = s end,
    button = function(label) return PRESSED == label end,
    small_button = function(label) return PRESSED == label end,
    separator = function() end,
    same_line = function() end,
}

-- Downloads wait until a test finishes them with FINISH_DOWNLOADS.
HTTP_PENDING = {}
http = {
    download = function(url, options)
        assert(url:find("^https://raw.githubusercontent.com/"), "only an allowed host")
        assert(type(options.sha256) == "string" and #options.sha256 == 64, "downloads carry a sha256")

        local request = { url = url, options = options, finished = false, success = false, failure = nil }
        function request:done() return self.finished end
        function request:ok() return self.finished and self.success end
        function request:error() return self.failure end
        function request:path() return self.success and "/cache/" .. options.name or nil end
        function request:cancel() end
        HTTP_PENDING[#HTTP_PENDING + 1] = request

        return request
    end,
    get = function() error("snake does not use http.get") end,
    pending = function() return #HTTP_PENDING end,
}

function FINISH_DOWNLOADS(success)
    local pending = HTTP_PENDING
    HTTP_PENDING = {}

    for _, request in ipairs(pending) do
        request.finished, request.success = true, success
        request.failure = success and nil or "timeout"
        request.options.on_done(request)
    end
end

local HIDDEN = -1000

-- Where the pieces are, as lists of {x, z}.
function VISIBLE()
    local segs, foods = {}, {}

    for _, e in ipairs(ENTITIES) do
        if e.position[2] > HIDDEN / 2 then
            if e.name:find("_seg_", 1, true) then
                segs[#segs + 1] = { e.position[1], e.position[3] }
            elseif e.name:find("_food_", 1, true) or e.name:find("_avocado_", 1, true) then
                foods[#foods + 1] = { e.position[1], e.position[3] }
            end
        end
    end

    return segs, foods
end

-- "avocado" when the food on the board is the downloaded model, else "sphere".
function FOOD_KIND()
    for _, e in ipairs(ENTITIES) do
        if e.position[2] > HIDDEN / 2 and e.name:find("_avocado_", 1, true) then return "avocado" end
    end
    return "sphere"
end

-- How many pieces wear the head model and how many eyes are showing.
function HEADS()
    local heads, eyes = 0, 0
    for _, e in ipairs(ENTITIES) do
        if e.position[2] > HIDDEN / 2 then
            if e.name:find("_seg_", 1, true) and e.model and e.model.path:find("head", 1, true) then heads = heads + 1 end
            if e.name:find("_eye_", 1, true) then eyes = eyes + 1 end
        end
    end
    return heads, eyes
end

function COUNT(pattern)
    local n = 0
    for _, e in ipairs(ENTITIES) do
        if e.name:find(pattern, 1, true) then n = n + 1 end
    end
    return n
end

-- "" when the board is sound: every visible piece is on one board, no two segments share a cell, food does not sit
-- on the snake. Otherwise a message.
function CHECK_BOARD()
    local Config = require("snake.config")
    local segs, foods = VISIBLE()

    if #segs == 0 then return "no snake is visible" end

    local index = math.floor(segs[1][1] / 100 + 0.5) + 1
    local size = Config.sizes[index]
    if not size then return "the snake is on no board" end

    local ox = (index - 1) * 100
    local taken = {}

    local function cell(p)
        local cx, cz = math.floor(p[1] - ox), math.floor(p[2])
        if cx < 0 or cz < 0 or cx >= size.width or cz >= size.height then return nil end
        return cz * size.width + cx
    end

    for _, p in ipairs(segs) do
        local c = cell(p)
        if not c then return "a segment is off the board" end
        if taken[c] then return "two segments share a cell" end
        taken[c] = true
    end

    for _, p in ipairs(foods) do
        local c = cell(p)
        if not c then return "food is off the board" end
        if taken[c] then return "food is on the snake" end
    end

    local heads, eyes = HEADS()
    if heads ~= 1 then return "expected one head, found " .. heads end
    if eyes ~= 2 then return "expected two eyes, found " .. eyes end

    return ""
end

function HAS_TEXT(s)
    for _, t in ipairs(TEXTS) do if t:find(s, 1, true) then return true end end
    return false
end
