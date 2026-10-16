-- Rules of snake/logic.lua, run by test/snake_lua_test.cxx. A failed check raises an error that fails the run.
local Logic = require("snake.logic")
local function check(c, m) if not c then error("FAIL: " .. m, 2) end end
local seq = 0
local function rnd(n) seq = seq + 1; return (seq * 7) % n + 1 end
local cfg = { width = 10, height = 10, speed = 10, spawn_interval = 1, max_food = 3, start_length = 3 }
local s = Logic.new(cfg, rnd)
check(#s.snake == 3 and s.snake[1].x == 3 and s.snake[3].x == 1, "start layout")
check(#s.food == 1, "one food at start")
-- no reversal, queue limit
check(not Logic.turn(s, "left"), "reverse refused")
check(Logic.turn(s, "up") and Logic.turn(s, "left") and Logic.turn(s, "down") and Logic.turn(s, "right"), "queue of four")
check(not Logic.turn(s, "up"), "fifth turn refused")
-- step uses queued turn
Logic.step(s); check(s.direction[2] == 1 and s.snake[1].y == 6, "turned up")
-- wall death
local w = Logic.new({ width = 5, height = 5, speed = 5, spawn_interval = 0, max_food = 1 }, rnd)
for _ = 1, 10 do Logic.step(w) end
check(not w.alive, "wall kills")
-- self collision: grow a long snake in a tight loop
local c = Logic.new({ width = 8, height = 8, speed = 5, spawn_interval = 0, max_food = 1, start_length = 5 }, rnd)
c.snake = { {x=3,y=3},{x=3,y=2},{x=4,y=2},{x=4,y=3},{x=4,y=4} }
c.occupied = {}; for _, p in ipairs(c.snake) do c.occupied[p.y*8+p.x] = true end
c.direction = Logic.DIRECTIONS.right; c.queued = {}
c.food = {}; c.food_at = {}
Logic.step(c); check(not c.alive, "hits body (4,3)")
-- chasing the tail is legal
local t = Logic.new({ width = 6, height = 6, speed = 5, spawn_interval = 0, max_food = 1, start_length = 4 }, rnd)
t.snake = { {x=1,y=1},{x=1,y=2},{x=2,y=2},{x=2,y=1} }
t.occupied = {}; for _, p in ipairs(t.snake) do t.occupied[p.y*6+p.x] = true end
t.direction = Logic.DIRECTIONS.right; t.food = {}; t.food_at = {}; t.queued = {}
check(Logic.step(t) == "moved" and t.alive, "tail cell is free")
-- eating grows and scores, food respawns
local e = Logic.new({ width = 6, height = 6, speed = 5, spawn_interval = 0, max_food = 1, start_length = 3 }, rnd)
local h = e.snake[1]
e.food = { {x=h.x+1,y=h.y} }; e.food_at = {}; e.food_at[h.y*6+h.x+1] = e.food[1]
check(Logic.step(e) == "ate" and e.score == 1 and #e.snake == 4 and #e.food == 1, "eat")
-- clock: 10 cells/s, 0.35 s -> 3 steps
local u = Logic.new({ width = 40, height = 4, speed = 10, spawn_interval = 0, max_food = 1, start_length = 3 }, rnd)
check(Logic.update(u, 0.35) == 3, "three steps in 0.35s")
-- food spawning fills up to max_food and never on the snake
local f = Logic.new({ width = 6, height = 6, speed = 0.01, spawn_interval = 0.5, max_food = 4, start_length = 3 }, rnd)
Logic.update(f, 5)
check(#f.food == 4, "max food reached: " .. #f.food)
for _, p in ipairs(f.food) do check(not f.occupied[p.y*6+p.x], "food off snake") end
-- full board does not hang
local full = Logic.new({ width = 3, height = 2, speed = 1, spawn_interval = 1, max_food = 9, start_length = 2 }, rnd)
for _ = 1, 20 do Logic.spawn_food(full) end
check(#full.food + #full.snake <= 6, "never over-fills")
-- fuzz: random play never errors and invariants hold
math.randomseed(7)
for round = 1, 60 do
  local g = Logic.new({ width = 5 + round % 7, height = 5 + round % 5, speed = 8, spawn_interval = 0.7, max_food = 5, start_length = 3 }, function(n) return math.random(n) end)
  local names = { "up", "down", "left", "right" }
  for _ = 1, 300 do
    if math.random() < 0.3 then Logic.turn(g, names[math.random(4)]) end
    Logic.update(g, 0.05)
    local cells = {}
    for _, p in ipairs(g.snake) do local k = p.y*g.width+p.x; check(not cells[k] or not g.alive, "overlap"); cells[k] = true end
    for _, p in ipairs(g.food) do check(not cells[p.y*g.width+p.x], "food on snake") end
    if not g.alive then break end
  end
end
return {}
