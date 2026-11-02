-- The town seen from fixed poses, for screenshots (and for hunting z-fighting): a new pose every 150 frames after the
-- model has streamed in.
--   lathe --game=lua --script=assets/scripts/gi_test/town_shots.lua --player --exit-after-frames=1000 \
--         --inject-keys=330:301,480:301,630:301,780:301,930:301
-- The walker defines the callbacks; this script wraps its on_update and gives it a `pose` (it looks that up when it runs).
require("gi_test.town_walk")
local base_update = on_update

local POSES = {
    { eye = { 276, 2.6, 296 }, look = { 292, 1.2, 304 }, fov = 70 }, -- the west staircase of the plaza
    { eye = { 270, 14, 266 }, look = { 304, 4, 304 }, fov = 70 }, -- the plaza from the south-west
    { eye = { 492, 10, 70 }, look = { 516, 8, 84 }, fov = 70 }, -- the watchtower
    { eye = { 277, 3, 200 }, look = { 277, 20, 262 } },
    { eye = { 277, 60, 304 }, look = { 230, 10, 304 }, fov = 80 },
}
local frame = 0

-- Counted per update, not per call of pose(): the camera is asked for more than once a frame.
function on_update(dt)
    frame = frame + 1
    return base_update(dt)
end

local shown = 0
function pose()
    local index = math.max(1, math.min(#POSES, math.floor((frame - 190) / 150) + 1))
    if index ~= shown then
        shown = index
        print("town: pose " .. index .. " from update " .. frame)
    end
    return POSES[index]
end
