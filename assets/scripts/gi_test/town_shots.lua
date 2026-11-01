-- The town seen from fixed poses, for screenshots (and for hunting z-fighting): a new pose every 150 frames after the
-- model has streamed in.
--   lathe --game=lua --script=assets/scripts/gi_test/town_shots.lua --player --gi --exit-after-frames=1000 \
--         --inject-keys=330:301,480:301,630:301,780:301,930:301
local town = require("gi_test.town_walk")

local POSES = {
    { eye = { 277, 2.5, 290 }, look = { 240, 6, 290 } },
    { eye = { 277, 2.5, 330 }, look = { 250, 15, 330 } },
    { eye = { 277, 2.5, 250 }, look = { 250, 12, 250 } },
    { eye = { 277, 3, 200 }, look = { 277, 20, 262 } },
    { eye = { 277, 60, 304 }, look = { 230, 10, 304 }, fov = 80 },
}
local frame = 0

-- Counted per update, not per call of pose(): the camera is asked for more than once a frame.
local on_update = town.on_update
town.on_update = function(dt)
    frame = frame + 1
    return on_update(dt)
end

local shown = 0
town.pose = function()
    local index = math.max(1, math.min(#POSES, math.floor((frame - 190) / 150) + 1))
    if index ~= shown then
        shown = index
        print("town: pose " .. index .. " from update " .. frame)
    end
    return POSES[index]
end

return town
