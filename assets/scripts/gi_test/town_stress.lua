-- Worst case for the probe volume, in the town: a camera that flies fast along the streets (the window scrolls more than
-- a probe a frame, so probes are reassigned and relit constantly), jumps 150 m every 150 frames (the whole volume is
-- reset), and a sun that sweeps from dusk to noon and back every 400 frames (every probe's light changes at once).
--   lathe --game=lua --script=assets/scripts/gi_test/town_stress.lua --player --screen-type=headless \
--         --bench-warmup 300 --bench-frames 900
local town = require("gi_test.town_walk")

local SPEED = 1.4 -- metres per frame, about 84 m/s
local frame = 0
local eye = { 277, 3, 60 }
local heading = 0.0 -- along +z

local on_update = town.on_update
town.on_update = function(dt)
    frame = frame + 1
    if frame % 400 == 1 then
        scene.set_environment({ sky = "procedural", skybox = true, ambient = 0.03, exposure_ev = 0.5, sun_intensity = 2, sun_elevation = 5 })
    elseif frame % 400 == 200 then
        scene.set_environment({ sky = "procedural", skybox = true, ambient = 0.05, exposure_ev = 0, sun_intensity = 3, sun_elevation = 60 })
    end
    return on_update(dt)
end

town.pose = function()
    if frame % 150 == 0 then
        -- Jump: a street crossing somewhere else in the town, at street height.
        local n = math.floor(frame / 150)
        eye = { 14 + ((n * 3) % 10) * 54 - 7, 3 + (n % 3), 14 + ((n * 7) % 10) * 54 - 7 }
        heading = (n % 4) * math.pi / 2
    end
    eye[1] = eye[1] + math.sin(heading) * SPEED
    eye[3] = eye[3] + math.cos(heading) * SPEED
    return { eye = eye, look = { eye[1] + math.sin(heading) * 20, eye[2] + 1, eye[3] + math.cos(heading) * 20 }, fov = 80 }
end

return town
