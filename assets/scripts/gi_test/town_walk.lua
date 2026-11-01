-- Walk around the generated test town with the probe GI on.
--   lathe --game=lua --script=assets/scripts/gi_test/town_walk.lua --player --gi
-- WASD walk, Shift run, Space jump, mouse look, F toggles free flight (Q down, E up), 1 noon, 2 dusk, 3 night, Esc quit.
--
-- The model comes from tools/blender/gi_town.py (it describes what is in the town):
--   ~/dev/git/build_linux/bin/blender -b --python tools/blender/gi_town.py -- ~/Assets/Meshes/gi_town

local MODEL = "/home/edwin/Assets/Meshes/gi_town.glb"

local EYE_HEIGHT = 1.6
local WALK, RUN, FLY, FLY_FAST = 3.2, 6.5, 12.0, 40.0

---@type GameScript
local M = {}

local town, town_model
local player
local spawned = false
local announced = false
local was_jumping = false
local flying = false
local yaw, pitch = 0.0, 0.0
-- Until the collider is ready (and the player placed) the camera hangs over the plaza.
local eye = { 262, 40, 270 }
local look = { 304, 3, 304 }
local clock = 0.0

local LIGHTING = {
    [1] = { sky = "procedural", skybox = true, ambient = 0.05, exposure_ev = 0, sun_intensity = 3, sun_elevation = 55 },
    [2] = { sky = "procedural", skybox = true, ambient = 0.03, exposure_ev = 0.5, sun_intensity = 2, sun_elevation = 5 },
    [3] = { sky = "flat", skybox = false, ambient = 0.0, exposure_ev = 1, sun_intensity = 0 },
}

-- The plaza (a raised platform with stairs) is the middle of the town; start on the street west of it. Lathe's world is the model as Blender has it
-- with Y and Z swapped: Blender (x, y, z) is (x, z, y) here.
local function find_spawn()
    -- The terrain rises and falls, so look for the ground from high above.
    for radius = 0, 40, 2 do
        for step = 0, 7 do
            local angle = step * math.pi / 4
            local x, z = 277 + radius * math.cos(angle), 304 + radius * math.sin(angle)
            local distance, _, py = physics.raycast(x, 80, z, 0, -1, 0, 120)
            if distance and py > -5 and py < 15 then
                return x, py + 0.1, z
            end
        end
    end
end

function M.on_populate()
    -- Without this the default scene stays: its ground plane (with the debug grid) at y = 0 fights the terrain.
    scene.clear()
    scene.set_environment(LIGHTING[2])
    town = scene.spawn("town")
    town_model = assets.load_model(MODEL, { collider = true, double_sided = true })
    town:set_model(town_model)
    physics.add_mesh_collider(town, town_model)
end

function M.on_update(dt)
    clock = clock + dt

    if not spawned then
        if physics.ready(town_model) then
            if not announced then
                announced = true
                print("town: collider ready")
            end
            local x, y, z = find_spawn()
            if x then
                player = physics.character(x, y, z, { radius = 0.3, height = 1.75, step_height = 0.35 })
                yaw = math.pi / 2 -- toward the plaza (+x)
                spawned = true
                print(string.format("town: player placed at %.1f %.1f %.1f", x, y, z))
            end
        end
        return
    end

    local forward, strafe, run
    if M.drive then
        -- Tests and demos drive the player through this instead of the keyboard.
        forward, strafe, run, _, yaw = M.drive(dt, player)
    else
        local mx, my = mouse.delta()
        yaw = yaw + mx * 0.0025
        pitch = math.max(-1.45, math.min(1.45, pitch - my * 0.0025))
        forward = (key.down(key.W) and 1 or 0) - (key.down(key.S) and 1 or 0)
        strafe = (key.down(key.D) and 1 or 0) - (key.down(key.A) and 1 or 0)
        run = key.down(key.LEFT_SHIFT)
    end

    if flying then
        local cp = math.cos(pitch)
        local fx, fy, fz = math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp
        local rx, rz = math.cos(yaw), -math.sin(yaw)
        local up = (key.down(key.E) and 1 or 0) - (key.down(key.Q) and 1 or 0)
        local speed = (run and FLY_FAST or FLY) * dt
        eye = {
            eye[1] + (fx * forward + rx * strafe) * speed,
            eye[2] + (fy * forward + up) * speed,
            eye[3] + (fz * forward + rz * strafe) * speed,
        }
    else
        local jump = not M.drive and key.down(key.SPACE)
        local length = math.sqrt(forward * forward + strafe * strafe)
        local speed = run and RUN or WALK
        local vx, vz = 0.0, 0.0
        if length > 0 then
            local fx, fz = math.sin(yaw), math.cos(yaw)
            local rx, rz = fz, -fx
            vx = (fx * forward + rx * strafe) / length * speed
            vz = (fz * forward + rz * strafe) / length * speed
        end
        player:update(dt, vx, vz, jump and not was_jumping, jump)
        was_jumping = jump
        local px, py, pz = player:position()
        eye = { px, py + EYE_HEIGHT, pz }
    end

    look = {
        eye[1] + math.sin(yaw) * math.cos(pitch),
        eye[2] + math.sin(pitch),
        eye[3] + math.cos(yaw) * math.cos(pitch),
    }
end

function M.on_key(code)
    if code == key.ESCAPE then
        game.quit()
    elseif code == key.F then
        flying = not flying
        if not flying and player then
            -- Back on foot where the camera is, dropping to the ground.
            player:teleport(eye[1], eye[2] - EYE_HEIGHT, eye[3])
        end
    elseif code == key.NUM_1 then
        scene.set_environment(LIGHTING[1])
    elseif code == key.NUM_2 then
        scene.set_environment(LIGHTING[2])
    elseif code == key.NUM_3 then
        scene.set_environment(LIGHTING[3])
    end
end

function M.camera(_aspect)
    if M.pose then
        -- Tests and demos place the camera through this instead of the player.
        local pose = M.pose()
        return { eye = pose.eye, target = pose.look, fov = pose.fov or 70, near = 0.2, far = 1200 }
    end
    -- The depth buffer is not reversed: a near plane of 5 cm leaves under 10 cm of depth resolution at 80 m, which
    -- makes anything thin (windows on a wall) fight with what is behind it. 0.2 m keeps it under 2 cm there.
    return { eye = eye, target = look, fov = 70, near = 0.2, far = 1200 }
end

function M.on_ui()
    if spawned and not M.pose then
        -- What the script sees, so a window that gets no input is easy to tell from a character that cannot move.
        ui.window("##town_hud", { x = 16, y = 16, no_inputs = true, bg_alpha = 0.6 }, function()
            ui.text_disabled("WASD move, Shift run, Space jump, F fly (Q/E down/up), 1/2/3 noon/dusk/night, Esc quit")
            ui.text(string.format("pos %.1f %.1f %.1f%s", eye[1], eye[2] - EYE_HEIGHT, eye[3], flying and "  (flying)" or ""))
            local held = ""
            for _, name in ipairs({ "W", "A", "S", "D" }) do
                held = held .. (key.down(key[name]) and name or "-")
            end
            ui.text("keys " .. held)
        end)
    end
    if not spawned then
        ui.window("loading", { centred = true }, function()
            ui.text("Loading the town" .. string.rep(".", math.floor(clock * 2) % 4), 1.5)
        end)
    end
end

return M
