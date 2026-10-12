-- Walk around the Amazon Lumberyard Bistro (exterior) as an animated character, with people wandering about.
--   lathe --game lua --script assets/scripts/bistro/main.lua --player
-- WASD walk, Shift run, Space jump, mouse look, V first/third person, Esc quit.
--
-- The scene is NVIDIA's ORCA "Bistro" (CC BY 4.0, https://developer.nvidia.com/orca/amazon-lumberyard-bistro). It is
-- not part of this repo: download BistroExterior.glb with its Textures/ folder and point MODEL at it. The first run
-- cooks its ~400 textures, which takes a few minutes; later runs read them from the cache.

local MODEL = "/home/edwin/Downloads/Bistro_v5_2/BistroExterior.glb"
local PERSON = "assets/models/animated_human.glb"

local DEBUG = false
local EYE_HEIGHT = 1.6
local WALK, RUN = 3.2, 6.5

---@type LatheGame
local M = {}

local city, city_model
local player, player_actor, rig
local third_person = true
local CAMERA_DISTANCE = 3.2
local yaw, pitch = 0.0, 0.0
local eye = { 0, 10, 0 }
local look = { 0, 10, 1 }
local body_yaw = 0.0
local spawned = false
local was_jumping = false
local npcs = {}
local NPC_COUNT = 24
local NPCS_PER_FRAME = 3
local npc_turn = 0
local clock = 0.0

-- Finds street level: rays cast down from head height through a grid, keeping the first floor near y = 0. Starting low
-- keeps the rays from landing on awnings and roofs.
local function find_spawn(cx, cz, spread)
    for radius = 0, spread, 2 do
        for step = 0, 7 do
            local angle = step * math.pi / 4
            local x, z = cx + radius * math.cos(angle), cz + radius * math.sin(angle)
            local distance, _, py = physics.raycast(x, 2.5, z, 0, -1, 0, 6)
            if distance and distance > 2.0 and py > -1 and py < 1 then
                return x, py + 0.1, z
            end
        end
    end
end

local function make_npc(index, around_x, around_z)
    local x, y, z = find_spawn(around_x + (math.random() - 0.5) * 30, around_z + (math.random() - 0.5) * 30, 8)
    if not x then
        return
    end

    npcs[#npcs + 1] = {
        actor = rig:spawn(),
        character = physics.character(x, y, z, { radius = 0.28, height = 1.75, step_height = 0.35 }),
        heading = math.random() * math.pi * 2,
        retarget = 1 + math.random() * 3,
        pace = (math.random() < 0.2) and 3.2 or 1.3,
        last_x = x,
        last_z = z,
        stuck = 0,
    }
end

-- Every capsule sweep walks a 2.8M-triangle BVH, so NPCs take turns: a few per frame, each stepping by the time it has
-- saved up.
local function update_npc(npc, dt, player_x, player_z)
    local x, y, z = npc.character:position()
    -- Wander in a heading that changes every few seconds, or at once when the way is blocked.
    npc.retarget = npc.retarget - dt
    local moved = math.sqrt((x - npc.last_x) ^ 2 + (z - npc.last_z) ^ 2)
    npc.last_x, npc.last_z = x, z
    npc.stuck = (moved < 0.4 * npc.pace * dt) and npc.stuck + dt or 0
    if npc.retarget <= 0 or npc.stuck > 0.4 then
        npc.heading = npc.heading + (math.random() - 0.5) * 3.0 + (npc.stuck > 0.4 and math.pi * 0.7 or 0)
        npc.retarget = 1.5 + math.random() * 4
        npc.stuck = 0
    end

    -- Close to the player, an NPC stops and turns to look.
    local dx, dz = player_x - x, player_z - z
    local near = dx * dx + dz * dz < 3.5 * 3.5
    local vx, vz = 0.0, 0.0
    if near then
        npc.heading = math.atan(dx, dz)
    else
        vx, vz = math.sin(npc.heading) * npc.pace, math.cos(npc.heading) * npc.pace
    end

    npc.character:update(dt, vx, vz, false, false)
    local nx, ny, nz = npc.character:position()
    local _, vy = npc.character:velocity()
    npc.actor:place(nx, ny, nz, npc.heading)
    npc.actor:move(math.sqrt(vx * vx + vz * vz), npc.character:grounded(), vy)
end

function M.on_populate()
    city = scene.spawn("bistro")
    city_model = assets.load_model(MODEL, { collider = true, min_extent = 0.35 })
    city:set_model(city_model)
    physics.add_mesh_collider(city, city_model)

    rig = animation.rig(assets.load_model(PERSON), { height = 1.75 })
end

function M.on_update(dt)
    clock = clock + dt

    if not spawned then
        if physics.ready(city_model) then
            local x, y, z = find_spawn(0, 0, 40)
            if x then
                player = physics.character(x, y, z, { radius = 0.3, height = 1.75, step_height = 0.35 })
                player_actor = rig:spawn()
                for index = 1, NPC_COUNT do
                    make_npc(index, x, z)
                end
                spawned = true
            end
        end
        return
    end

    local forward, strafe, run, jump
    if M.drive then
        -- Tests and demos drive the player through this instead of the keyboard.
        forward, strafe, run, jump, yaw = M.drive(dt, player)
    else
        local mx, my = mouse.delta()
        yaw = yaw + mx * 0.0025
        pitch = math.max(-1.45, math.min(1.45, pitch - my * 0.0025))
        forward = (key.down(key.W) and 1 or 0) - (key.down(key.S) and 1 or 0)
        strafe = (key.down(key.D) and 1 or 0) - (key.down(key.A) and 1 or 0)
        run = key.down(key.LEFT_SHIFT)
        jump = key.down(key.SPACE)
    end

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
    local _, pvy = player:velocity()
    local moving = math.sqrt(vx * vx + vz * vz)
    if moving > 0.1 then
        body_yaw = math.atan(vx, vz)
    end
    player_actor:place(px, py, pz, body_yaw)
    player_actor:move(moving, player:grounded(), pvy)
    player_actor:set_visible(third_person)

    if third_person then
        -- Behind the head, pulled in where a wall would be in the way.
        local cp, sp = math.cos(pitch), math.sin(pitch)
        local dx, dy, dz = -math.sin(yaw) * cp, -sp, -math.cos(yaw) * cp
        local hx, hy, hz = px, py + 1.55, pz
        local hit = physics.raycast(hx, hy, hz, dx, dy, dz, CAMERA_DISTANCE + 0.3)
        local distance = hit and math.max(hit - 0.3, 0.4) or CAMERA_DISTANCE
        eye = { hx + dx * distance, hy + dy * distance + 0.2, hz + dz * distance }
        look = { hx, hy, hz }
    else
        eye = { px, py + EYE_HEIGHT, pz }
        look = { eye[1] + math.sin(yaw) * math.cos(pitch), eye[2] + math.sin(pitch), eye[3] + math.cos(yaw) * math.cos(pitch) }
    end

    for _, npc in ipairs(npcs) do
        npc.saved = (npc.saved or 0) + dt
    end
    for _ = 1, math.min(NPCS_PER_FRAME, #npcs) do
        npc_turn = npc_turn % #npcs + 1
        local npc = npcs[npc_turn]
        update_npc(npc, math.min(npc.saved, 0.1), px, pz)
        npc.saved = 0
    end

    if DEBUG then
        report = (report or 0) + dt
        if report > 0.5 then
            report = 0
            print(string.format("pos %.2f %.2f %.2f grounded=%s", px, py, pz, tostring(player:grounded())))
        end
    end
end

function M.on_key(code)
    if code == key.ESCAPE then
        game.quit()
    elseif code == key.V then
        third_person = not third_person
    end
end

function M.camera(aspect)
    return { eye = eye, target = look, fov = 70, near = 0.05, far = 1500 }
end

function M.on_ui()
    if not spawned then
        ui.window("loading", { centred = true }, function()
            ui.text("Loading the Bistro" .. string.rep(".", math.floor(clock * 2) % 4), 1.5)
            ui.text_disabled("first run cooks ~400 textures; later runs are quick")
        end)
    end
end

M.state = function() return player, spawned end

return M
