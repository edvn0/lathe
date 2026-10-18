-- Amazon Lumberyard Bistro (CC BY 4.0), a large test scene, with eighty shadow-casting point lights strewn through it.
-- With a textured copy on disk (textured.lua points at NVIDIA ORCA's BistroExterior.glb, 2.8 million triangles) that is used.
-- Otherwise the scene is fetched with the engine's own http client from zeux/niagara_bistro (4.2 million triangles
-- instanced, 2,900 mesh instances), pinned to a commit and to a hash per file, into the download cache under bistro/ so
-- the glTF finds its buffers next to it. That copy renders white: its albedo lives in the deprecated
-- KHR_materials_pbrSpecularGlossiness extension (the glTF parser is built without it) and its PNG textures are not in
-- the repo, so it only stresses geometry, culling and lights.
-- Started with --game=lua --script=assets/scripts/bistro_http/main.lua (download) or .../textured.lua (local copy), with
-- --player; add --max-quality to turn every feature up. The walkable version of this scene is assets/scripts/bistro/main.lua.
-- Keys: 1-3 camera shot (orbit, walk through, overview), Esc quits.

---@type GameScript
local M = {}

local BASE = "https://raw.githubusercontent.com/zeux/niagara_bistro/2bdb6a410f8ebd475d3737e7c8e038ed2b00b02e/"
local FILES = {
    { name = "bistro/bistro.gltf", url = BASE .. "bistro.gltf",
      sha256 = "c0972d6cd032c1aecb1ad609291a60d124975cd405097212c29de806851de2bb" },
    { name = "bistro/bistro.bin", url = BASE .. "bistro.bin",
      sha256 = "5fb11a1d61aeb2537197aa97833ff950638c9d42451ee4fafbc9550f03918faa" },
    { name = "bistro/bistro-anim.bin", url = BASE .. "bistro-anim.bin",
      sha256 = "285d49eeaa58fb453c34bf7c6ffbeae65c6d77e8b706cb1a00db1736bca31b78" },
}

-- A textured copy on disk (NVIDIA ORCA's BistroExterior.glb, also what assets/scripts/bistro/main.lua walks around) is
-- used when it is there: the repo copy below has no usable textures. Bounds come from each glTF's node transforms and
-- mesh extents.
local LOCAL_MODEL = nil -- set by configure
local LOCAL_BOUNDS = { min = { -62.5, -7.6, -96.5 }, max = { 111.3, 43.5, 87.9 } }
local REMOTE_BOUNDS = { min = { -52.4, -4.7, -46.9 }, max = { 58.7, 27.2, 72.4 } }

local MIN, MAX = REMOTE_BOUNDS.min, REMOTE_BOUNDS.max
local CENTRE = { 0, 5, 0 }
local use_local = false

local LIGHT_COLUMNS, LIGHT_ROWS = 10, 8

local ORBIT_SPEED = 0.12 -- radians per second
local EYE_HEIGHT = 1.6
local WALK_SPEED = 2.8 -- m/s
local LOOKAHEAD = 4.5 -- turn when a wall is closer than this
local SIDE_CLEARANCE = 1.3

local orbit_radius, orbit_height = 150, 65
local bistro_entity, bistro_model
local walk -- the street walker's state, made once the collision geometry is ready

local requests = {}
local state = "downloading" -- downloading, loading, ready, failed
local lights = {}
local time = 0
local shot = 0
local eye = { 70, 25, 13 }
local look_at = { CENTRE[1], CENTRE[2], CENTRE[3] }

-- A repeatable value in [0, 1) from an integer, so the light colours do not change between runs.
local function noise(i)
    local x = math.sin(i * 12.9898) * 43758.5453
    return x - math.floor(x)
end

local function add_lights()
    local count = 0
    for column = 0, LIGHT_COLUMNS - 1 do
        for row = 0, LIGHT_ROWS - 1 do
            count = count + 1
            local tx = (column + 0.5) / LIGHT_COLUMNS
            local tz = (row + 0.5) / LIGHT_ROWS
            local e = scene.spawn("bistro_light_" .. count)
            e:set_position(
                MIN[1] + tx * (MAX[1] - MIN[1]) + (noise(count) - 0.5) * 4,
                3 + noise(count + 1000) * 3,
                MIN[3] + tz * (MAX[3] - MIN[3]) + (noise(count + 2000) - 0.5) * 4
            )
            local warm = noise(count + 3000)
            e:add_light({
                type = "point",
                colour = { 1, 0.65 + 0.3 * warm, 0.35 + 0.55 * warm },
                intensity = 30 + 30 * noise(count + 4000),
                range = 11 + 5 * noise(count + 5000),
                casts_shadows = true,
            })
            lights[#lights + 1] = e
        end
    end
end

-- Uses a textured copy on disk instead of the download: configure({ local_model = "/path/BistroExterior.glb" }).
function M.configure(options)
    LOCAL_MODEL = options.local_model
end

function M.on_populate()
    scene.clear()
    lights = {}
    time = 0

    scene.set_environment({ sky = "procedural", skybox = true, ambient = 0.05, exposure_ev = 0, sun_intensity = 3,
                            sun_elevation = 35 })

    use_local = LOCAL_MODEL ~= nil
    local bounds = use_local and LOCAL_BOUNDS or REMOTE_BOUNDS
    MIN, MAX = bounds.min, bounds.max
    CENTRE = { (MIN[1] + MAX[1]) / 2, 5, (MIN[3] + MAX[3]) / 2 }
    -- Far enough out and high enough that the orbit never enters a building: the walls are single-sided, so a camera
    -- inside one sees through it.
    orbit_radius = 0.5 * math.sqrt((MAX[1] - MIN[1]) ^ 2 + (MAX[3] - MIN[3]) ^ 2) + 25
    orbit_height = MAX[2] + 20
    walk = nil
    eye = { CENTRE[1] + orbit_radius, orbit_height, CENTRE[3] }
    look_at = { CENTRE[1], CENTRE[2], CENTRE[3] }
    add_lights()

    if use_local then
        print("[bistro] using the textured copy at " .. LOCAL_MODEL .. " (the first run cooks its textures)")
        return
    end
    for i, file in ipairs(FILES) do
        requests[i] = http.download(file.url, { sha256 = file.sha256, name = file.name })
    end
    print("[bistro] downloading " .. #FILES .. " files (about 100 MB), then loading the model; it has no usable textures")
end

-- The ground height under (x, z), looking down from a little above `from_y`, or nil with nothing there.
local function ground_height(x, from_y, z)
    local distance, _, hit_y = physics.raycast(x, from_y + 2.5, z, 0, -1, 0, 8)
    return distance and hit_y or nil
end

-- How far a ray from the eye goes along `heading` before it meets geometry.
local function clear_distance(x, y, z, heading, limit)
    return physics.raycast(x, y, z, math.sin(heading), 0, math.cos(heading), limit) or limit
end

-- A walk along the streets: street level is found by rays through a spiral around the middle (as in
-- assets/scripts/bistro/main.lua), then the camera heads forward, turning toward the more open side whenever a wall is
-- near, so it never passes through one.
local function start_walk()
    for radius = 0, 40, 2 do
        for step = 0, 7 do
            local a = step * math.pi / 4
            local x, z = CENTRE[1] + radius * math.cos(a), CENTRE[3] + radius * math.sin(a)
            local distance, _, hit_y = physics.raycast(x, 2.5, z, 0, -1, 0, 6)
            if distance and distance > 2.0 and hit_y > -1 and hit_y < 1 then
                walk = { x = x, z = z, y = hit_y, heading = 0, turn_to = nil }
                return
            end
        end
    end
end

local function update_walk(dt)
    if not bistro_model or not physics.ready(bistro_model) then
        return -- the collision geometry is still being built; keep the last view
    end
    if not walk then
        start_walk()
        if not walk then
            return
        end
    end

    local eye_y = walk.y + EYE_HEIGHT
    local ahead = clear_distance(walk.x, eye_y, walk.z, walk.heading, LOOKAHEAD)
    local left = clear_distance(walk.x, eye_y, walk.z, walk.heading - 1.0, 6)
    local right = clear_distance(walk.x, eye_y, walk.z, walk.heading + 1.0, 6)

    if ahead < LOOKAHEAD and not walk.turn_to then
        local turn = (left > right and -1 or 1) * (0.7 + 0.6 * noise(math.floor(time) + 7))
        walk.turn_to = walk.heading + turn
    end

    if walk.turn_to then
        local difference = walk.turn_to - walk.heading
        local step = math.min(math.abs(difference), 1.8 * dt)
        walk.heading = walk.heading + (difference >= 0 and step or -step)
        if math.abs(walk.turn_to - walk.heading) < 1e-3 then
            walk.turn_to = nil -- if the way is still blocked, the next frame picks another turn
        end
    else
        -- Ease away from a wall alongside.
        if left < SIDE_CLEARANCE then
            walk.heading = walk.heading + 0.8 * dt
        elseif right < SIDE_CLEARANCE then
            walk.heading = walk.heading - 0.8 * dt
        end
        local x = walk.x + math.sin(walk.heading) * WALK_SPEED * dt
        local z = walk.z + math.cos(walk.heading) * WALK_SPEED * dt
        local y = ground_height(x, walk.y, z)
        if y and math.abs(y - walk.y) < 0.6 then
            walk.x, walk.z, walk.y = x, z, y
        else
            walk.turn_to = walk.heading + 1.2 -- no floor or a drop ahead
        end
    end

    eye = { walk.x, walk.y + EYE_HEIGHT, walk.z }
    look_at = { walk.x + math.sin(walk.heading) * 6, walk.y + EYE_HEIGHT - 0.2, walk.z + math.cos(walk.heading) * 6 }
end

function M.on_update(dt)
    time = time + dt

    if use_local and state == "downloading" then
        bistro_entity = scene.spawn("bistro")
        bistro_model = assets.load_model(LOCAL_MODEL, { collider = true, double_sided = true })
        bistro_entity:set_model(bistro_model)
        physics.add_mesh_collider(bistro_entity, bistro_model)
        state = "ready"
    elseif state == "downloading" then
        local finished = 0
        for i, request in ipairs(requests) do
            if request:done() then
                finished = finished + 1
                if not request:ok() then
                    print("[bistro] " .. FILES[i].name .. " failed: " .. tostring(request:error()))
                    state = "failed"
                    return
                end
            end
        end
        if finished == #requests then
            state = "loading"
            print("[bistro] downloaded, loading the model")
            bistro_model = assets.load_model(requests[1], { collider = true, double_sided = true })
            bistro_entity = scene.spawn("bistro")
            bistro_entity:set_model(bistro_model)
            physics.add_mesh_collider(bistro_entity, bistro_model)
            state = "ready"
        end
    end

    local angle = time * ORBIT_SPEED
    if shot == 0 then
        eye = { CENTRE[1] + math.cos(angle) * orbit_radius, orbit_height, CENTRE[3] + math.sin(angle) * orbit_radius }
        look_at = { CENTRE[1], CENTRE[2], CENTRE[3] }
    elseif shot == 1 then
        update_walk(dt)
    else
        eye = { MIN[1] - 10, MAX[2] + 25, MIN[3] - 10 }
        look_at = { CENTRE[1], CENTRE[2], CENTRE[3] }
    end
end

function M.on_key(code)
    if code == key.ESCAPE then
        game.quit()
    elseif code == key.NUM_1 then
        shot = 0
    elseif code == key.NUM_2 then
        shot = 1
    elseif code == key.NUM_3 then
        shot = 2
    end
end

function M.camera(aspect)
    return { eye = eye, target = look_at, fov = 60, near = 0.2, far = 600 }
end

return M
