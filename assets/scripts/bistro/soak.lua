-- Collision soak test for the Bistro: walks, runs and jumps the player in random directions for a while and reports
-- tunnelling (a step whose path a ray shows to cross geometry), falls out of the world and time spent stuck.
--   lathe --game lua --script assets/scripts/bistro/soak.lua --player
local M = require("bistro.main")

local DURATION = 240.0
local HOP_EVERY = 20.0
local clock, turn_timer = 0.0, 0.0
local heading, run, jump_timer = 0.0, false, 0.0
local last = nil
local tunnelled, fell, stuck_time, steps, jumps = 0, 0, 0.0, 0, 0
local min_x, max_x, min_z, max_z = 1e9, -1e9, 1e9, -1e9
local reported = false
local covered, hop_timer, hops = 0, 0.0, 0
local covered_x, covered_z = {}, {}

math.randomseed(12345)

-- Rays at knee, hip and head height along the step; a hit closer than the step means the capsule crossed geometry.
local function crossed(x0, y0, z0, x1, y1, z1)
    local dx, dz = x1 - x0, z1 - z0
    local length = math.sqrt(dx * dx + dz * dz)
    if length < 1e-4 then
        return false
    end
    for _, height in ipairs({ 0.5, 1.0, 1.6 }) do
        local hit = physics.raycast(x0, y0 + height, z0, dx / length, 0, dz / length, length)
        if hit and hit < length - 0.05 then
            return true
        end
    end
    return false
end

function M.drive(dt, player)
    clock = clock + dt
    turn_timer = turn_timer - dt
    jump_timer = jump_timer - dt
    if turn_timer <= 0 then
        heading = math.random() * math.pi * 2
        turn_timer = 1 + math.random() * 3
        run = math.random() < 0.4
    end

    -- Every so often jump to a random street-level spot, so the whole map is covered and not one corner.
    hop_timer = hop_timer - dt
    if hop_timer <= 0 then
        hop_timer = HOP_EVERY
        for _ = 1, 60 do
            local rx, rz = -80 + math.random() * 180, -80 + math.random() * 220
            local distance, _, py = physics.raycast(rx, 2.5, rz, 0, -1, 0, 6)
            if distance and distance > 2.0 and py > -1 and py < 1 then
                player:teleport(rx, py + 0.1, rz)
                last = nil
                hops = hops + 1
                break
            end
        end
    end

    local x, y, z = player:position()
    if physics.raycast(x, y + 1.8, z, 0, 1, 0, 3.5) then
        covered = covered + 1
        covered_x[#covered_x + 1], covered_z[#covered_z + 1] = x, z
    end
    if last then
        steps = steps + 1
        if crossed(last[1], last[2], last[3], x, y, z) then
            tunnelled = tunnelled + 1
        end
        if y < -5 then
            fell = fell + 1
            player:teleport(last[1], last[2] + 1, last[3])
        end
        local moved = math.sqrt((x - last[1]) ^ 2 + (z - last[3]) ^ 2)
        if moved < 0.3 * dt then
            stuck_time = stuck_time + dt
        end
    end
    last = { x, y, z }
    min_x, max_x, min_z, max_z = math.min(min_x, x), math.max(max_x, x), math.min(min_z, z), math.max(max_z, z)

    local jump = false
    if jump_timer <= 0 then
        jump = math.random() < 0.5
        jump_timer = 0.8 + math.random() * 2
        if jump then jumps = jumps + 1 end
    end

    if clock > DURATION and not reported then
        reported = true
        print(string.format("SOAK steps=%d tunnelled=%d fell=%d stuck=%.1fs jumps=%d hops=%d area x[%.0f,%.0f] z[%.0f,%.0f]",
            steps, tunnelled, fell, stuck_time, jumps, hops, min_x, max_x, min_z, max_z))
        print(string.format("SOAK covered (a ceiling within 3.5 m): %d of %d steps", covered, steps))
        game.quit()
    end

    return 1, 0, run, jump, heading
end

return M
