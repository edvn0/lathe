---@type Behaviour
local M = {}

M.properties = {
    { name = "speed", type = "number", default = 90, min = -720, max = 720, tooltip = "Degrees per second" },
    { name = "axis", type = "choice", options = { "x", "y", "z" }, default = "y" },
    { name = "extra", type = "integer", default = 42 },
    { name = "bob_height", type = "number", default = 0, min = 0, max = 10 },
    { name = "tint", type = "colour", default = { 1, 1, 1 } },
    { name = "target", type = "entity", tooltip = "Optional entity to follow" },
}

function M:on_start()
    self.angle = 0
    self.time = 0
    self.base_x, self.base_y, self.base_z = self.entity:position()
end

function M:on_update(dt)
    if not self.said then self.said = true print("v2 extra=" .. tostring(self.extra) .. " speed=" .. self.speed) end
    self.angle = self.angle + math.rad(self.speed) * dt
    self.time = self.time + dt

    local x, y, z = 0, 0, 0
    if self.axis == "x" then x = self.angle elseif self.axis == "y" then y = self.angle else z = self.angle end
    self.entity:set_euler(x, y, z)

    if self.bob_height > 0 then
        self.entity:set_position(self.base_x, self.base_y + math.sin(self.time * 2) * self.bob_height, self.base_z)
    end
end

return M
