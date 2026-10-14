---@meta

---@class render
render = {}

---@class BloomSettings
---@field enabled boolean
---@field threshold number HDR brightness where bloom starts
---@field knee number softness of that threshold
---@field intensity number 0 .. 1

---The current bloom settings. Only from on_populate and later.
---@return BloomSettings
function render.bloom() end

---Changes the bloom fields given and keeps the rest (intensity is clamped to 0 .. 1). Only from on_populate and later.
---@param options { enabled?: boolean, threshold?: number, knee?: number, intensity?: number }
function render.set_bloom(options) end
