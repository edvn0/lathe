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

---@class ColourGradeSettings
---@field lut string|nil name of the active colour LUT, nil when grading is off
---@field intensity number 0 .. 1, how far the LUT is blended in
---@field luts string[] every LUT that can be named: cinematic, vintage and noir, then any .cube file loaded

---The current colour grading settings. Only from on_populate and later.
---@return ColourGradeSettings
function render.grade() end

---Changes the colour grading fields given and keeps the rest. `lut` is a built-in name ("cinematic", "vintage",
---"noir"), the data path of a 3D .cube file, or false to turn grading off. Returns nil and a message when the LUT is
---unknown or the file cannot be read. Only from on_populate and later.
---@param options { lut?: string|false, intensity?: number }
---@return nil, string?
function render.set_grade(options) end
