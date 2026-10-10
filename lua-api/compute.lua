---@meta

---@class compute
compute = {}

---A loaded effect manifest and its shader. See docs/lua-api.md and rendering/effect_manifest.hxx.
---@class EffectShader

---An opaque run of floats only effects can bind. Freed when nothing refers to it.
---@class EffectBuffer

---An instance of an effect shader with its own values.
---@class Effect
local Effect = {}

---A value for an effect field: a number, a list of one to four numbers, an image source name or a buffer.
---@alias EffectValue number|number[]|"scene_colour"|"scene_depth"|EffectBuffer

---Reads the manifest at `path` (a .json file under assets/) and registers its shader.
---@param path string
---@return EffectShader
function compute.load(path) end

---Reads the manifest again. A manifest that does not parse leaves the old one in place.
---@param shader EffectShader
function compute.reload(shader) end

---Makes an effect, optionally setting fields by name. Raises an error for an unknown name or a value of the
---wrong type or range, and then makes nothing.
---@param shader EffectShader
---@param values? table<string, EffectValue>
---@return Effect
function compute.instance(shader, values) end

---A buffer of `count` floats, 1 .. 16777216, zero-filled.
---@param count integer
---@return EffectBuffer
function compute.buffer(count) end

---Sets a param (number or list), an image input ("scene_colour" or "scene_depth") or a buffer binding.
---@param name string
---@param value EffectValue
function Effect:set(name, value) end

---Why the engine last dropped this effect from a frame, or nil.
---@return string?
function Effect:problem() end
