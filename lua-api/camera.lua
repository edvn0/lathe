---@meta

---@class camera
camera = {}

---Where the ray through a normalised device coordinate hits the plane y = plane_y, as x and z; nil if it misses.
---@param ndc_x number
---@param ndc_y number
---@param plane_y number
---@return number? x
---@return number? z
function camera.pick_plane(ndc_x, ndc_y, plane_y) end
