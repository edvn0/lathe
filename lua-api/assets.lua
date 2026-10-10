---@meta

---@class assets
assets = {}

---An opaque model handle.
---@class Model

---An opaque material handle. Create with assets.material; release with assets.release_material.
---@class Material

---Starts loading a model asset (streamed; the cube stands in until it is ready). Only from on_populate and later.
---With `collider = true` the model's triangles also become collision geometry once it has loaded (see
---physics.add_mesh_collider). `min_extent` leaves out parts smaller than that many metres, such as wires and clutter.
---@param path string An asset path such as "assets/models/x.glb".
---@param options? { collider?: boolean, min_extent?: number }
---@return Model
function assets.load_model(path, options) end

---The built-in unit cube model.
---@return Model
function assets.cube() end

---Creates a material. Only from on_populate and later.
---@param name string
---@param red number
---@param green number
---@param blue number
---@param roughness? number default 0.6
---@param metallic? number default 0.0
---@return Material
function assets.material(name, red, green, blue, roughness, metallic) end

---@param material Material
function assets.release_material(material) end

---How many model and texture loads are still in flight.
---@return integer
function assets.pending() end
