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
---@param textures? MaterialTextures textures to use in place of the flat defaults
function assets.material(name, red, green, blue, roughness, metallic, textures) end

---Textures for assets.material. The viewport textures ("scene_colour", "scene_depth") cannot be used here.
---@class MaterialTextures
---@field base_colour_texture? Texture
---@field normal_texture? Texture
---@field metallic_roughness_texture? Texture
---@field occlusion_texture? Texture
---@field emissive_texture? Texture

---An image: one registered under a name, or one of the viewport's ("scene_colour", "scene_depth"), which compute
---effects sample.
---@class Texture

---The texture registered under `name`, or nil. "scene_colour" and "scene_depth" give the viewport's images, which
---can be given to a compute effect input (compute.instance) but not to a material.
---@param name string
---@return Texture?
function assets.texture(name) end

---@param material Material
function assets.release_material(material) end

---How many model and texture loads are still in flight.
---@return integer
function assets.pending() end
