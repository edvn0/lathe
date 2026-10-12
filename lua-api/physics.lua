---@meta

---Physics for games: a character controller, collision geometry from models, and ray casts. Everything except
---add_mesh_collider and ready needs the game to be playing (in on_update and later, not on_populate).
---@class physics
physics = {}

---@class CharacterOptions
---@field radius? number capsule radius in metres (default 0.35)
---@field height? number capsule height in metres, more than twice the radius (default 1.8)
---@field gravity? number m/s^2 (default 25)
---@field jump_height? number metres (default 1.2)
---@field step_height? number the highest step it climbs without jumping (default 0.3)
---@field max_slope? number steepest walkable slope in degrees (default 50)
---@field ground_accel? number (default 50)
---@field ground_decel? number (default 60)
---@field air_accel? number (default 20)

---A capsule that walks: it slides along geometry, climbs steps and slopes, falls and jumps. Its position is at its feet.
---@class Character
local Character = {}

---Advances the character by `dt`, moving at the given world-space velocity in metres per second.
---@param dt number seconds
---@param move_x number
---@param move_z number
---@param jump_pressed boolean jump on this update
---@param jump_held boolean the jump key is still down (releasing it early cuts the jump short)
function Character:update(dt, move_x, move_z, jump_pressed, jump_held) end
---@return number x
---@return number y
---@return number z
function Character:position() end
---@return number x
---@return number y
---@return number z
function Character:velocity() end
---@return boolean
function Character:grounded() end
---Moves the character without sweeping and clears its velocity.
---@param x number
---@param y number
---@param z number
function Character:teleport(x, y, z) end

---Creates a character standing at (x, y, z). Only while playing.
---@param x number
---@param y number
---@param z number
---@param options? CharacterOptions
---@return Character
function physics.character(x, y, z, options) end

---The first thing a ray hits, or nil. Only while playing.
---@param ox number origin
---@param oy number
---@param oz number
---@param dx number direction
---@param dy number
---@param dz number
---@param max_distance? number (default 1000)
---@return number? distance
---@return number? px
---@return number? py
---@return number? pz
---@return number? nx
---@return number? ny
---@return number? nz
function physics.raycast(ox, oy, oz, dx, dy, dz, max_distance) end

---Makes the entity's position the origin of the model's triangles as static collision geometry. The model must have
---been loaded with `{ collider = true }`; collision begins once its triangles are built (see physics.ready).
---@param e Entity
---@param model Model
function physics.add_mesh_collider(e, model) end

---True once the model's mesh collider is built.
---@param model Model
---@return boolean
function physics.ready(model) end
