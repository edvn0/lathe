---@meta

---An entity in the scene. It becomes invalid once destroyed; using it then raises an error.
---@class Entity
local Entity = {}

---@class entity
entity = {}

---@return boolean
function Entity:valid() end
function Entity:destroy() end
---@return string
function Entity:name() end
---@param x number
---@param y number
---@param z number
function Entity:set_position(x, y, z) end
---@return number x
---@return number y
---@return number z
function Entity:position() end
---Rotation from Euler angles in radians.
---@param x number
---@param y number
---@param z number
function Entity:set_euler(x, y, z) end
---Scale; y and z default to x.
---@param x number
---@param y? number
---@param z? number
function Entity:set_scale(x, y, z) end
---@param model Model
function Entity:set_model(model) end
---@param material Material
function Entity:set_material(material) end
---@param outlined boolean
function Entity:set_outlined(outlined) end
---Adds a particle emitter. Raises an error if the entity has one (use set_particles) or a field is invalid.
---@param options ParticleOptions
function Entity:add_particles(options) end
---Changes the fields given and keeps the rest. Raises an error if there is no emitter or a field is invalid.
---@param options ParticleOptions
function Entity:set_particles(options) end
function Entity:remove_particles() end
---Gives the entity a physics body, simulated while the game plays. Raises an error on an unknown shape or a
---non-positive size.
---@param options? BodyOptions
function Entity:add_body(options) end
---Sets the body's horizontal velocity and keeps its vertical speed. Only while playing.
---@param x number
---@param y number
---@param z number
function Entity:set_velocity(x, y, z) end
---Pushes the body. Only while playing.
---@param x number
---@param y number
---@param z number
function Entity:apply_impulse(x, y, z) end

---Puts a behaviour (assets/scripts/behaviours/<name>.lua) on the entity, with property values that differ from
---the script's defaults. It starts running while the game plays. Raises an error for an unknown script, an unknown
---property or a value of the wrong type.
---@param name string
---@param properties? table<string, any>
function Entity:add_behaviour(name, properties) end
---The running instance of a behaviour on this entity (the `self` its callbacks get), or nil: while the game
---is not playing, or when the entity has no such script.
---@param name string
---@return table?
function Entity:behaviour(name) end
---Takes the behaviour off; its on_destroy runs at the next update.
function Entity:remove_behaviour() end

---entity.add_particles(e, options) is e:add_particles(options).
---@param e Entity
---@param options ParticleOptions
function entity.add_particles(e, options) end
---@param e Entity
---@param options ParticleOptions
function entity.set_particles(e, options) end
---@param e Entity
function entity.remove_particles(e) end

---Particle emitter fields (src/scripting/lua_particles.cxx). Out-of-range or non-finite numbers, unknown fields and
---wrong types raise an error and change nothing.
---@class ParticleOptions
---@field count? integer particle slots, 1 .. 1048576 (default 1024)
---@field rate? number particles per second, 0 .. 1e6 (default 256)
---@field lifetime? number seconds, 0.01 .. 3600 (default 2)
---@field gravity? number[] {x, y, z}, each -1e4 .. 1e4 (default {0, -9.81, 0})
---@field shape? ParticleShape (default "cone")
---@field shape_size? number sphere radius or box half extent, 0 .. 1e4 (default 0.5)
---@field cone_degrees? number spread around +Y for "box" and "cone", 0 .. 180 (default 25)
---@field speed? number initial speed, -1e4 .. 1e4 (default 3)
---@field speed_variance? number fraction of speed either way, 0 .. 1 (default 0.25)
---@field size_start? number world units at birth, 0 .. 1e3 (default 0.1)
---@field size_end? number world units at death, 0 .. 1e3 (default 0.02)
---@field colour_start? number[] {r, g, b [, a]}, components 0 .. 1000, alpha 0 .. 1 (default {1, 0.6, 0.2, 1})
---@field colour_end? number[] {r, g, b [, a]} (default {1, 0.1, 0, 0}); alpha fades the particle over its life
---@field emitting? boolean spawn new particles (default true)
---@field material? Material tints the particles (base colour times, emissive plus)

---@alias ParticleShape
---| "point" # every particle starts at the entity and flies off in a random direction
---| "sphere" # starts inside a ball of shape_size and flies outwards
---| "box" # starts inside a cube of half extent shape_size and flies along +Y within cone_degrees
---| "cone" # starts at the entity and flies along +Y within cone_degrees

---@alias BodyShape "box"|"sphere"|"capsule"

---@class BodyOptions
---@field shape? BodyShape (default "box")
---@field half_extents? number[] {x, y, z} of a box (default {0.5, 0.5, 0.5})
---@field radius? number of a sphere or capsule (default 0.5 and 0.4)
---@field height? number of a capsule
---@field mass? number (default 1)
---@field restitution? number bounciness (default 0.4)
---@field static? boolean never moves (default false)
---@field lock_rotation? boolean stays upright (default false; true for capsules)

---What a behaviour file returns (assets/scripts/behaviours/<name>.lua). `self` in the callbacks is the instance:
---`self.entity` and one field per property, plus anything the script stores. All callbacks are optional.
---@class Behaviour
---@field properties? BehaviourProperty[]|table<string, BehaviourProperty> editable values; the list form keeps its order
local Behaviour = {}

---Once, before the first on_update of the instance.
---@param self table
function Behaviour.on_start(self) end
---@param self table
---@param dt number seconds since the last frame
function Behaviour.on_update(self, dt) end
---When the entity or its script goes away, or the game stops.
---@param self table
function Behaviour.on_destroy(self) end

---@class BehaviourProperty
---@field name? string required in the list form
---@field type BehaviourPropertyType
---@field default? any boolean, number, string, {x, y, z} or {r, g, b [, a]}; asset types take a registered name
---@field min? number number, integer and vec3 only
---@field max? number
---@field step? number inspector drag speed
---@field options? string[] required for "choice"
---@field tooltip? string

---@alias BehaviourPropertyType
---| "boolean"
---| "integer"
---| "number"
---| "string"
---| "vec3" # a Vec3: {x, y, z}
---| "colour" # {r, g, b, a}; also spelled "color"
---| "choice" # one of `options`, a string
---| "entity" # an Entity or nil
---| "model" # a Model (by registered name) or nil
---| "material" # a Material or nil
---| "texture" # a Texture or nil; "scene_colour" and "scene_depth" are the viewport
