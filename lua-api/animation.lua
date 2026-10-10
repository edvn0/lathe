---@meta

---Skinned, animated characters. Load a skinned model with assets.load_model, make a rig from it, spawn an actor per
---character, then every update place each actor and say how it is moving; the engine plays the matching clip (idle,
---walk, run, jump) and draws it. Clips are found by the names Idle, Walk, Run, Jump and Death; a model without them
---loops its first clip as the idle.
---@class animation
animation = {}

---@class RigOptions
---@field height? number metres the model is scaled to (default 1.75)
---@field model_height? number the model's own height in its units (default 5.535, animated_human.glb)
---@field model_feet_y? number where the model's feet are on its y axis (default -0.015)
---@field yaw_offset? number radians that turn the model to face +Z (default pi)
---@field walk_speed? number metres per second the walk clip covers, so feet do not slide (default 1.4)
---@field run_speed? number the same for the run clip (default 4.5)

---A skinned model prepared for drawing characters. One rig per model.
---@class Rig
local Rig = {}

---True once the model has loaded and has a skeleton and animation.
---@return boolean
function Rig:ready() end
---Adds a character. Actors are drawn once the rig is ready.
---@return Actor
function Rig:spawn() end

---One animated character.
---@class Actor
local Actor = {}

---Where its feet are and which way it faces, in radians about +Y (0 faces +Z).
---@param x number
---@param y number
---@param z number
---@param yaw number
function Actor:place(x, y, z, yaw) end
---How it is moving, which chooses idle, walk, run or a jump pose.
---@param speed number horizontal speed in metres per second
---@param grounded? boolean (default true)
---@param vertical_velocity? number metres per second, positive when rising (default 0)
function Actor:move(speed, grounded, vertical_velocity) end
---@param visible boolean
function Actor:set_visible(visible) end
---The animation state: "idle", "walk", "run", "jump_rise", "jump_apex", "jump_fall", "lying_down", "prone" or
---"getting_up"; empty before the first update.
---@return string
function Actor:state() end

---Makes a rig for a skinned model. Only from on_populate and later.
---@param model Model
---@param options? RigOptions
---@return Rig
function animation.rig(model, options) end
