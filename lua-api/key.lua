---@meta

---GLFW key codes for on_key.
---@class key
key = {}
---@type integer
key.ENTER = 257
---@type integer
key.SPACE = 32
---@type integer
key.ESCAPE = 256
---@type integer
key.BACKSPACE = 259
---@type integer
key.TAB = 258
---@type integer
key.UP = 265
---@type integer
key.DOWN = 264
---@type integer
key.LEFT = 263
---@type integer
key.RIGHT = 262
---@type integer
key.A = 65
---@type integer
key.B = 66
---@type integer
key.C = 67
---@type integer
key.D = 68
---@type integer
key.E = 69
---@type integer
key.F = 70
---@type integer
key.G = 71
---@type integer
key.H = 72
---@type integer
key.I = 73
---@type integer
key.J = 74
---@type integer
key.K = 75
---@type integer
key.L = 76
---@type integer
key.M = 77
---@type integer
key.N = 78
---@type integer
key.O = 79
---@type integer
key.P = 80
---@type integer
key.Q = 81
---@type integer
key.R = 82
---@type integer
key.S = 83
---@type integer
key.T = 84
---@type integer
key.U = 85
---@type integer
key.V = 86
---@type integer
key.W = 87
---@type integer
key.X = 88
---@type integer
key.Y = 89
---@type integer
key.Z = 90
---@type integer
key.F1 = 290
---@type integer
key.F2 = 291
---@type integer
key.F3 = 292
---@type integer
key.LEFT_SHIFT = 340
---@type integer
key.LEFT_CONTROL = 341
---True while the key is held down. Only keys the game receives while playing are tracked.
---@param code integer one of the key constants
---@return boolean
function key.down(code) end
