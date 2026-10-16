---@meta

---@class text
text = {}

---A colour as `{ r, g, b, a }` (or `{ 1, 2, 3, 4 }`), linear, each 0 .. 1. Alpha defaults to 1.
---@alias TextColor { r?: number, g?: number, b?: number, a?: number }|number[]

---Reads a TrueType or OpenType font. Returns its id for the `font` option of the other functions, or nil and a
---message. The path is relative to the data directory. Font 0 is the engine's own and needs no loading.
---@param path string
---@return integer|nil font
---@return string|nil error
function text.load_font(path) end

---Shows text for this frame only: call it every frame the text should be visible. Glyphs are drawn from a signed
---distance field, so the size is free and the edges stay sharp. `\n` starts a new line.
---@param options { text: string, x?: number, y?: number, size?: number, font?: integer, color?: TextColor }
---  `x` and `y` are the text's top-left, in pixels of the scene (y down). `size` is the em size in pixels (24).
function text.draw(options) end

---Shows text placed in the world for this frame only. It is drawn depth-tested and alpha-blended with the rest of
---the scene. Before `rotation` turns it, it reads left to right along +x with its up on +y, as seen by a viewer looking
---along +z, and the first line's baseline starts at `position`.
---@param options { text: string, position?: number[]|{x:number,y:number,z:number}, rotation?: number[]|{x:number,y:number,z:number}, size?: number, font?: integer, color?: TextColor }
---  `rotation` is in degrees, applied about the world's x axis first, then y, then z. `size` is the em size in world units (1).
function text.draw_world(options) end

---The width and height, in pixels, `str` takes drawn at `size`.
---@param str string
---@param size number
---@param font? integer
---@return number width
---@return number height
function text.measure(str, size, font) end
