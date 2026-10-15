# Lua API for editors (LuaLS)

`lua-api/` holds `---@meta` stub files describing everything the engine puts in a game script's environment: the
libraries `scene`, `assets`, `camera`, `ui`, `game`, `entity`, `compute`, `render`, `key` and `mouse`, the `Entity` and `Effect`
handles with their methods, the option tables (`ParticleOptions`, `UiWindowOptions`, `CameraResult`) and `GameScript`,
the table of callbacks a script returns. They are never loaded by the engine; they only give completion, hover docs
and type checks. The repo's `.luarc.json` selects Lua 5.4 (what the engine embeds), points `workspace.library` at
`lua-api/` and declares the injected globals.

A script opts in to callback checking with an annotation on the table it returns:

```lua
---@type GameScript
local M = {}
function M.on_update(dt) end
return M
```

## Enabling it

- **VS Code:** install the "Lua" extension (sumneko.lua) and open the repo root. `.luarc.json` is picked up.
- **Neovim:** with `nvim-lspconfig`, `require("lspconfig").lua_ls.setup{}`. It reads `.luarc.json` from the root.
- Any other LuaLS client works the same way.

## Keeping the stubs in sync

`test/lua_api_stubs_test.cxx` opens the real Lua state with the engine libraries registered and compares every
function and constant (and every `Entity` / `Effect` method) with the stubs, both ways. Add a function to
`lua_game_api.cxx` and the test fails until its stub exists; a stub with no registration fails too. The test matches
`function lib.name(`, `function Class:name(` and `lib.NAME = value` lines, so keep that shape. Parameter types,
ranges and defaults are written by hand from the C++ (`lua_game_api.cxx`, `lua_particles.cxx`, `effect_manifest.hxx`)
and are not checked.

Not covered: the native modules `native.chess` and `native.net` (opened with `require`) have no stubs.

## Lights, environment and bloom

`e:add_light{ type = "point" | "spot", colour, intensity, range, inner_cone, outer_cone, casts_shadows }` puts a light on
an entity and `e:aim(x, y, z)` turns it (a spot light points along the entity's local -Y). `scene.set_environment{...}`
changes the sky, ambient light, exposure and sun; `render.bloom()` / `render.set_bloom{...}` read and change bloom.
`assets.sphere()` and `assets.capsule()` give the built-in models, and `assets.material` takes `emissive`,
`emissive_strength` and `double_sided` in its options table. `assets/scripts/lights/main.lua` (`--game=lights`) uses
all of them.

## Cameras

`GameScript.camera(aspect)` returns `eye`, `target`, `fov`, `near` and `far`. Add `ortho_height` for an orthographic
view that many world units tall (the width follows the aspect ratio), and `up` for a camera that looks along the
y axis (a top-down view needs `up = {0, 0, 1}`). The renderer's perspective-based fits, such as sun shadow cascades,
assume a perspective camera, so use an unlit scene with an orthographic view. `assets/scripts/snake/` does.

## Behaviours

A file in `assets/scripts/behaviours/` that returns a class table is a behaviour. Attach it in the editor
(Add Component > Behaviour) or from Lua with `entity:add_behaviour(name, { speed = 5 })`. While the game plays each
entity gets an instance, `self`, holding `self.entity` and a field per property; `on_start(self)`,
`on_update(self, dt)` and `on_destroy(self)` are called on it. An error in one instance stops only that instance.

```lua
return {
    properties = {
        { name = "speed", type = "number", default = 90, min = -720, max = 720 },
        { name = "skin", type = "texture" },
    },
    on_update = function(self, dt) self.entity:set_euler(0, self.angle or 0, 0) end,
}
```

`properties` is the reflection: the inspector draws a widget for each, scenes save the values an entity changed (the
rest follow the script's defaults), and the script reads them from `self`. Types: `boolean`, `integer`, `number`
(with `min`/`max`/`step`), `string`, `vec3`, `colour`, `choice` (with `options`), `entity`, and the assets `model`,
`material` and `texture`, which arrive as `Model`, `Material` and `Texture` values or nil. `scene_colour` and
`scene_depth` are textures too: the viewport's images, which `compute.instance` effects take as inputs
(`assets.texture("scene_colour")`); materials cannot use them. `assets.material` takes a table of textures as a 7th
argument. Behaviours reload live: edit the file while the editor or the game runs and the change is picked up at once (a file
watcher, as for shaders). Running instances switch to the new code, keep the fields the script set, gain fields for new properties
and lose those that were removed; the inspector shows the new property list, and an instance stopped by an error runs
again. A file that doesn't load keeps the old code running and shows the error in the inspector. The Reload button
forces it.
