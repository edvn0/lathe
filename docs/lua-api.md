# Lua API for editors (LuaLS)

`lua-api/` holds `---@meta` stub files describing everything the engine puts in a game script's environment: the
libraries `scene`, `assets`, `camera`, `ui`, `game`, `entity`, `compute`, `key` and `mouse`, the `Entity` and `Effect`
handles with their methods, the option tables (`ParticleOptions`, `UiWindowOptions`, `CameraResult`) and `LatheGame`,
the table of callbacks a script returns. They are never loaded by the engine; they only give completion, hover docs
and type checks. The repo's `.luarc.json` selects Lua 5.4 (what the engine embeds), points `workspace.library` at
`lua-api/` and declares the injected globals.

A script opts in to callback checking with an annotation on the table it returns:

```lua
---@type LatheGame
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
