# Framework first, editor second

Lathe is a framework with an editor built on it, not the other way round.

```
engine_runtime   Runtime + run_engine     what a running game needs; startup and the frame loop
engine_app       Application : Runtime    the editor: panels, gizmos, scene files, packaging
engine_benchmark                          drives an Application headless
```

- **`Runtime`** (`include/runtime/runtime.hxx`) owns the renderer, the scene a game is loaded into (`source_scene`), the
  copy it plays in (`runtime_scene`), the `IGame` and the input it receives, terrain, particles, effects and scene
  loading. Every default is the player's behaviour, so a `Runtime` on its own is a complete host.
- **`Application`** (`include/app/application.hxx`) derives from it and overrides the virtual hooks: `on_ui` (the
  dock layout), `on_event` (camera, gizmo and file shortcuts, then `Runtime::on_event`), `update`, `editing_camera`,
  and the `on_play_starting` / `on_play_stopping` / `on_scene_opening` / `on_source_scene_replaced` notifications.
- **`run_engine`** (`include/runtime/engine_main.hxx`) is the shared `main`: command line, data paths, shader and
  resource packs, Vulkan, window callbacks and the frame loop. An executable is an `EngineProfile` passed to it:
  `src/main.cxx` (`lathe`, the editor and benchmark runner) and `src/player_main.cxx` (`lathe-player`, the installed
  game).

## Embedding the engine in your own `main`

Link `lathe::engine` and write a profile; the defaults give you the player:

```cpp
class MyProfile final : public EngineProfile {
public:
    auto name() const -> std::string_view override { return "mygame"; }
    auto description() const -> std::string_view override { return "My game"; }
    auto always_player() const -> bool override { return true; }
    auto create_runtime(VulkanContext &context) -> std::unique_ptr<Runtime> override {
        return std::make_unique<Runtime>(context);
    }
};

auto main(int argc, char **argv) -> int {
    MyProfile profile;
    return run_engine(argc, argv, profile);
}
```

Games are looked up through `game_names()` / `create_game()`, which the executable defines (see
`game/src/main_entry.cxx`).

## Projects

A project is a directory with a `lathe.project` at its root (the keys of `game.toml`: `name`, `title`, `version`,
`game`, `entry`, `scene`; `game` may be left out). `lathe --project PATH` opens one, as does running from inside one.
File > New Project, Open Project and Recent Projects switch in process; Edit > Project Settings edits the manifest.

`Paths` has two content roots. The project's is the data root. The engine's (shaders, fonts, built-in textures and
models) is the engine root, found beside the executable or in a checkout. `data_path` takes the project's file and falls
back to the engine's; `engine_path` is engine-only; `Paths::data_directories` lists both for include paths and
watchers. Editor state lives in `<project>/.lathe/state` (gitignore `.lathe/`); the cache is user-wide.

## CVars

Every command line option and `LATHE_*` environment variable is a `CVar` (`include/core/cvar.hxx`): one typed value
declared at namespace scope, which registers itself with `CVarRegistry::global()`. The runtime's are in
`include/runtime/engine_cvars.hxx` (`cvars::render_scale`, `cvars::vsync`, ...), the benchmark runner's in
`include/benchmark/benchmark_cvars.hxx`.

A value comes from, weakest first: its default, the environment (`LATHE_` + the name in capitals, `LATHE_RENDER_SCALE=2`),
the command line (`--render-scale 2`; `_` in the name is `-`), then a run-time set (the editor's Debug > CVars panel).
Options that start an action (`--bake-shaders`, `--benchmark`, `--scene`) are command line only. `--help` is generated
from the registry.

Read one with `get()`. A setting whose default depends on the run (a benchmark turns vsync off) is read with
`has_value()` or `value_or(fallback)` instead, which tell "asked for" from "left alone". To add one, define it in the
module's `*_cvars.cxx` next to the others (a definition nothing refers to is dropped from a static library, so it must
live in a file its readers already pull in) and declare it in the header.

## Packaging

The editor's *Package* action ships `lathe-player` (found beside the editor executable) rather than the editor itself;
it falls back to copying the editor when there is no player beside it.

## What `lathe-player` links

`lathe-player` links `lathe::engine` only: neither `engine_app` nor `engine_benchmark` is on its link line. Scene content
shared by a sample game and the benchmark runner (the shadow city) lives in `engine_game`, so no game depends on a module
to its right.
