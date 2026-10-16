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

## Packaging

The editor's *Package* action ships `lathe-player` (found beside the editor executable) rather than the editor itself;
it falls back to copying the editor when there is no player beside it.

## What `lathe-player` links

`lathe-player` links `lathe::engine` only: neither `engine_app` nor `engine_benchmark` is on its link line. Scene content
shared by a sample game and the benchmark runner (the shadow city) lives in `engine_game`, so no game depends on a module
to its right.
