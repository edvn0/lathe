# Git

Branch names are `<type>/<name>`, where type is one of `feat`, `chore`, `fix`, `docs`, `refactor`, `test` (e.g.
`feat/native-model-importers`, `fix/shadow-cascade-flicker`). This applies to every branch, including ones created by
Claude Code sessions: use a name like this instead of the session's default `claude/...` branch.

# Building

Build via `cargo xtask` (alias in `.cargo/config.toml`, source in
`tools/xtask`) -- it runs CMake/Ninja inside a Docker container (image
`cross-build:latest`), bind-mounting the repo and `~/.cache/CPM` at their host
paths. Do not invoke `cmake`/`ninja` directly on the host against `build/` --
the configuration was produced inside the container and won't resolve.

```
cargo xtask configure
cargo xtask build
cargo xtask rebuild     # clean + configure + build
cargo xtask clean
cargo xtask shell       # interactive shell inside the build container
cargo xtask test        # build lathe-tests, run CTest (extra args after --)
cargo xtask tidy        # build, then clang-tidy (.clang-tidy); the nightly tidy workflow fails on any finding
cargo xtask profile     # run the executable under perf/callgrind on the host
```

Configuration is via environment variables (`cargo xtask --help` lists them
all), e.g. `CMAKE_BUILD_TYPE` (default `Debug`), `LINKER=mold`, `SANITIZE=1`,
`WERROR=1`. Build output goes to `build/<build_type lowercased>`.

# Module layering

Each `engine_<module>` library is `src/<module>/` plus `include/<module>/`, and may include only modules to its left
(the order is the one in `CMakeLists.txt`):

```
core < gpu < assets < animation < physics < scene < terrain < rendering < serialisation < scripting
     < game < luagame < runtime < app < benchmark
```

`net`, `gameserver` and `gameclient` depend on `core` alone (`luagame` uses `net` for the scripts' `http` library). `game` is the `IGame` interface; `luagame` is the Lua
implementation of it and the Lua API; `runtime` is `Runtime` (everything a running game needs: renderer, scenes, the
`IGame` lifecycle and input, scene loading) plus `run_engine`, the startup and frame loop both executables share; `app` is
the editor, a `Runtime` subclass (`Application`) with panels, gizmos and scene files on top; `benchmark` runs the editor
headless. `lathe` is the editor and `lathe-player` is the installed game, a bare `Runtime` with no editor; each is a thin
`EngineProfile` passed to `run_engine`. `lathe::engine` links the framework without `app` and `benchmark`, `lathe::core`
links everything. The sample games in
`game/` are linked into the executable only and may use any module. Do not add an include that points rightwards.
