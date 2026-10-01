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
cargo xtask tidy        # build, then clang-tidy (.clang-tidy); CI fails on any finding
cargo xtask profile     # run the executable under perf/callgrind on the host
```

Configuration is via environment variables (`cargo xtask --help` lists them
all), e.g. `CMAKE_BUILD_TYPE` (default `Debug`), `LINKER=mold`, `SANITIZE=1`,
`WERROR=1`. Build output goes to `build/<build_type lowercased>`.
