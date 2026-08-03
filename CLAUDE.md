# Building

Build via `./compile.sh` -- it runs CMake/Ninja inside a Docker container
(image `cross-build:latest`), bind-mounting the repo at `/workspace` and
`~/.cache/CPM` at `/cpm-cache`. Do not invoke `cmake`/`ninja` directly on the
host against `build/` -- paths there are container-relative and won't
resolve.

```
TARGET=windows-mingw ./compile.sh --configure   # default target
TARGET=linux-native  ./compile.sh --configure
./compile.sh --build
./compile.sh --rebuild   # clean + configure + build
./compile.sh --clean
./compile.sh --shell     # interactive shell inside the build container
```

`TARGET` selects `windows-mingw` (cross-compile via mingw-w64) or
`linux-native`; build output goes to `build/<target>-<build_type>` (build
type from `CMAKE_BUILD_TYPE`, default `Debug`).
