# Perf benchmark

`--benchmark` turns the app into a repeatable GPU benchmark. It is run by
hand, on real hardware: CI runners have no GPU, and lavapipe's timings say
little about how passes compare on one.

## What a run does

- **Scene**: the game's normal editor scene, with every procedural RNG
  seeded from `--seed` (`core/random.hxx`), so grass and props land in the
  same places every run.
- **Camera**: parked on the first keyframe for warmup (at least
  `--benchmark-warmup` frames, and until texture + terrain streaming is idle,
  capped by `--benchmark-max-warmup`). It then flies one closed lap of a
  uniform Catmull-Rom spline through the game's
  `IGame::benchmark_camera_path()`: 15 keyframes in `BasicGame`, mixing
  grass-level shots, walls and canopies up close, and high overviews.
  Position and look-at target are splined independently
  (`scene/camera_path.hxx`).
- **Resolution**: the default editor layout (a saved `imgui.ini` is
  ignored), since the Viewport panel's size is the render resolution.
- **Time**: fixed 1/60 s per frame, so wind sway and enemy motion are the
  same at frame N in every run, however slow the device is.
- **Output**: per-stage GPU timings (the same timestamps as the Frame
  timings panel) for every measured frame, summarised as mean / median /
  p95 / min / max in JSON, plus the full-frame time of each frame in path
  order.

```
./lathe --benchmark=perf/head.json [--benchmark-frames=600]
               [--benchmark-warmup=60] [--benchmark-max-warmup=1200]
               [--seed=1337] [--benchmark-screenshots]
```

`--benchmark-screenshots` saves one screenshot per keyframe into
`screenshots/`, which shows what the run looked at.

## Locally

```
tools/perf/run_benchmark.sh build/linux-native-relwithdebinfo perf/base.json
# ...switch branch, rebuild...
tools/perf/run_benchmark.sh build/linux-native-relwithdebinfo perf/head.json
tools/perf/compare_benchmarks.py --base perf/base.json --head perf/head.json
```

With no display (`DISPLAY` and `WAYLAND_DISPLAY` unset), `run_benchmark.sh`
runs the app with `--screen-type=headless`: GLFW's null platform, presenting
through `VK_EXT_headless_surface`. No X server is involved, and the app picks
the best GPU it can see, falling back to lavapipe. On a real GPU, compare runs
from the same machine, with nothing else loading it.

- **Check the device**: the comparison's first lines name the device each
  run used; `llvmpipe` means the GPU wasn't visible.
- **More frames**: a GPU finishes a 240-frame lap in about a second. Use e.g.
  `--benchmark-frames=3000 --benchmark-warmup=300`, so per-frame noise
  averages out and each keyframe segment spans a few seconds.
- **Noise**: boost clocks move with temperature, so run base and head back to
  back. `nvidia-smi -lgc` can pin clocks when numbers wobble.
- **Flags**: `compare_benchmarks.py` flags a stage whose median moves more
  than 10% (stages under 0.5 ms never are), and exits 1 when the full-frame
  median regresses more than 20%.
- **In the build container**: `./compile.sh --shell` passes no GPU through,
  so the app renders on lavapipe there. That shows a run works, not how fast
  it is.

To change what gets measured, edit `BasicGame::benchmark_camera_path()`.
Runs from before and after that change fly different loops, so they don't
compare.
