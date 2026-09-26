# Perf benchmark

`--benchmark` turns the app into a repeatable GPU benchmark, and the `perf`
workflow runs it for every PR against the PR's base.

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
./mingw-vulkan --benchmark=perf/head.json [--benchmark-frames=600]
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

`run_benchmark.sh` starts a private Xvfb when there's no `DISPLAY`. On a
real GPU, compare runs from the same machine, with nothing else loading it.

## In CI (`.github/workflows/perf.yml`)

1. Builds the PR (merge commit) and its base, both RelWithDebInfo.
2. Benchmarks each with 240 frames under Xvfb + lavapipe.
3. Posts the comparison table as a PR comment, updated in place on later
   pushes. It also goes into the job summary.
4. Uploads the JSON results and head's keyframe screenshots as the
   `perf-benchmark` artifact.

- **Flags**: a stage is flagged when its median moves more than 10%. Stages
  under 0.5 ms are never flagged.
- **Failure**: the job fails when the full-frame median regresses more than
  20%.
- **Old bases**: a base branch older than benchmark mode gets a head-only
  report.

Hosted runners have no GPU, so lavapipe does the rendering. It is good at
catching large regressions, such as the meshlet change that doubled frame
time (see [meshlet-rendering.md](meshlet-rendering.md)), but it is not a
stand-in for profiling on real hardware. The relative costs of passes can
differ a lot between a software rasterizer and a GPU.

### On a real GPU

The workflow reads three repository variables (Actions > Variables):
`PERF_RUNNER` (runner label), `PERF_DOCKER_GPU_ARGS` (extra `docker run`
args for the benchmark steps) and `PERF_BENCHMARK_ARGS`. For a self-hosted
Linux box with an NVIDIA card:

- **Container GPU access**: install nvidia-container-toolkit and pass
  `--gpus all --env NVIDIA_DRIVER_CAPABILITIES=all`. The `graphics`
  capability is what mounts the NVIDIA Vulkan ICD into the container.
- **A real X session**: NVIDIA's Vulkan driver can't present to Xvfb, so
  point the run at an X server on the box with `--env DISPLAY=:0 --volume
  /tmp/.X11-unix:/tmp/.X11-unix`, which also needs `xhost +local:` or an
  Xauthority mount. `run_benchmark.sh` only starts Xvfb when `DISPLAY` is
  unset.
- **More frames**: a GPU finishes the 240-frame lap in about a second.
  Use e.g. `--benchmark-frames=3000 --benchmark-warmup=300`, so per-frame
  noise averages out and each keyframe segment spans a few seconds.
- **Noise**: boost clocks move with temperature. Base and head run back to
  back, which helps. `nvidia-smi -lgc` can pin clocks when numbers wobble.
- **Security**: a self-hosted runner executes the PR's code. Keep it to
  private repos or trusted contributors.

The workflow is Linux + Docker only; a Windows runner would need its own
job built around the windows-mingw target.

To change what gets measured, edit `BasicGame::benchmark_camera_path()`.
Base and head then fly different loops until the change is merged, so
expect one noisy comparison.
