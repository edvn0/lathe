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
               [--cluster-grid=16x9x24:256] [--occlusion-culling=on|off]
               [--meshlet-occlusion=on|off]
               [--occlusion-test=hiz|never_occluded|always_defer]
```

`--benchmark-screenshots` saves one screenshot per keyframe into
`screenshots/`, which shows what the run looked at. It captures the viewport
target, not the window: the editor panels show live counters and log lines that
can never match between runs. The shader clock (`time`) starts at 0 with the
first measured frame, so the length of the warmup (which ends when streaming
settles) cannot change what frame N looks like. Two runs of one build give
byte-identical screenshots; `tools/perf/compare_screenshots.py --base <dir>
--head <dir>` checks that (exit 0 only if every keyframe is identical, otherwise
it prints the first differing pixel).

`--occlusion-test=` selects `Renderer::occlusion_test_mode()`. The two stubs
(`never_occluded`, `always_defer`) must render exactly like occlusion culling
off.

The JSON has a `counters` object: for each culling and clustering counter
(`frustum_visible_instances`, `early_instances`, `occlusion_candidates`,
`late_instances`, `occluded_instances`, `deferred_meshlets`,
`occluded_meshlets`, `occupied_clusters`, `overflowing_clusters`,
`maximum_lights`, `stored_lights`) the mean over the measured frames it was
read back for and the last such value (`null` if it never was, e.g. occlusion
counters with occlusion off). They lag the frame by the frames in flight.

`tools/perf/capture_baselines.sh <build dir> perf/baseline` captures the four
baselines the frame-graph migration is checked against (`occ_off`, `occ_on`,
`occ_meshlet`, `always_defer`): JSON plus screenshots each. `perf/` is
gitignored.

`--cluster-grid=XxYxZ[:capacity]` sets the clustered-lighting grid (see
`docs/clustered-lighting.md`), so two runs of one build can compare grids.
The JSON records it as `cluster_grid`.

`--occlusion-culling=on|off` toggles two-phase Hi-Z occlusion culling (see
`docs/occlusion-culling.md`; off by default). The JSON records whether it was
active as `occlusion_culling`. Its stages (`hiz_build`, `occlusion_culling`,
`depth_prepass_late`) are always in the JSON, near 0 ms while it is off, so
compare on vs off by the sum `depth_prepass + hiz_build + occlusion_culling +
depth_prepass_late + forward_pass` as well as the full frame. With
`--benchmark-screenshots`, the keyframe screenshots of an on and an off run
should be bit-identical. `run_benchmark.sh` passes extra arguments through,
e.g. `tools/perf/run_benchmark.sh <build dir> perf/on.json
--occlusion-culling=on`. Comparing a base from before these stages existed is
fine: `compare_benchmarks.py` lists stages new in head on their own.

`--meshlet-occlusion=on|off` adds per-meshlet Hi-Z occlusion in the task shader
on top of it (`docs/occlusion-culling.md`, "Meshlet level"; off by default, and
inert unless `--occlusion-culling=on`). The JSON records it as
`meshlet_occlusion`, true only when it was actually active. Compare it against
an `--occlusion-culling=on` run with the same stage sum, and look at the forward
pass's task/mesh invocations in Scene stats.

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

## Stage ids

The JSON `stages` list has `full_frame` plus one entry per frame graph pass seen, keyed by the pass's stable id
(`gpu_culling`, `shadow_pass`, `depth_prepass`, `hiz_build`, `occlusion_culling`, `depth_prepass_late`, `gtao`,
`gtao_denoise`, `forward_pass`, `bloom`, `composition`, `ui`, `environment`, `light_cull`, `light_cluster`, ...). A pass
not part of a frame (occlusion off, no shadow update) counts as 0 ms for it, and a pass first seen mid-run is backfilled with
0. `--frame-graph-dump` logs the compiled graph (passes, batches, waits, barriers, transfers) whenever it changes, and
`--async-passes=` / `--frame-graph-alias=` / `--stress-resize=` are described in `docs/frame-graph-status.md`.
