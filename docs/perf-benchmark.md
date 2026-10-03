# Perf benchmark

The engine measures itself: `--benchmark` and `--benchmark-suite` turn the app
into a repeatable benchmark, and `--benchmark-compare` compares two results.
Everything (running, statistics, reports, comparison) is C++ in the engine;
no scripts are needed. Run it by hand, on real hardware: CI runners have no
GPU, and lavapipe's timings say little about how passes compare on one.

```
# Everything: every scenario at each load level, 3 interleaved repeats
./lathe --benchmark-suite=perf/head

# Compare two suites (or two single runs); exit 1 on a real regression
./lathe --benchmark-compare=perf/base/suite.json,perf/head/suite.json \
        --benchmark-report=perf/compare.md
```

Run from the build's `bin/` (where the build copies `assets/`), and from a
Release or RelWithDebInfo build: the results record the build type and warn
about Debug. Without a display, add `--screen-type=headless`. A suite with the
default 600 frames per run takes a few minutes on a GPU; lower
`--benchmark-frames` or narrow `--benchmark-scenarios` while iterating.

## The suite

`--benchmark-suite=<dir>` runs each scenario at each of its load levels, each
`--benchmark-repeats` times (3 by default), and writes:

- `report.md`: per-case table (median across repeats, range in brackets),
  scaling fits, hitch attribution and any warnings about the machine.
- `suite.json`: every run's statistics, the per-case aggregates across
  repeats, and the scaling fits.
- `frames/<case>_r<repeat>.csv`: one row per measured frame (below).

### Scenarios

One heavy scene gives you one data point. Each scenario stresses one axis and
is swept over a load, so the result says how cost grows, not just what it is
in one place:

| Scenario | Load axis (defaults) | Stresses |
|---|---|---|
| `game` | -- | the game's own scene and camera path: the realistic mix |
| `game_resolution` | `render_scale_percent` (50, 100, 150) | the game scene at several resolutions: cost that follows resolution is GPU pixel work, cost that doesn't is CPU or per-draw |
| `draw_calls` | `objects` (1000, 4000, 16000) | one entity per object, 16 materials: per-object CPU submission, culling, draw count |
| `instancing` | `instances` (10000, 50000, 200000) | one instanced model of 12-triangle cubes: GPU culling and per-instance cost, shadows included, without per-object CPU cost |
| `instancing_no_shadows` | `instances` (10000, 50000, 200000) | the same field casting no shadows: subtract it from `instancing` for the shadow passes' share |
| `lights` | `point_lights` (64, 512, 4096) | 400 boxes lit by many point lights: light culling, clustering, shading per light |
| `overdraw` | `layers` (4, 16, 64) | full-screen alpha-blended layers: fill rate and blending |

The synthetic scenes are built from engine primitives
(`src/app/benchmark_scenarios.cxx`); while they run, the game's terrain is
neither streamed nor drawn and `IGame::on_ui()` isn't called. Add a scenario
there to cover a new system.

Suite flags:

```
--benchmark-scenarios=lights,draw_calls     only these (default: all)
--benchmark-sweep=lights:256,1024,8192      replace a scenario's loads (repeatable)
--benchmark-repeats=5                       runs per case (3)
--benchmark-render-size=2560x1440           render resolution (1920x1080)
--benchmark-target-hz=240                   the frame budget is 1000 / this (144)
```

plus the shared ones below (`--benchmark-frames`, `--benchmark-warmup`,
`--seed`, ...).

### Method

What the suite does so that numbers are repeatable and comparable:

- **Determinism**: a fixed seed for every case and repeat, a fixed simulated
  step per frame, and a scripted camera loop, so frame N of a case shows the
  same thing in every run, however fast the device is.
- **Warmup**: each case waits at its first keyframe until streaming has
  settled (`--benchmark-warmup`, capped by `--benchmark-max-warmup`), so
  pipeline creation, uploads and allocator growth stay out of the measured lap.
- **Interleaved repeats**: cases run repeat-major (every case once, then every
  case again), so slow drift such as heat soak spreads across all of them
  instead of landing on whichever ran last. The spread across repeats is the
  noise floor of that metric on that machine.
- **Uncapped presentation**: vsync is off while benchmarking and IMMEDIATE is
  preferred (then MAILBOX), so the display doesn't cap what is measured. Some
  compositors still pace MAILBOX to the refresh rate by holding images back,
  which shows up as time in `acquire`. `--present-mode=immediate|mailbox|fifo|fifo_relaxed`
  and `--swapchain-images=<2..8>` override the choice; `--vsync=on` brings FIFO
  back to check whether frames make every refresh. Results record the present
  mode, whether it was the one requested, and the image count.
- **Fixed resolution**: the suite renders at `--benchmark-render-size`,
  whatever the window.
- **Environment**: device, driver, Vulkan version, present mode, build type,
  source revision, CPU, OS, CPU governor and time go into every result, plus
  GPU/CPU temperature, power and clocks before and after each run where hwmon
  exposes them (amdgpu, i915/xe, nouveau; not NVIDIA's proprietary driver).
  Warnings flag a governor other than `performance`, FIFO, lavapipe, Debug
  builds and memory tracking.

Measuring has a small cost of its own: a few clock reads per frame and one
atomic add per recorded event. Build without Tracy (or don't connect it) when
benchmarking; use Tracy afterwards to find out *why* a case is slow.

## What is recorded per frame

Every measured frame becomes a row in the CSV (`BenchmarkFrameSample`):

- **Displayed interval**: present to present, as the CPU sees it
  (`present_interval_ms`). This is the closest thing to what a player sees;
  `displayed_ms` falls back to the CPU frame when there is no previous present.
- **CPU phases** (`app/frame_clock.hxx`): `events`, `update`, `slot_wait`
  (blocked on the frame slot's timeline: the GPU is behind), `acquire`
  (blocked on vkAcquireNextImageKHR), `scene_submit`, `ui`, `prepare`,
  `record`, `submit`, `present` (blocked in vkQueuePresentKHR). *CPU busy* is
  the frame minus the three waits.
- **GPU**: the full frame and every frame graph pass, from timestamp queries.
  They are read back frames-in-flight later; each readback carries the serial
  of the frame that recorded it (`FrameTimings::frame_serial`), so GPU and CPU
  numbers in a row belong to the same frame. After the lap the camera stays
  parked for a few frames to collect the last frames' GPU timings.
- **Workload**: submitted triangles, instances, indirect commands, model and
  mesh submissions, point and spot lights.
- **Events** (`core/perf_events.hxx`): texture uploads, model installs,
  terrain chunk uploads, shader compiles, shader object builds, frame graph
  recompiles, transient (re)allocations, swapchain recreations, render
  resizes and device-wide waits that happened during the frame. Engine code
  records them with `perf_events::record()`.
- **Allocations**: heap allocations and bytes during the frame, in builds
  with memory tracking (not Release).

## Reading the results

- **Distributions, not averages**: every timing is summarised as mean,
  standard deviation, min, p50, p90, p95, p99, p99.9 and max. Average and
  "1% low" frame rates are 1000 / mean and 1000 / p99 of the displayed
  interval. A steady 6 ms beats a 4 ms average with 20 ms spikes, and only the
  tail shows that.
- **Budget**: the share of frames whose displayed interval, CPU busy time or
  GPU time exceeds 1000 / `--benchmark-target-hz` ms by more than 1%: frames
  paced to the refresh rate land a hair either side of it, and that is not a
  miss.
- **Hitches**: displayed frames longer than both twice the run's median and
  the budget. Each lists the events of its frame and the one before (work in
  frame N often shows up in N's or N+1's present); the report counts how many
  hitches each event kind coincided with, and how many had none recorded.
  Per event, the analysis also gives the mean displayed interval of frames
  with and without it.
- **What limits a frame**: *presentation-bound* when the CPU spent at least a
  quarter of the displayed interval blocked in `acquire` or `present` while
  the GPU used less than 90% of it: the swapchain or compositor set the pace,
  so that case's displayed times say nothing about the engine (the report
  warns, and the comparison doesn't judge them). Otherwise *GPU-bound* when
  the GPU time is at least the CPU busy time, else *CPU-bound*. `slot_wait`
  growing means the CPU is waiting on the GPU; `game_resolution` confirms it,
  since GPU-bound cost follows the resolution.
- **Scaling**: for each scenario with a load axis, least-squares fits of GPU
  p50, CPU busy p50 and displayed p99 against the load: the marginal cost per
  unit, R², an exponent from a log-log fit (about 1 is linear, below 1 means a
  fixed cost dominates, above 1 means each unit gets more expensive) and the
  load at which the linear fit reaches the budget.

## Making a claim

A performance claim is conditional and comparative. State the conditions the
report header records (device, driver, resolution, build) and quote tails, not
means: "On <GPU> / <CPU>, 1920x1080, Release: the game scene holds p99 under
<x> ms; GPU cost grows linearly with point lights (+<y> ms per 1000, R² <r>)
up to 4096." Back it with a suite of at least three repeats whose ranges are tight,
and compare against your own previous commits more than against other
engines: equivalent content across engines is hard to build.

## Comparing

`--benchmark-compare=<base.json>,<head.json>` reads two results (suites, or
single runs of either schema), prints a Markdown report and exits without
opening a window:

```
--benchmark-report=<out.md>        also write the report to a file
--benchmark-threshold=<percent>    flag a timing whose median moved more (10)
--benchmark-fail-threshold=<pct>   exit 1 past this on a headline metric (20)
```

Headline metrics are displayed p50 and p99, GPU p50 and CPU busy p50; the
displayed ones are reported but not judged for a case either side ran mostly
presentation-bound. With
repeats on both sides, a change past the threshold only counts when the two
builds' ranges across repeats don't overlap; otherwise it is reported as
within noise. Single runs have no spread, so they are judged on the
thresholds alone. Timings under 0.5 ms are never flagged. Mismatched device,
resolution, present mode, build type, frame count or seed are called out.

## Single runs

`--benchmark=<out.json>` is one run of the game's scene, as before the suite
existed, and also writes the per-frame samples to `<out>.frames.csv`. Its JSON
keeps the earlier keys, so `tools/perf/compare_benchmarks.py` and the other
scripts still read it, and adds `environment` and `analysis` sections.

### What a single run does

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
  p95 / min / max (plus p99 and standard deviation) in JSON, the full-frame
  time of each frame in path order, the analysis described above, and the
  per-frame CSV.

```
./lathe --benchmark=perf/head.json [--benchmark-frames=600]
               [--benchmark-warmup=60] [--benchmark-max-warmup=1200]
               [--seed=1337] [--benchmark-screenshots]
               [--benchmark-target-hz=144] [--benchmark-render-size=WxH]
               [--vsync=on|off]
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

### Locally, with the scripts

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

### Stage ids

The JSON `stages` list has `full_frame` plus one entry per frame graph pass seen, keyed by the pass's stable id
(`gpu_culling`, `shadow_pass`, `depth_prepass`, `hiz_build`, `occlusion_culling`, `depth_prepass_late`, `gtao`,
`gtao_denoise`, `forward_pass`, `bloom`, `composition`, `ui`, `environment`, `light_cull`, `light_cluster`, ...). A pass
not part of a frame (occlusion off, no shadow update) counts as 0 ms for it, and a pass first seen mid-run is backfilled with
0. Both of a pass's timestamps are written at `ALL_COMMANDS`, so its time runs from the
moment the earlier work on its queue has drained to its own end: passes on one queue don't overlap, and their times add
up to at most the queue's span. Work a pass starts before the previous one drains isn't counted in either, so pass times
are a slight underestimate rather than double-counted; results from before this (a `TOP_OF_PIPE` begin) credited a
cheap pass with the tail of the expensive pass before it. `--frame-graph-dump` logs the compiled graph (passes, batches, waits, barriers, transfers) whenever it changes, and
`--async-passes=` / `--frame-graph-alias=` / `--stress-resize=` are described in `docs/frame-graph-status.md`.
