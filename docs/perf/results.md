# Independent verification of perf/lit-pass-and-async

Verified on 2026-10-18 against `git diff main...HEAD` (af1d9a8, 8dcb89d, 73f6dcf, 529060c). The binaries were built
from the working tree, which also holds the user's uncommitted rt-poc, editor-camera and resource-table work.

## Conditions

- RTX 4070 Ti SUPER, driver 615.71.09, SM clock locked at 2610 MHz (checked with `nvidia-smi`), CPU governor `performance`.
- Release build: `ASSIMP_PREFIX=~/builds/assimp-5.4.3 CMAKE_BUILD_TYPE=Release cargo xtask build`.
- Command: `build/release/bin/lathe <scene> --player --screen-type=fullscreen --present-mode=immediate --bench-frames=300`.
  This runs visibly at a 2560x1440 swapchain, with 120 warm-up frames and a fixed 1/60 s step.
- Each configuration was run 3 times. The tables give the median, in mean GPU ms per frame.
- "Original" means `--bench-pipeline-stats --async-passes=light,occlusion,gtao`, run on the same binary.
- "All async, no stats" means `--async-passes=light,occlusion,gtao` without the statistics query. It separates the
  effect of the query from the effect of the async choice.

## Main scenes: before vs after

`c` = async compute queue, `g` = graphics queue. Passes under 0.02 ms in every column are left out.

### shadowcity (`--game=shadowcity`)

| pass | original | all async, no stats | `--async-passes=none` | **default (after)** |
|---|---|---|---|---|
| forward_pass | 39.217 (g) | 1.716 (g) | 1.689 (g) | **1.674 (g)** |
| shadow_pass | 1.119 (g) | 0.983 (g) | 0.970 (g) | 1.210 (g) |
| depth_prepass | 0.418 (g) | 0.438 (g) | 0.383 (g) | 0.430 (g) |
| depth_prepass_late | 0.082 (g) | 0.083 (g) | 0.054 (g) | 0.054 (g) |
| gtao | 0.350 (c) | 0.315 (c) | 0.309 (g) | 0.408 (c) |
| gtao_denoise | 0.252 (c) | 0.227 (c) | 0.244 (g) | 0.307 (c) |
| hiz_build | 0.044 (c) | 0.039 (c) | 0.029 (g) | 0.030 (g) |
| local_shadow_cull | 0.014 (g) | 0.009 (g) | 0.054 (g) | 0.009 (g) |
| local_shadow_pass | 0.115 (g) | 0.115 (g) | 0.113 (g) | 0.114 (g) |
| composition | 0.111 (g) | 0.109 (g) | 0.109 (g) | 0.109 (g) |
| sum of passes | 41.775 | 4.086 | 4.003 | 4.391 |
| **full frame** | **41.792** | 4.114 | 4.034 | **3.774** |
| full-frame runs | 41.35 / 41.79 / 42.06 | 4.114 / 4.115 / 4.110 | 4.031 / 4.034 / 4.048 | 3.777 / 3.765 / 3.774 |

### rt-poc (`--rt-poc`)

| pass | original | all async, no stats | `--async-passes=none` | **default (after)** |
|---|---|---|---|---|
| forward_pass | 2.166 (g) | 0.736 (g) | 0.734 (g) | **0.731 (g)** |
| gtao | 0.634 (c) | 0.628 (c) | 0.630 (g) | 0.651 (c) |
| gtao_denoise | 0.186 (c) | 0.186 (c) | 0.188 (g) | 0.185 (c) |
| rt_poc | 0.258 (c) | 0.286 (c) | 0.284 (c) | 0.290 (c) |
| depth_prepass | 0.192 (g) | 0.189 (g) | 0.194 (g) | 0.182 (g) |
| shadow_pass | 0.069 (g) | 0.070 (g) | 0.066 (g) | 0.118 (g) |
| hiz_build | 0.045 (c) | 0.046 (c) | 0.029 (g) | 0.032 (g) |
| depth_prepass_late | 0.042 (g) | 0.042 (g) | 0.034 (g) | 0.036 (g) |
| bloom | 0.109 (g) | 0.109 (g) | 0.106 (g) | 0.106 (g) |
| composition | 0.117 (g) | 0.116 (g) | 0.116 (g) | 0.116 (g) |
| gpu_culling | 0.038 (g) | 0.038 (g) | 0.037 (g) | 0.038 (g) |
| sum of passes | 3.894 | 2.483 | 2.460 | 2.530 |
| **full frame** | **3.954** | 2.531 | 2.540 | **2.485** |
| full-frame runs | 3.955 / 3.945 / 3.954 | 2.528 / 2.533 / 2.532 | 2.540 / 2.540 / 2.544 | 2.485 / 2.478 / 2.487 |

### Compared with docs/perf/baseline.md (Round 2)

| scene | baseline before | measured before | baseline after | measured after |
|---|---|---|---|---|
| shadowcity | 41.70 | 41.79 (spread 0.70) | 3.78 | 3.774 |
| rt-poc | 3.89 | 3.954 | 2.46 | 2.485 |

- Both headline results reproduce.
  - **shadowcity:** 41.8 -> 3.77 ms (11x).
  - **rt-poc:** 3.95 -> 2.49 ms (-37%).
- **Almost all of the gain comes from gating the pipeline-statistics query.** Turning the query off alone, with the async
  setup unchanged, gives shadowcity 41.79 -> 4.11 ms and rt-poc 3.95 -> 2.53 ms. The async GTAO choice adds the rest:
  0.26 ms in shadowcity (4.03 none -> 3.77) and 0.055 ms in rt-poc (2.54 -> 2.485).
- **Run-to-run spread:** about 0.01-0.02 ms in every configuration except "original" in shadowcity, which spread 0.70 ms.
  The "+-0.01 ms" stated in baseline.md does not hold for that configuration.
- **rt-poc "after" came out at 2.485 ms against the stated 2.46 ms.** The difference is small, but it is outside the
  stated spread.

## Other games: default (auto async GTAO) vs `--async-passes=none` vs forced `gtao`

Full frame, median of 3, same conditions.

| game | none | default (auto) | forced gtao | default vs none |
|---|---|---|---|---|
| lights | 2.031 | 2.023 | 2.114 | -0.008 (within noise) |
| materials | 1.810 | 1.809 | 1.811 | -0.001 |
| meshlets | 0.954 | 0.970 | 1.013 | **+0.015 (+1.6%)** |
| meshlets, `--bench-warmup=800` | 0.952 | 0.967 | - | **+0.015** |

- **lights and materials:** the heuristic correctly stays off.
- **meshlets regresses slightly with the default.** In that scene the auto switch turns async GTAO **on** during the
  heavy startup frames, then holds it for 240 frames before turning it off again. The log shows 3 plan compiles:
  1 batch, then 4 batches, then 1 batch. After it switches back, the plan is the same as with `none`, but the `gtao`
  pass on the graphics queue runs about 12 µs slower (0.266 vs 0.254 ms). The gap persists after an 800-frame warm-up.
  - The likely cause is a different transient memory placement after the reallocation. Stale synchronisation state is
    less likely, since validation stays clean. I did not prove the cause.

## Validation

Debug build, `--player --screen-type=headless --exit-after-frames=500`, with `VK_LAYER_KHRONOS_validation` and
synchronization validation on. The log confirms "Synchronization validation enabled". Lines counted as errors match
`VUID|SYNC-HAZARD|[E] Vulkan validation`.

| run | validation errors | [E] | [W] |
|---|---|---|---|
| shadowcity default | 0 | 0 | 16 |
| shadowcity original flags | 0 | 0 | 16 |
| shadowcity `--async-passes=none` | 0 | 0 | 16 |
| shadowcity `--async-compute=same_family` | 0 | 0 | 16 |
| shadowcity `--async-compute=off` | 0 | 0 | 16 |
| rt-poc default | 0 | 0 | 21 |
| rt-poc original flags | 0 | 0 | 21 |
| lights default | 0 | 0 | 16 |
| meshlets default | 0 | 0 | 16 |
| meshlets, Release + layer forced on (`VK_LOADER_LAYERS_ENABLE=*validation`, `VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true`), fullscreen, 1500 frames; covers the auto switch going on and then off | 0 | - | - |

- **Every `[W]` line comes from the new `log_shader_read_placement` check in buffer.cxx.** The flagged buffers are
  `renderer.ubo`, `frame_lights`, `frame_decals`, `frame_lod_jobs`, `frame_occlusion_views`, `frame_frustum_planes`,
  `frame_local_shadow_tiles/planes` and `imgui.vertices`, plus `rt_poc_*` in rt-poc. All of them sit in host memory on
  purpose.
- There are no other new warnings.

## Rendering comparison

- **Setup:** Release, headless 1280x720. Screenshots taken with F12 at frames 60, 120 and 180 (plus 240 in lights), with
  shot keys 2/3/4 pressed in between.
- **Fixed time step:** `--bench-frames=1000000` forces a fixed 1/60 s step on HEAD. For `main` I used a scratch copy
  (`git archive main`) with a one-line `LATHE_FIXED_DT` patch, built with `cargo xtask` and run from the repo root so it
  had the same assets.
- **Diff method:** per-pixel diff with numpy/pillow.

| comparison | result |
|---|---|
| shadowcity: default vs original flags vs `none` vs a second default run (3 shots each) | bit-identical (max diff 0) |
| shadowcity: HEAD default vs `main` (3 shots) | bit-identical |
| lights: default vs original flags vs `none` vs a second default run (4 shots each) | bit-identical |
| lights: HEAD default vs `main` (4 shots) | bit-identical |
| rt-poc: default vs original flags vs `none` (2 shots) | mean abs diff ~1.15, ~3.5% of pixels differ by more than 8 |
| rt-poc: default vs a second default run | the same magnitude (mean ~1.18, ~3.5%) |

- **rt-poc:** two runs with identical flags differ by the same amount, so the differences are not caused by the change.
  The images look the same by eye: the RT sphere, shadows and particles are all present. rt-poc does not exist on
  `main`, so it could not be compared against `main`.
- **Lights, shadows, models and AO are present in every case.**

## Unit tests

`cargo xtask test` (Release): 767/767 passed, including the 6 new `submission_plan` subcases.

## Code review findings

| # | severity | finding |
|---|---|---|
| 1 | **high (repo hygiene)** | **The four commits do not build on their own.** They contain parts of the user's uncommitted, unrelated work:<br>- 529060c/af1d9a8 `engine_main.cxx` uses `runtime.ray_tracing_poc`, `RayTracingPoc::view_names`, `runtime.game_factory`, `runtime.available_games` and `runtime.frame_view/frame_projection`.<br>- 73f6dcf `application.cxx` adds `camera_pivot`, `focus_selection`, `CameraDrag`, the `camera.orbit/pan/dolly` calls and `TimingSeries::smoothed/primed`.<br>- af1d9a8 `vulkan_bootstrap.cxx` enables `VK_KHR_ray_query` and acceleration structures.<br>None of these are declared in any header at HEAD. They exist only in the uncommitted `runtime.hxx`, `application.hxx`, `editor_camera.hxx` and the untracked `ray_tracing_poc.*`. A clean checkout of the branch will not compile, and the editor-camera, rt-poc and ray-query changes will show up in a perf PR. The commits need splitting before merge. |
| 2 | low | **Auto async GTAO turns on spuriously at startup in light scenes (meshlets).** The EMA is seeded with the first measured frame, which is heavy, so async GTAO turns on and is then held for 240 frames. This costs 2 extra plan and transient reallocations at startup, and in meshlets about +0.015 ms that persists afterwards (see above). Possible fixes: seed the EMA after a few frames, or skip the hold on the first decision. The thresholds (2.4/1.9 ms) are absolute GPU ms tuned on one GPU at 2560x1440 and will not carry over to other GPUs or resolutions. The hysteresis gap does exceed the measured async gain, so the switch does not oscillate. |
| 3 | low | **Local shadow passes are now declared on every frame with any local shadow tile, even when no tile is redrawn.** On frames with no dirty tiles the executor still begins and ends a dynamic rendering pass over the whole atlas (LOAD/STORE), plus the layout transitions sampled -> depth attachment -> sampled. It also records an empty cull pass with barriers. I did not isolate the per-frame cost of those empty frames: lights shows `local_shadow_pass` at 0.042 ms mean and `local_shadow_cull` at 0.009 ms. The fallback is correct:<br>- the cull buffers are per frame slot and checked with `valid()`;<br>- when `frame.local_shadow_culled` is false but the buffers exist, the cull record returns early and the raster pass draws from `source_indirect`, which is now always declared as a read;<br>- the extra declared reads are harmless.<br>One edge case: if the atlas is not yet initialised and a frame has tiles but no render tiles, the pass uses `dont_care`/store and does not set `local_shadow_atlas_initialized_`. That is benign, because nothing valid is cached in that state. |
| 4 | info (verified safe) | **The redundant-wait elimination is sound.**<br>- `issued` lives only inside one `plan_submissions` call, so nothing is skipped across frames.<br>- It is keyed by (waiting logical queue, waited logical queue). Values from one logical queue increase strictly, so a prior wait at value >= V covers V. A semaphore wait's second scope covers all later submissions on the same queue, limited to its stage mask.<br>- Stage coverage is checked: a later wait is skipped only if the prior wait covered ALL_COMMANDS or a superset of its stages.<br>- When both logical queues alias one physical queue and timeline (`--async-compute=off`), keying by logical queue is only more conservative.<br>- No submit in the plan is skipped (empty batches are still submitted), so the wait that was kept is always actually issued.<br>- In practice the compiler forces every wait to ALL_COMMANDS, so the stage logic is never exercised outside the unit tests. |
| 5 | info (verified safe) | **The pipeline-statistics gating pairs correctly.**<br>- The query pool is reset every frame unconditionally.<br>- Begin and end are guarded by the same null-pool check inside one function.<br>- `has_results` is derived from the same `pipeline_stats_hold_frames_` value used when recording the forward pass. `on_ui`, which calls `request_pipeline_stats`, runs before `record_frame`, and the counter is decremented only in `record_frame_end`.<br>- If they ever disagreed, `vkGetQueryPoolResults` without WAIT returns NOT_READY and the stats are marked invalid.<br>- Minor: `last_frame_pipeline_stats_` keeps its old values while the query is off, so reopening the panel briefly shows stale numbers.<br>- **Design note:** while the Scene stats panel is open, every editor frame pays the full ~37 ms. Sampling the query once every N frames would make the panel almost free. |
| 6 | low | **Debug labels.**<br>- Begin and end are paired on the normal path. On the `translate()` error path in executor.cxx, the label opened before `vkCmdBeginRendering` is never closed. The command buffer is abandoned on that error, so this is cosmetic.<br>- `VK_EXT_debug_utils` is now enabled on the instance in Release whenever it is available, and the function pointers come from volk after `volkLoadInstance`.<br>- Labels on the dedicated compute family are legal.<br>- One `std::string` is allocated per pass per frame. That is negligible, but it could use the `string_view`'s data when it is NUL-terminated. |
| 7 | low | **The buffer placement warning (Debug only) fires 16-21 times per run** for buffers that are host-resident on purpose (UBO, lights, decals and so on). There is no opt-out or allowlist, so the warning will be ignored in practice. It needs a `BufferMemory`-based exemption, or a downgrade to `debug` for intended host placements. |
| 8 | info | **`--bench-frames`:** the scene advances by a fixed 1/60 s per frame, warm-up is counted in frames, and each frame serial is sampled once. A pass missing from some frames is averaged as 0 for those frames, which is correct for "mean per frame". After printing it `break`s out of the main loop and goes through normal shutdown. |
| 9 | info | **`--async-passes` accepts unknown tokens with only a warning.** An empty token, for example `gtao,`, falls through silently. This is harmless. |

## What is overclaimed or unmeasured in baseline.md

- **"Run-to-run spread is about +-0.01 ms."** True for every configuration except the original-behaviour shadowcity run,
  which spread 0.70 ms (41.35-42.06).
- **"Async GTAO chosen automatically matches the best fixed choice in every scene to within ~0.015 ms."** This is
  borderline for meshlets: default is 0.970 against 0.954 for none (+0.015, reproducible with a 0.002 spread). The cause
  is a startup flip, not steady-state behaviour.
- **The `shadow_city@24` suite p99/p99.9 improvements** from declaring the local shadow passes every frame were not
  re-measured here (no suite run). I could not verify them.
- **"Redundant timeline waits dropped: no measurable change"** is consistent with the code: every wait is ALL_COMMANDS,
  and the frame has 2-5 waits.
- **The brief's P1 acceptance criteria** (async overlap above 50%, no graphics idle during step 6) were not checked,
  because no Nsight capture was allowed. The per-pass start times show that GTAO on compute overlaps `shadow_pass` in
  shadowcity: `shadow_pass` grows from 0.97 to 1.21 ms while the frame shrinks by 0.26 ms.
- **baseline.md does not mention that the branch commits carry unrelated editor and rt-poc changes** (issue 1).

## Verdict

**The performance claims hold.**

- **shadowcity:** 41.8 -> 3.77 ms. About 37.7 ms of that comes from gating the pipeline-statistics query, and 0.26 ms
  from async GTAO.
- **rt-poc:** 3.95 -> 2.49 ms.
- **Rendering:** bit-identical to `main` and across async modes in shadowcity and lights.
- **Validation and tests:** sync validation is clean in every configuration tested, and all unit tests pass.

**Do not merge as-is, because of issue 1.** The commits do not compile without the uncommitted working tree, and they
contain unrelated editor and rt-poc work. The other findings are low severity.

## Status after the verifier's report

Written by the author after the verification above, not by the verifier.

- Issue 1 (commits did not build on their own) is fixed: the unrelated ray-query / editor / resource-table work is now its
  own commit (`0161f0c`) below the perf commits, and every commit from `0161f0c` to the tip was built in a clean worktree
  with `cargo xtask build` (Release).
- Issue 2 (async GTAO switching on at startup): the average now ignores the first 90 frames. Meshlets default vs
  `--async-passes=none` re-measured at 0.960 vs 0.959 ms (3 runs each), lights 2.012 vs 2.017, gallery 0.945 vs 0.946,
  materials 1.800 vs 1.799; shadowcity 3.785 vs 3.992 and rt-poc 2.455 vs 2.514 keep their gain.
- Issue 4 (debug label left open when the render pass cannot be built): the label is ended before the early return.
- Issue 5 (placement warning noise): it now warns only for device-mode buffers that fell back to host memory; buffers
  written from the host on purpose get a debug line.
- Not changed: issue 3 (empty local-shadow passes on frames with nothing to redraw, about 0.05 ms in the lights game) and
  the stats-panel cost (design note 6).
