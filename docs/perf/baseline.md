# GPU baseline (before perf/lit-pass-and-async)

Conditions: RTX 4070 Ti SUPER, Release build (`ASSIMP_PREFIX=~/builds/assimp-5.4.3 CMAKE_BUILD_TYPE=Release cargo xtask build`),
`--player --screen-type=headless` (1280x720 viewport), clocks NOT locked, 120 warm-up + 300 sampled frames.
Per-pass times are `vkCmdWriteTimestamp2` begin/end of each frame graph pass, so passes on the two queues overlap and
the sum can exceed the frame time.

Command: `build/release/bin/lathe --game=shadowcity|--rt-poc --player --screen-type=headless --bench-frames=300`

| Pass | queue | shadowcity ms | rt-poc ms |
|---|---|---|---|
| forward_pass (lit) | gfx | 39.2825 | 2.6035 |
| shadow_pass | gfx | 0.9235 | 0.4008 |
| depth_prepass | gfx | 0.2663 | 0.2501 |
| local_shadow_pass | gfx | 0.1078 | - |
| light_cull | cmp | 0.6433 | 0.0099 |
| hiz_build | cmp | 0.0202 | 0.3347 |
| gtao + gtao_denoise | cmp | 0.1280 | 0.3217 |
| rt_poc | cmp | - | 0.1575 |
| bloom | gfx | - | 0.0955 |
| gpu_culling | gfx | 0.0133 | 0.1267 |
| composition | gfx | 0.0279 | 0.0676 |
| other (<0.05 each) | | ~0.1 | ~0.15 |
| **sum of passes** | | 41.4730 | 4.4968 |
| **full frame (GPU)** | | 40.8797 | 4.2661 |

Note: rt-poc here is a different resolution/clock state than the Nsight capture (1533x1015, base clocks, 2.1 ms), so
compare only against runs with the same flags.

## Phase 2 finding: the cost was the pipeline-statistics query, not the shaders

The brief's hypotheses (per-light work in `main_mesh`, host-memory buffers, atomics) did not hold up:

- `main_mesh` has no light loops or atomics (lights are shaded in `main_fs`).
- Moving the per-frame shader-read buffers (UBO, lights, decals, shadow tiles) from host RAM into ReBAR VRAM changed
  `forward_pass` from 39.3 to 39.7 ms, so it was reverted (the debug-build placement warning stays).
- An empty `main_mesh` (`SetMeshOutputCounts(0,0); return;`) still cost ~40 ms, and the cost was linear in the number of
  mesh groups launched (all 41.0 ms, half 21.6, quarter 11.9, none 1.76).
- Moving the renderer's `VK_QUERY_TYPE_PIPELINE_STATISTICS` query (task + mesh + fragment invocation counts) from the
  forward pass to the early depth prepass moved the whole ~39 ms with it (depth_prepass 39.0 ms, forward_pass 0.81 ms).

The statistics query only feeds the editor's "Scene stats" panel, but it ran every frame in every build. It now runs only
while that panel requests it (`Renderer::request_pipeline_stats`).

| Scene (same conditions as above) | full frame before | full frame after | forward_pass before | forward_pass after |
|---|---|---|---|---|
| shadowcity | 40.88 ms | 4.28 ms | 39.28 ms | 1.44 ms |
| rt-poc | 4.27 ms | 3.44 ms | 2.60 ms | 0.99 ms |

Note on `shaders.lsp`: the shader pack is keyed by source path, not content, so a stale pack silently ignores shader
edits. Experiments here used `--shader-pack=<fresh path>` to force live compilation.

## Round 2 (P1/P2), measured reproducibly

Conditions for every number below: RTX 4070 Ti SUPER with the SM clock locked at 2610 MHz (`nvidia-smi -lgc 2610,2610`),
CPU governor `performance`, Release build, `--screen-type=fullscreen --present-mode=immediate` on a 2560x1440@144 display
(render size 2560x1440), `--bench-frames=300` after the default 120 warm-up frames. `--bench-frames` advances the scene
by a fixed 1/60 s per frame, so every run times the same frames (before this, the game animated by wall-clock time and
runs differed by 2x depending on where the camera orbit was). Run-to-run spread is about +-0.01 ms. "Before" means the
original behaviour reproduced on the same binary with `--bench-pipeline-stats --async-passes=light,occlusion,gtao`.

| Scene | before | after |
|---|---|---|
| shadowcity (`--game=shadowcity`) | 41.70 ms | 3.78 ms |
| rt-poc (`--rt-poc`) | 3.89 ms | 2.46 ms |

### What each change was worth

| Change | Result |
|---|---|
| Pipeline-statistics query gated behind the stats panel | the whole 40 ms (see above) |
| Async queue per pass (`--async-passes`), shadowcity | all 4.054, none 4.000, **gtao only 3.784**, occlusion only 4.069, light only 4.053 ms |
| Async queue per pass, rt-poc | all 2.500, none 2.512, **gtao only 2.457**, occlusion only 2.508, light only 2.516 ms |
| Async queue per pass, light scenes (lights / meshlets / gallery) | none is best: 1.995 / 0.950 / 0.941 ms vs gtao-only 2.082 / 0.997 / 0.976 ms |
| Async queue per pass, heavy suite cases (GPU p50) | gtao-only best: `draw_calls@16000` 4.10 vs 4.24 (none) vs 4.31 (all), `instancing@200000` 2.63 vs 2.82 vs 2.79 ms |
| Async GTAO chosen automatically (default) | matches the best fixed choice in every scene above to within ~0.015 ms |
| Local shadow passes declared every frame | median frame unchanged (3.35 ms); `shadow_city@24` suite p99 5.9-6.1 -> 4.0-4.2 ms, p99.9 9.0 -> 6.8-7.0 ms; the plan now recompiles only during startup |
| Redundant timeline waits dropped | no measurable change (unit tested, validation clean) |
| Narrow semaphore wait stages | **not kept**: no faster (rt-poc 2.20-2.34 vs 2.32-2.35 ms) and sync validation reports layout-transition WRITE_AFTER_WRITE hazards |
| Removing the lit pass's global token barrier outright | no measurable change (shadowcity 3.785 vs 3.790 ms, rt-poc 2.454 vs 2.455 ms), so it was left alone |

Validation: a Debug build with `VK_LAYER_KHRONOS_validation` and sync validation reports no messages for `--rt-poc` or
`--game=shadowcity` over 500 frames.

### Remaining gaps and notes

- Async occlusion culling and async light clustering never paid off in these scenes, so they are off by default
  (`--async-passes=light,occlusion,gtao` forces the old behaviour).
- `rt_poc` reads the lit HDR image, so it cannot overlap the lit pass; the graphics queue idles around it.
- The brief's rt-poc target (<= 1.6 ms) was stated for a 1533x1015 render at base clocks; these runs are 2560x1440 at 2610
  MHz and are not comparable to it.

### Corrections after independent verification (docs/perf/results.md)

- The "+-0.01 ms" spread holds for every default-configuration run but not for the original behaviour on shadowcity
  (`--bench-pipeline-stats`), which spread by 0.70 ms across 3 runs: the statistics query makes that frame slow and uneven.
- The first version of the automatic async-GTAO switch averaged frame time from the very first frame, so shader
  compilation and first-use allocations switched it on for light scenes: meshlets came out 0.015 ms (1.6%) slower than
  `--async-passes=none`. It now ignores the first 90 frames. Re-measured (3 runs, medians): meshlets 0.960 vs 0.959 ms,
  lights 2.012 vs 2.017, gallery 0.945 vs 0.946, materials 1.800 vs 1.799, shadowcity 3.785 vs 3.992 (none), rt-poc 2.455
  vs 2.514 (none).
- The `shadow_city@24` p99 and p99.9 improvements were measured by the author (two interleaved repeats each) and not
  repeated by the verifier.
- The brief's async-overlap targets (Nsight timeline) were not checked: no Nsight capture was taken. Per-pass start times
  show GTAO on the compute queue overlapping `shadow_pass` in shadowcity.
- The placement warning now fires only for device-mode buffers that fell back to host memory; intentional host buffers
  get a debug line.
- The branch history was reorganised after verification: the user's unrelated ray-query and editor work is its own commit
  below the perf commits, and every commit builds on its own.
