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
