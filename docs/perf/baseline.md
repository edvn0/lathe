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
