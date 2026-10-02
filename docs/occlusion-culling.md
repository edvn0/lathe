# Occlusion culling (two-phase Hi-Z, instance and meshlet level)

Geometry hidden behind depth that is already drawn is culled on the GPU before the camera passes draw it. The test is
per instance (its world AABB) in the culling compute passes and, optionally, per meshlet (its world bounding sphere)
in the task shader; see [Meshlet level](#meshlet-level).

**Status: both toggles default off** (`Renderer::occlusion_culling_` and `meshlet_occlusion_culling_` are `false`).
CI builds, unit-tests and clang-tidies this code and compiles `frustum_cull.slang` (`main_cs`) and `hiz_build.slang`
(`main_cs`) with `slangc` for push-constant reflection, but CI has no GPU: every other shader entry point is only
compiled at runtime, and nothing about the GPU behaviour (pixel identity, synchronisation, timings) has been run.
Turn either on by default only after the checks in [Verifying on a GPU](#verifying-on-a-gpu) pass.

## Design

- **Depth convention.** `glm::perspectiveLH_ZO` plus a viewport with `minDepth = 1, maxDepth = 0` and a flipped Y:
  stored depth is `1 - ndc.z`, larger is nearer, the clear (0) is far, and pixel row 0 is `ndc.y = +1`. The pyramid
  keeps the **farthest** depth of each footprint, i.e. the MIN.
- **Two phases** (Nanite-style). Phase 1 (`main_cs`, in `prepare_frame`) tests each frustum-visible instance's
  *current* AABB against **last frame's** pyramid, projected with **last frame's** view-projection. Passing instances
  are drawn by the early prepass; failing ones are not dropped but deferred as *candidates*. Phase 2 (`late_cs`, in
  `record_frame`) re-tests only the candidates against a pyramid built from **this frame's** early depth; survivors
  are drawn by the late prepass, the rest are culled. Correctness depends only on phase 2 (its pyramid holds a subset
  of the final depth); phase 1 is a heuristic, so stale history (camera cuts, moved or LOD-switched instances) only
  costs time. No per-instance state survives a frame, so batch reordering is harmless.
- **No history** (first frame, resize, toggling, a frame with culling off): phase 1 draws every frustum-visible
  instance, phase 2 is empty, and the pyramid is still built for the next frame.
- **MSAA** (on by default, 4x): the early prepass resolves depth with `VK_RESOLVE_MODE_MIN_BIT` (each pixel's
  farthest sample) and the pyramid is built from that; the late prepass re-resolves with `SAMPLE_ZERO`, so GTAO sees
  exactly today's depth. Without MIN support under MSAA (`VulkanContext::depth_resolve_min_supported`),
  `Renderer::occlusion_culling_supported()` is false and culling stays inactive.
- **Pyramid** (`Renderer::HizPyramid hiz_`): R32_SFLOAT, shared by the frames in flight (one queue, in-order
  submission). Logical level `L` is `ceil(W / 2^(L+1)) x ceil(H / 2^(L+1))`; `hiz_mip_count()` levels down to 1x1. The
  image is `hiz_image_extent()` (level 0 rounded up to powers of two, since Vulkan floor-halves mips) and only the
  logical region is written or read. Each texel is the MIN of its 2x2 children inside the source's logical extent
  (1.0 is the neutral value). Built by one dispatch per level (`hiz_build.slang`, written through an `r32f` storage
  declaration, read through `sampled_2d_depth`). Recreated on resize.
- **Test** (`aabb_occluded()` in `assets/shaders/hiz_occlusion.slang`, mirrored by
  `include/rendering/hiz_occlusion.hxx`): project the 8 corners; any corner with `clip.w <= 0` or `clip.z < 0`, or
  a rect fully off screen, means "visible". Grow the pixel rect by `guard_pixels` (1), pick the level
  `firstbithigh(max span)` where it spans at most 2x2 texels, and cull only if the box's nearest depth plus
  `depth_epsilon` (1e-6) is less than the farthest of those texels. Ties are visible.
- **Which batches.** Opaque and mask batches are occluders and occludees (`occludable_batch_count`); blend batches
  never write prepass depth and are never deferred. Shadows are unaffected (they draw the un-culled buffers).
  Instanced (vertex-shader) and meshlet batches are treated alike.
- **Forward `EQUAL` guarantee.** Forward draws `merged_indirect` (phase 1 + phase 2 instances) with the same
  `cull_flags` as the prepass (or, with meshlet occlusion, replays its recorded bits), so it draws every primitive that
  won the depth test. Instances within a batch are
  ordered phase 1 first, which can only change pixels where two instances of one batch have bit-identical depth.

## Frame order

`prepare_frame`:

1. `upload_frame_data`, then `vkCmdFillBuffer(occlusion_stats_buffer, 0)` + barrier (transfer -> compute/task).
2. Decide `frame.occlusion_active = occlusion_culling_ && occlusion_culling_supported() && hiz_ exists &&
   hiz_.depth_extent == render extent`. Write `occlusion_views_buffer`: view [0] = history pyramid + history
   view-projection, view [1] = this frame's view-projection; `enabled` from `occlusion_view_states()`.
3. **GPU Culling** stage: `main_cs` over all batches (frustum test, then the view [0] test), then barriers for the
   prepass readers and for `late_cs`.

`record_frame`:

| Stage (timestamps) | Work | Inactive |
|---|---|---|
| Shadow Pass | unchanged | |
| Depth prepass | phase `early`: CLEAR, draws `culled_indirect` (n1), MIN resolve | phase `only` (today's pass) |
| Hi-Z build | `render_pass::build_hiz` | empty timestamps |
| Occlusion culling | `late_cs` over all batches, view [1] | empty timestamps |
| Depth prepass (late) | phase `late`: LOAD, draws `late_indirect` (n2), SAMPLE_ZERO resolve; begins rendering even with no draws | empty timestamps |
| (no stage) | statistics copy to `occlusion_stats_readback_buffer` | same |
| Ambient Occlusion, Forward Pass, ... | forward draws `merged_indirect` (n1 + n2) | forward draws `culled_indirect` |

After the build is recorded: `hiz_history_view_projection_ = frame.view_projection; hiz_history_valid_ = true`.
A frame without occlusion clears `hiz_history_valid_`. Every stage writes both timestamps every frame
(`Renderer::write_empty_stage`); a missing one makes `vkGetQueryPoolResults` return `VK_NOT_READY` and drops the
frame's timings.

## Barriers

- **Stats clear** -> `main_cs`: `ALL_TRANSFER/TRANSFER_WRITE` -> `COMPUTE|TASK`, storage read/write.
- **After `main_cs`**: visible draws/transforms -> vertex/task/mesh read and compute read/write (late_cs appends);
  `culled_indirect` -> draw-indirect/task/compute read; candidates and counts -> compute read; stats -> compute
  read/write and copy read.
- **Hi-Z build** (`build_hiz`): MSAA depth `DEPTH_ATTACHMENT -> DEPTH_ATTACHMENT` (fragment tests + colour output,
  write -> read/write); source depth `DEPTH_ATTACHMENT -> SHADER_READ_ONLY` (src fragment tests + colour output,
  because depth resolves run in `COLOR_ATTACHMENT_OUTPUT`); all pyramid levels `UNDEFINED -> GENERAL` (src
  `COMPUTE|TASK|FRAGMENT`, access none: the previous build's readers); per level after its dispatch `GENERAL ->
  SHADER_READ_ONLY` (dst `COMPUTE|TASK|FRAGMENT`, sampled read); source depth back to `DEPTH_ATTACHMENT`.
- **Before `late_cs`**: execution-only `VERTEX|TASK|MESH -> COMPUTE` (the early prepass reads ranges `late_cs` does
  not write, but device-address accesses can't be tracked per range).
- **After `late_cs`**: visible draws/transforms -> vertex/task/mesh read; `late_indirect`, `merged_indirect` ->
  draw-indirect/task read; stats -> copy read.
- **Stats copy**: `COMPUTE|ALL_TRANSFER` write -> copy read, copy, then copy -> host read.
- **History across frames**: no extra barrier. Frame N's last per-level barrier has `COMPUTE|TASK` in its second
  scope, which covers frame N + 1's `main_cs` (later in submission order on the same queue).

## Buffers

All per frame (`RendererFrame`), created in `Renderer::initialize`:

| Buffer | Size | Written by | Read by |
|---|---|---|---|
| `occlusion_views_buffer` | 2 x `GpuOcclusionView` (112 B), upload | CPU, `prepare_frame` | `main_cs` ([0]), `late_cs` ([1]) |
| `occlusion_candidates_buffer` | `maximum_draw_count` x u32 | `main_cs` at `[first_instance, + c)` | `late_cs` |
| `candidate_counts_buffer` | batches x u32 | `main_cs` | `late_cs` |
| `culled_indirect_buffer` | existing | `main_cs`: `(first, n1)` | early prepass, `late_cs`, forward when inactive |
| `late_indirect_buffer` | batches x `GpuDrawCommand` | `late_cs`: `(first + n1, n2)` | late prepass |
| `merged_indirect_buffer` | batches x `GpuDrawCommand` | `late_cs`: `(first, n1 + n2)` | forward |
| `visible_draw_buffer` / `visible_transform_buffer` | existing | `main_cs` `[first, + n1)`, `late_cs` `[first + n1, + n2)` | all camera passes |
| `meshlet_visibility_buffer` | one bit per meshlet, power-of-two bytes, grown on demand | `vkCmdFillBuffer` clear; task shaders (`InterlockedOr`) in both prepass phases | the late prepass phase, forward (meshlet occlusion only) |
| `occlusion_stats_buffer` | 8 x u32 | atomics in `main_cs`/`late_cs` and the task shaders | readback copy |
| `occlusion_stats_readback_buffer` | 8 x u32, readback | copy | `consume_culled_readback` |

"Batches" is `min(maximum_draw_count, 65535)`: one cull workgroup per batch, and `prepare_frame` refuses more.
`n1 + c <= instance_count` and `n2 <= c`, so nothing overflows a batch's range. Statistics slots
(`occlusion_stat_*`): 0 frustum-visible, 1 phase 1 drawn, 2 candidates, 3 phase 2 drawn, 4 meshlets deferred by
phase 1, 5 meshlets occluded by phase 2, 6-7 unused. The Hi-Z pyramid itself is a Renderer member (`hiz_`), roughly 5.3 MB at 1080p.

## Meshlet level

Off by default; acts only while occlusion culling and meshlet culling are both on and the bitset fits (see Limits).
Meshlet batches then also test each meshlet in the task shader (`run_meshlet_task`, `assets/shaders/meshlet_task.slang`)
against the same pyramids, with the world bounding sphere (`sphere_occluded`, the box of centre +- radius; radius
already includes the max axis scale and wind padding). Instanced (vertex-shader) batches have no task shader and stay
instance level.

**Why forward replays.** Today forward re-runs the prepass's per-meshlet frustum and cone tests, a pure function of
the same inputs, so it draws exactly the prepass winners. An occlusion test is not: the two prepass phases test
against different pyramids and forward has no pyramid that reproduces both. A re-test could skip a meshlet that won
the prepass (a hole showing the HDR clear colour). So each prepass phase **records** a bit per meshlet it emits and
forward **replays** the bits.

**Bitset.** One bit per meshlet of every opaque and mask meshlet-path instance (`MeshletVisibilityLayout`,
`include/rendering/meshlet_visibility.hxx`). `emit_batch` reserves a range per batch and stores each instance's first
bit in `GpuDraw::meshlet_visibility_offset`; the task shader addresses word `bit >> 5`, bit `bit & 31`. The buffer
lives in `RendererFrame::meshlet_visibility_buffer`, grows to a power-of-two size, is zeroed every frame in
`prepare_frame`, and is reached through `OcclusionView::meshlet_visibility` (both views carry the same address).

**Flags per pass** (`render_pass::cull_*`, mirrored by `cull_*_bit` in `scene_types.slang`; `extra_cull_flags` is
OR-ed into the opaque and mask draws, with `occlusion_view_address` as `PC::occlusion`):

| Pass | View | Extra flags | Task shader does |
|---|---|---|---|
| early prepass | [0] history | `occlusion \| record \| stats` | frustum, cone, then Hi-Z test; sets the bit of each emitted meshlet |
| late prepass | [1] this frame | `occlusion \| skip_recorded \| record \| stats` | skips meshlets whose bit is set, else frustum, cone, Hi-Z test; sets the bit |
| forward | [1] (bitset only) | `replay` | emits exactly the meshlets whose bit is set, no other test |
| shadows | none | none | unchanged (never dereferences `PC::occlusion`) |

Blend draws never take the bits. With no history (first frame, resize, toggle) `sphere_occluded` returns false, so the
early phase records every frustum- and cone-visible meshlet and the late phase has nothing to add.

**Late command.** `late_cs` gives a meshlet batch the union command `(first, n1 + n2)` instead of `(first + n1, n2)`
(`CullPC::late_union_meshlet_batches = 1` while meshlet occlusion is active). The late prepass then runs the task
shader for phase 1's instances too, which is how their meshlets the history test deferred get retested against this
frame's pyramid, and `skip_recorded` keeps the ones phase 1 already drew from drawing twice. Instanced batches keep
`(first + n1, n2)`.

**Statistics.** Each view's `stats_address` points at its own counter, so the task shader always adds to
`pc.occlusion->stats[0]`: view [0] counts meshlets phase 1 deferred (slot 4), view [1] the ones phase 2 culled for
good (slot 5). `FrameStats::deferred_meshlet_count` / `occluded_meshlet_count`, shown in Scene stats.

**Barriers.** The clear is a `vkCmdFillBuffer` plus a transfer -> task-shader barrier in `prepare_frame`.
`record_meshlet_visibility_barrier` (task write -> task read | write) runs before the late prepass, and (task write ->
task read) before forward. The statistics readback barrier includes the task-shader stage as a source.

**Limits.** The bitset is capped at 2^28 bits (32 MiB per frame in flight); a frame needing more runs without meshlet
occlusion (instance level continues) and the renderer warns once. Phase 1's instances of meshlet batches run their
task shaders a second time in the late prepass; measure that against the saving before enabling it by default (if it
dominates, only add phase-1 instances to the union when phase 1 deferred any of their meshlets).

## Toggles

- `Renderer::set_occlusion_culling(bool)` / `--occlusion-culling=on|off` / Lighting > "Occlusion culling (Hi-Z,
  two-phase)" (disabled with a tooltip when unsupported). The benchmark JSON records `occlusion_culling`.
- `Renderer::set_occlusion_test_mode(OcclusionTestMode)` / Lighting > "Occlusion test": `hiz` (real), or the debug
  stubs `never_occluded` (everything in phase 1) and `always_defer` (view [0] reports every opaque/mask instance
  occluded, view [1] none, so phase 2 draws everything). Either stub must render exactly like culling off.
- `Renderer::set_meshlet_occlusion_culling(bool)` / `--meshlet-occlusion=on|off` / Lighting > "Meshlet occlusion
  (task shader)" (under the occlusion checkbox, disabled while meshlet culling is off). It acts only while occlusion
  culling and meshlet culling are on. The benchmark JSON records `meshlet_occlusion` (true only when it was active).
- Lighting > "Hi-Z mip": shows one pyramid level (red = farthest depth, brighter is nearer), cropped to its logical
  extent. Needs linear filtering of R32_SFLOAT; otherwise `hiz_debug_view()` returns an invalid handle.
- Scene stats: frustum-culled, occluded (count and % of frustum-visible), phase 1 / phase 2 counts, and with meshlet
  occlusion the meshlets occluded (final) and deferred by phase 1. Like the rest of the GPU stats they lag a
  frames-in-flight cycle.

## Debugging checklist

- **Holes showing the HDR clear colour** (dark blue): forward missed a prepass winner. Switch to the
  `always_defer` stub: if holes stay, the draw lists (late/merged commands, compaction offsets) are wrong; if they
  go, look at phase 2's test. Compare `merged_indirect` against `culled_indirect` + `late_indirect` in RenderDoc.
- **Popping or missing objects with the real test**: freeze the camera; phase 2 must be ~0 when parked. Check the
  Hi-Z mip view against the depth (Y flip, mip edges), and that `hiz_.depth_extent` equals the render extent.
- **Holes only with meshlet occlusion on**: forward replayed a bitset that lacks a prepass winner. Check that the
  bitset is cleared in `prepare_frame`, the barrier before the late phase and before forward is recorded, and that
  forward's view address is view [1] (the same bitset both phases wrote). With the `always_defer` stub the early
  phase records nothing and the late phase records everything, which isolates the late phase.
- **Doubled draws or counts with meshlet occlusion**: the late phase lacks `cull_skip_recorded_bit`, or a batch's late
  command isn't the union `(first, n1 + n2)` (`CullPC::late_union_meshlet_batches`).
- **SIGSEGV in `spvtools::opt::blockmergeutil::MergeWithSuccessor` at startup** (`spirv_opt::run` in
  `slang_compiler.cxx`): a SPIRV-Tools optimizer bug on some control flow, hit by the runtime pipeline (Slang
  `-O3`, then `RegisterPerformancePasses`) but not by CI, which only runs `slangc` for push-constant reflection.
  `hiz_build.slang` hit it with nested `[unroll]`ed conditional accumulation; it now fetches clamped, branch-free.
  To check shaders the way the runtime does: `slangc <file> -entry <e> -stage <s> -target spirv -profile spirv_1_6
  -matrix-layout-column-major -force-glsl-scalar-layout -O3 -g2 -emit-spirv-directly -fvk-use-entrypoint-name`, then
  `spirv-opt -O --skip-validation` (build SPIRV-Tools at the tag in the `Dockerfile`, `SPIRV_TOOLS_TAG`).
- **Timings panel empty** after a change: some stage skipped a timestamp.
- **Unavailable checkbox**: the device lacks MIN depth resolve under MSAA (logged at startup as "Depth resolve MIN
  support").
- The CPU tests in `test/hiz_occlusion_test.cxx` cover the projection, level choice, pyramid layout, a
  conservativeness property and the sphere test; when changing the shader test, change `hiz_occlusion.hxx` too and
  run them. `test/meshlet_visibility_test.cxx` covers the bitset layout and its cap.

## Verifying on a GPU

None of this has been run on a GPU yet. Before flipping the default:

1. **Screenshot equality.** Run the benchmark path with and without culling and compare the keyframe
   screenshots (`screenshots/screenshot_<timestamp>_<n>.png`, in order); they must be bit-identical:
   ```
   ./lathe --benchmark=perf/off.json --benchmark-screenshots --occlusion-culling=off
   ./lathe --benchmark=perf/on.json  --benchmark-screenshots --occlusion-culling=on
   ```
   Repeat with each stub mode (select it in the UI, or temporarily change `occlusion_test_mode_`). The only expected
   difference source is a z-tie between two instances of one batch (see Design).
   Then add `--meshlet-occlusion=on` to the `=on` run: its screenshots must be bit-identical too, and so must both
   stub modes with it on.
2. **Synchronisation validation**: a Debug build (validation layers on) with synchronisation validation enabled
   (vkconfig, or the `validate_sync` layer setting) for ~300 frames, including a viewport resize and toggling culling
   and the test modes on and off. Expect no errors; in particular check the resolve-related barriers.
3. **RenderDoc** (`ENABLE_RENDERDOC`, `cmake/renderdoc_config.cmake`):
   - Event order: `main_cs` -> shadows -> prepass (CLEAR, MIN resolve) -> N Hi-Z dispatches -> `late_cs` -> late
     prepass (LOAD, SAMPLE_ZERO resolve) -> GTAO -> forward.
   - Hi-Z level 0 is the 2x2 MIN of the early resolved depth; odd-sized level edges are right; the last level is 1x1.
   - Buffers: `culled_indirect` (n1), `candidate_counts` (c), `late_indirect` (first + n1, n2), `merged_indirect`
     (n1 + n2), with `n1 + c <= instance_count` and `n2 <= c`.
   - Pixel history on a pixel drawn by the late prepass: forward's `EQUAL` test passes; no clear colour in forward.
   - With meshlet occlusion: `meshlet_visibility_buffer` after the early prepass has bits only for meshlets phase 1
     drew; after the late prepass it is a superset; forward's task invocations emit exactly the set bits.
4. **Steady state**: parked camera -> phase 2 instances ~0; occluded instances well above 0 on the skull pile.
   With meshlet occlusion on, forward task/mesh invocations (Scene stats) drop versus instance-level only on the
   pile, the occluded meshlet count is non-zero there and ~0 in open terrain.
5. **Benchmark** (see `docs/perf-benchmark.md`): `--benchmark-frames=3000 --benchmark-warmup=300`, on vs off, and
   compare the sum of `depth_prepass + hiz_build + occlusion_culling + depth_prepass_late + forward_pass`, plus the
   full frame. Expect a cost when little is occluded (extra pass, second MSAA resolve, Hi-Z build).

## Code map

- `include/rendering/hiz_occlusion.hxx`: CPU mirror (extents, projection, level choice); `test/hiz_occlusion_test.cxx`.
- `include/rendering/meshlet_visibility.hxx`: bitset layout, offsets and cap; `test/meshlet_visibility_test.cxx`.
- `assets/shaders/hiz_occlusion.slang` (`OcclusionView`, `aabb_occluded`, `sphere_occluded`),
  `assets/shaders/hiz_build.slang`, `assets/shaders/frustum_cull.slang` (`CullPC`, `workgroup_compact`, `main_cs`,
  `late_cs`).
- `include/rendering/renderer.hxx`: `OcclusionTestMode`, `FrameStats` occlusion fields, `GpuOcclusionView`,
  `occlusion_stat_*`, `RendererFrame` occlusion buffers, `HizPyramid`, `hiz_*` members.
- `src/rendering/renderer.cxx`: `occlusion_view_states`, pipelines 21 (`late_cs`) and 22 (`hiz_build`),
  `create_hiz_pyramid`, `prepare_frame` (stats clear, views, `main_cs`), `record_frame`, `record_hiz_build`,
  `record_occlusion_cull_pass`, `record_occlusion_stats_readback`, `consume_culled_readback`,
  `early_/late_/forward_view_draws`.
- `include/rendering/render_passes.hxx` / `src/rendering/render_passes.cxx`: `DepthPrepassPhase`,
  `DepthPrepassInfo::depth_resolve_mode`, `HizBuildInfo`, `build_hiz`.
- Meshlet level: `assets/shaders/meshlet_task.slang` (`run_meshlet_task`), `assets/shaders/scene_types.slang`
  (`cull_*_bit`, `GpuDraw::meshlet_visibility_offset`, `PC::occlusion`), `emit_batch` and the meshlet block in
  `prepare_frame`, `record_meshlet_visibility_barrier`, and `render_pass::cull_*` with
  `DepthPrepassInfo`/`ForwardGeometryInfo::occlusion_view_address` / `extra_cull_flags`.
- `src/app/application.cxx` (Lighting and Scene stats panels), `src/main.cxx` (`--occlusion-culling`,
  `--meshlet-occlusion`),
  `src/vulkan_bootstrap.cxx` (`depth_resolve_min_supported`), `include/rendering/render_stage.hxx`.
