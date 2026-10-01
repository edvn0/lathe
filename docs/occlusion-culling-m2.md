# M2 spec: meshlet-level occlusion culling

Implementation spec for the next step of `docs/occlusion-culling.md` (read its Design and Frame order sections
first). Self-contained: it names every symbol and file to touch. Line numbers are as of commit `85a45e9` (M1b) and
drift; find symbols by name.

## Goal and non-goals

**Goal.** In the task shader (`assets/shaders/meshlet_task.slang`), cull individual meshlets of meshlet-path batches
against the Hi-Z, in the same two phases as the instance test, while keeping the forward pass's `EQUAL` depth test
exact. Behind its own toggle (`Renderer::meshlet_occlusion_culling()`, default off), only active while instance-level
occlusion is active and meshlet culling is on.

**Non-goals.** Instanced (vertex-shader) batches (no task shader: instance level only); blend batches (never in the
prepass); shadows; a second Hi-Z rebuild after phase 2; single-pass downsampling; any change to the instance-level
test or the pyramid.

## Why forward must replay

Today forward re-runs the prepass's per-meshlet frustum/cone tests, which are a pure function of the same inputs, so
it draws exactly the prepass winners. An occlusion test is not: phase 1 and phase 2 test against different pyramids,
and forward has no pyramid that reproduces both decisions. If forward re-tested, it could skip a meshlet that won the
prepass (a hole showing the HDR clear colour) or draw one the prepass skipped (harmless, but wasted). So each prepass
phase **records** a bit per meshlet it emits, and forward **replays** the bits instead of testing.

## What M1b already provides

| What | Where |
|---|---|
| `OcclusionView` (112 B) with the reserved `Ptr<uint> meshlet_visibility` and `Ptr<uint> stats`, `aabb_occluded()`, `sphere_occluded(view, centre, radius)` | `assets/shaders/hiz_occlusion.slang` (`sphere_occluded` at line 116) |
| `GpuOcclusionView` mirror with `meshlet_visibility_address` (offset 64, always 0 today) and `stats_address` | `include/rendering/renderer.hxx:667` |
| Two views per frame in `occlusion_views_buffer`: [0] history (phase 1), [1] this frame (phase 2); written in `prepare_frame` | `src/rendering/renderer.cxx:2929` |
| `occlusion_view_states()` (disabled / enabled / force-occluded per view) | `src/rendering/renderer.cxx:238` |
| `CullPC::late_union_meshlet_batches`: when non-zero, `late_cs` gives meshlet batches the late command `(first, n1 + n2)` instead of `(first + n1, n2)`; C++ passes 0 | `assets/shaders/frustum_cull.slang:313`; push constants in `prepare_frame` (`renderer.cxx:3049`) and `record_occlusion_cull_pass` (`renderer.cxx:3856`) |
| Statistics buffer, 8 x u32, slots 4-5 reserved for meshlets; cleared in `prepare_frame` (`renderer.cxx:2709`), copied by `record_occlusion_stats_readback` (`renderer.cxx:3909`), read by `consume_culled_readback` (`renderer.cxx:3376`) | `occlusion_stat_*` in `renderer.hxx:694` |
| Prepass phases and per-phase draw buffers: `record_depth_prepass(..., DepthPrepassPhase)`, `early_view_draws` / `late_view_draws` / `forward_view_draws` | `renderer.cxx:3702`, `renderer.cxx:3605-3630` |
| `frame.occlusion_active`, decided in `prepare_frame` (`renderer.cxx:2921`) | `RendererFrame`, `renderer.hxx:783` |
| Pyramid barriers already include `TASK_SHADER` as a reader stage | `render_pass::build_hiz`, `src/rendering/render_passes.cxx:810` |
| CPU mirror of the test and its tests | `include/rendering/hiz_occlusion.hxx`, `test/hiz_occlusion_test.cxx` |

## Data changes

### Cull bits (`assets/shaders/scene_types.slang:51-54`, mirrored in `render_pass::detail`, `render_passes.cxx:543`)

```
cull_occlusion_bit     = 4u   // test the meshlet's sphere against *pc.occlusion
cull_skip_recorded_bit = 8u   // phase 2: skip meshlets phase 1 already drew
cull_record_bit        = 16u  // set the meshlet's bit when it is emitted
cull_replay_bit        = 32u  // forward: emit exactly the set bits, no other test
cull_stats_bit         = 64u  // count occlusion rejections into pc.occlusion->stats[0]
```

C++: `detail::cull_occlusion = 4U`, `cull_skip_recorded = 8U`, `cull_record = 16U`, `cull_replay = 32U`,
`cull_stats = 64U` next to `cull_frustum` / `cull_frustum_and_backface`.

### `GpuDraw::transform_index` -> `meshlet_visibility_offset`

`scene_types.slang:63` and `Renderer::GpuDraw` (`renderer.hxx:645`). No shader reads `transform_index` today and it
always equals the source instance index, so the rename is free. New meaning: the absolute bit index of the
instance's meshlet 0 in this frame's bitset; 0 for instanced batches (never read).

### `PC::occlusion` (`scene_types.slang:193-222`)

Add `public Ptr<OcclusionView> occlusion;` after `cluster_lights`, with `import hiz_occlusion;` at the top of
`scene_types.slang` (`hiz_occlusion` imports only `bindless`, so there is no cycle). `PC` grows 96 -> 104 bytes and
`ShadowPC` 104 -> 112; the reflected `ForwardPushConstants` and `ShadowPushConstants` gain `occlusion_address`
after `cluster_lights_address`. The `ShadowPushConstants` designated initializers in `render_pass::shadow`
(`render_passes.cxx:645`, `677`) stay valid because the new field sits after `cull_flags` and before
`cascade_index`; leave it 0 there. Shadows never set occlusion bits, so they never dereference it.

### `RendererFrame` (`renderer.hxx`, next to the other occlusion buffers at line ~769)

```cpp
Buffer meshlet_visibility_buffer{};                // device, STORAGE | SHADER_DEVICE_ADDRESS | TRANSFER_DST
std::uint32_t meshlet_visibility_capacity_words = 0;
std::uint32_t meshlet_visibility_words = 0;        // this frame's bits, rounded up to 32
bool meshlet_occlusion_active = false;             // decided in prepare_frame, followed by record_frame
```

Destroy it in `Renderer::destroy` with the other occlusion buffers.

### Renderer API and state (`renderer.hxx`, next to `occlusion_culling()`)

```cpp
[[nodiscard]] auto meshlet_occlusion_culling() const noexcept -> bool { return meshlet_occlusion_culling_; }
auto set_meshlet_occlusion_culling(bool enabled) noexcept -> void { meshlet_occlusion_culling_ = enabled; }
...
bool meshlet_occlusion_culling_ = false;            // next to occlusion_culling_
bool meshlet_visibility_cap_warned_ = false;
static constexpr std::uint32_t maximum_meshlet_visibility_bits = 1U << 28U;   // 32 MiB of bits
```

### `FrameStats` (`renderer.hxx:110`)

```cpp
std::uint32_t deferred_meshlet_count = 0;  // passed frustum + cone in phase 1, rejected by the history test
std::uint32_t occluded_meshlet_count = 0;  // rejected by phase 2's test: culled for good
bool meshlet_occlusion_stats_valid = false;
```

Fill them in `consume_culled_readback` from slots 4 and 5; record `frame.meshlet_occlusion_active` next to
`occlusion_stats_active` in `record_occlusion_stats_readback` so validity follows the recorded frame.

### Statistics slots

Add `occlusion_stat_deferred_meshlets = 4` and `occlusion_stat_occluded_meshlets = 5` (C++ in `renderer.hxx`, the
Slang comment in `frustum_cull.slang`). Point each view's `stats_address` at **its slot** rather than the buffer
base: view [0] `= stats + 4 * 4`, view [1] `= stats + 5 * 4`. The task shader then always adds to
`pc.occlusion->stats[0]` and needs no phase flag. (`main_cs`/`late_cs` use `CullPC::occlusion_stats`, not the
view's pointer, so they are unaffected.)

## CPU changes (`src/rendering/renderer.cxx`)

### Bit offsets in `emit_batch` (line 2469; the `GpuDraw` push at 2502)

Keep a running `std::uint64_t meshlet_visibility_bits = 0` captured by reference. In `emit_batch`, for
`uses_meshlet_path(geometry.meshlets.meshlet_count)` batches:

```cpp
.meshlet_visibility_offset = meshlet_path ? static_cast<std::uint32_t>(meshlet_visibility_bits + instance * meshlet_count) : 0U,
...
if (meshlet_path) { meshlet_visibility_bits += std::uint64_t{instance_count} * meshlet_count; }
```

Compute the offsets in 64 bits and only narrow when the total is within the cap (below). Batches are emitted
opaque, mask, blend; blend meshlet batches also get offsets (harmless, they are never recorded) unless you skip
them, which saves bits: only add for opaque and mask batches (`emit_batch` can take the alpha mode, or count only
the first `opaque + mask` batches).

### Decide, grow and clear (`prepare_frame`)

After `frame.indirect_command_count` is set (line 2693) and before `upload_frame_data`:

```cpp
auto const bits_fit = meshlet_visibility_bits <= maximum_meshlet_visibility_bits;
if (!bits_fit && !meshlet_visibility_cap_warned_) { warn(...); meshlet_visibility_cap_warned_ = true; }
// frame.occlusion_active is decided later; decide this flag right after it (line 2921):
frame.meshlet_occlusion_active = frame.occlusion_active && meshlet_occlusion_culling_ && meshlet_culling_ && bits_fit
                                 && meshlet_visibility_bits != 0;
frame.meshlet_visibility_words = (bits + 31) / 32;
```

Grow when `meshlet_visibility_words > meshlet_visibility_capacity_words`: create a new buffer of
`std::bit_ceil(words) * 4` bytes and move it in (safe: this slot's fence has been waited on, as in
`prepare_cluster_buffers`, `renderer.cxx:3499`). The upload / `GpuDraw` offsets are written before the decision is
known, which is fine: offsets are ignored unless the bits are used.

Clear next to the statistics clear (line 2709) when `frame.meshlet_occlusion_active`:
`vkCmdFillBuffer(meshlet_visibility, 0, words * 4, 0)` plus a buffer barrier `ALL_TRANSFER / TRANSFER_WRITE ->
TASK_SHADER / SHADER_STORAGE_READ | SHADER_STORAGE_WRITE`. Since the clear has to come after the decision, either
move the decision of `occlusion_active` up next to the statistics clear or move the clear down after the view
upload; both are in `prepare_frame` before any draw.

Write `meshlet_visibility_address = frame.meshlet_occlusion_active ? buffer.device_address : 0` into **both**
views (line 2929).

### Late command union

Pass `.late_union_meshlet_batches = frame.meshlet_occlusion_active ? 1U : 0U` in `record_occlusion_cull_pass`'s
`CullPushConstants` (line 3856). Leave `main_cs`'s at 0 (`main_cs` ignores it). Phase-1 instances of meshlet
batches then run task shaders again in the late prepass, so their deferred meshlets get phase 2's test.

### Barriers for the bitset (`record_frame`, around line 4434)

- After the early prepass, before the late prepass: `TASK_SHADER / SHADER_STORAGE_WRITE -> TASK_SHADER /
  SHADER_STORAGE_READ | SHADER_STORAGE_WRITE` on `meshlet_visibility_buffer`. Simplest place: the start of
  `record_occlusion_cull_pass`'s pre-barrier, or a helper called after `record_hiz_build`.
- After the late prepass, before forward: `TASK_SHADER / SHADER_STORAGE_WRITE -> TASK_SHADER /
  SHADER_STORAGE_READ`.
- Statistics: the task shaders' atomics on slots 4-5 must be visible to the readback copy: add `TASK_SHADER` to the
  source stages of the barrier at the top of `record_occlusion_stats_readback`, which already runs after the late
  prepass.

### `cull_flags` per pass

Add to `DepthPrepassInfo` and `ForwardGeometryInfo` (`include/rendering/render_passes.hxx:102`, `162`):

```cpp
VkDeviceAddress occlusion_view_address = 0;  // PC::occlusion
std::uint32_t extra_cull_flags = 0;          // OR-ed into the opaque and mask cull_flags when meshlet_culling
```

and set `.occlusion_address = info.occlusion_view_address` in the `ForwardPushConstants` built at
`render_passes.cxx:774` (prepass) and `:1088` (forward).

| Pass | View | Opaque flags | Mask flags | Blend |
|---|---|---|---|---|
| early prepass | [0] | `frustum \| backface \| occlusion \| record \| stats` | `frustum \| occlusion \| record \| stats` | n/a |
| late prepass | [1] | `frustum \| backface \| occlusion \| skip_recorded \| record \| stats` | same without `backface` | n/a |
| forward | [1] (only its bitset is used) | `replay` (frustum/backface bits may stay set; replay ignores them) | `replay` | unchanged: `frustum`, no occlusion |
| shadows | none | unchanged | unchanged | n/a |

So `extra_cull_flags` is `occlusion | record | stats` (early), `occlusion | skip_recorded | record | stats` (late)
and `replay` (forward), and 0 when `!frame.meshlet_occlusion_active`. In `forward_geometry` the mask and blend draws
share `unculled_backface_pc` (`render_passes.cxx:1108`); split it so only mask gets `extra_cull_flags`. Set them
in `record_depth_prepass` / `record_forward_pass` from `frame.meshlet_occlusion_active` with view addresses
`occlusion_views_buffer.device_address` (+ `sizeof(GpuOcclusionView)` for [1]).

Why phase 2 re-tests with `occlusion` against view [1]: its pyramid holds only phase-1 depth, a subset of the final
depth, so a meshlet it rejects is hidden in the final image; phase 1's rejections (history) are only deferrals.

## Shader changes (`assets/shaders/meshlet_task.slang`)

`import hiz_occlusion;`. Split the bounds out of `meshlet_visible` (line 30) so the occlusion test reuses the world
sphere (`centre`, `radius` already include `max_scale` and the wind padding):

```
bool meshlet_visible(PC pc, uint plane_offset, uint instance_index, uint meshlet_index,
                     out float3 centre, out float radius)      // frustum + cone, as today
```

In `run_meshlet_task` (line 77), replace the visibility computation (line 92):

```
bool visible = group_valid && meshlet_index < command.meshlet_count;
bool occluded = false;
const uint flags = pc.cull_flags;

if (visible && (flags & (cull_replay_bit | cull_skip_recorded_bit | cull_record_bit)) != 0) {
    bit = pc.draws[instance_index].meshlet_visibility_offset + meshlet_index;
    bits = pc.occlusion[0].meshlet_visibility;            // non-null whenever these flags are set
}

if (visible && (flags & cull_replay_bit) != 0) {
    visible = (bits[bit >> 5] & (1u << (bit & 31))) != 0;  // forward: exactly what a prepass phase drew
} else if (visible) {
    if ((flags & cull_skip_recorded_bit) != 0 && (bits[bit >> 5] & (1u << (bit & 31))) != 0) {
        visible = false;                                   // phase 1 drew it; don't draw or count it twice
    }
    float3 centre; float radius;
    if (visible) { visible = meshlet_visible(pc, plane_offset, instance_index, meshlet_index, centre, radius); }
    if (visible && (flags & cull_occlusion_bit) != 0 && sphere_occluded(pc.occlusion[0], centre, radius)) {
        visible = false; occluded = true;
    }
    if (visible && (flags & cull_record_bit) != 0) {
        InterlockedOr(bits[bit >> 5], 1u << (bit & 31));
    }
}

if ((flags & cull_stats_bit) != 0) {
    uint occluded_count = WaveActiveCountBits(occluded);   // meshlets_per_task = 32 lanes; see note
    if (lane == 0 && occluded_count != 0) { InterlockedAdd(pc.occlusion[0].stats[0], occluded_count); }
}
```

Notes: the stats reduction must cover the whole workgroup; with a 32-lane group and wave size >= 32 one
`WaveActiveCountBits` suffices, otherwise reduce through a `groupshared` counter (the existing `s_visible` array
works: count `occluded` into a second groupshared array before the existing barrier). Atomics on `Ptr<uint>`
elements already work (`light_cluster.slang:222`). Never dereference `pc.occlusion` unless a flag needs it (the
shadow pass and M1 paths pass 0). `sphere_occluded` returns false for a disabled view, so phase 1 without history
records every frustum/cone-visible meshlet; the `always_defer` / `never_occluded` stubs stay exact.

## UI, CLI, benchmark

- Lighting panel (`src/app/application.cxx`, under "Occlusion culling (Hi-Z, two-phase)", line 2139): checkbox
  "Meshlet occlusion (task shader)", inside `BeginDisabled(!(meshlet_culling && occlusion_culling))`.
- Scene stats (after "Instances occluded (Hi-Z)", line 2034): `"Meshlets occluded: %u final (%u deferred by phase
  1)"` when `meshlet_occlusion_stats_valid`.
- `src/main.cxx` (next to `--occlusion-culling`, line 592): `--meshlet-occlusion=on|off`, same parsing; warn if on
  while occlusion culling is off.
- `BenchmarkEnvironment` (`include/app/benchmark.hxx:69`): `bool meshlet_occlusion = false;`, written as
  `"meshlet_occlusion"` in `BenchmarkRun::to_json` (`src/app/benchmark.cxx:234`) and set in `main.cxx` (line 761)
  from the toggle and `occlusion_culling_supported()`.

## Tests

- Move the offset arithmetic into a small header (e.g. `include/rendering/meshlet_visibility.hxx`:
  `meshlet_visibility_word_count(bits)`, a `MeshletVisibilityLayout` that hands out offsets per batch and reports
  `fits()` against the cap) and test it in a new `test/meshlet_visibility_test.cxx` (picked up by the glob in
  `test/CMakeLists.txt`): offsets of consecutive batches are contiguous and disjoint, instanced batches get none,
  the word count rounds up, the cap trips at `2^28 + 1` bits.
- Extend `test/hiz_occlusion_test.cxx` with a sphere case: `sphere_occluded` is `aabb_occluded` of
  `centre +- radius` (add a `sphere_occluded_reference` next to `aabb_occluded_reference`).
- `test/benchmark_test.cxx`: `"meshlet_occlusion": false/true` in the JSON.
- Keep every existing test unchanged.

## Manual GPU verification (CI has no GPU)

1. Screenshots: `--benchmark-screenshots` runs with `--occlusion-culling=off`, `=on`, and `=on
   --meshlet-occlusion=on`; keyframe screenshots must be bit-identical (same caveat about z-ties as in
   `docs/occlusion-culling.md`).
2. Stubs: with meshlet occlusion on, both stub modes must still match culling off.
3. Sync validation for ~300 frames with resize and toggles: no hazards on `meshlet_visibility_buffer`.
4. RenderDoc: the bitset after the early prepass has bits only for phase-1 instances; after the late prepass it is
   a superset; forward's task invocations emit exactly the set bits (pixel history on a late-drawn meshlet passes
   `EQUAL`; no HDR clear colour anywhere in forward).
5. Scene stats: forward task/mesh invocations drop versus instance-level only on the skull pile; the occluded
   meshlet count is non-zero there and ~0 in open terrain.
6. Benchmark on vs off (`docs/perf-benchmark.md`), comparing `depth_prepass + hiz_build + occlusion_culling +
   depth_prepass_late + forward_pass`.

## Risks

1. **Holes** if forward ever re-tests instead of replaying, or replays with a pointer to a different bitset:
   forward must use `replay` for every opaque and mask meshlet draw while meshlet occlusion is active.
2. **Double draws / double counts** if phase 2 lacks `skip_recorded` on the union range.
3. **Bitset size**: `sum(instances x meshlets)` over opaque + mask meshlet batches; over the cap, meshlet occlusion
   is off for that frame (instance level continues), warn once.
4. **Cost**: phase-1 instances of meshlet batches run task shaders twice (union range). Measure; if it dominates,
   only add phase-1 instances to the union when phase 1 deferred any of their meshlets (would need a per-instance
   flag written by the early task shader).
5. **Layout churn**: `PC` grows; the reflected structs and `static_assert`s regenerate at build time; keep the
   shadow designated initializers in declaration order.
6. **Atomic throughput** on the bitset: one `InterlockedOr` per emitted meshlet per phase; consider a wave-level OR
   per word if profiling shows contention.

## Ordered checklist

1. Cull bits in `scene_types.slang` and `render_pass::detail`; `GpuDraw` rename on both sides; `PC::occlusion`.
   Build: reflection regenerates; fix the push-constant initializers.
2. `meshlet_visibility.hxx` + tests; offsets in `emit_batch`.
3. `RendererFrame` buffer, growth, clear, `meshlet_occlusion_active`, view `meshlet_visibility_address` and
   per-slot `stats_address`, `late_union_meshlet_batches`.
4. `DepthPrepassInfo` / `ForwardGeometryInfo` fields; flags per pass; split forward's mask and blend push constants.
5. `meshlet_task.slang` changes.
6. Bitset and statistics barriers in `record_frame` / `record_occlusion_stats_readback`.
7. `FrameStats`, readback, UI, CLI, benchmark JSON.
8. Docs: a "Meshlet level" section in `docs/occlusion-culling.md` and the task-shader section of
   `docs/meshlet-rendering.md`.
9. `cargo xtask build`, `cargo xtask test`, `cargo xtask tidy`, then the GPU checks above.

**Done when**: all CI jobs pass; the toggle defaults off; with it on, the screenshots and stub checks are
bit-identical to off, sync validation is clean, and forward task/mesh invocations drop where things are occluded.

**Effort**: about 3-4 engineer-days (1 for plumbing and flags, 1 for the shader and barriers, 1-2 for GPU
verification and tuning).
