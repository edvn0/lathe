# Lathe GPU performance brief — from Nsight GPU Traces (2026-10-18)

Audience: a coding agent working in the Lathe repo. These findings come from six Nsight Graphics GPU Trace captures. Fix the items in priority order. After each fix, re-capture with the same settings and compare against the baselines below.

## Capture context

- GPU: RTX 4070 Ti SUPER (AD103, 66 SMs), driver 615.71.09, Debian 13, KDE Wayland.
- Nsight Graphics 2026.2, GPU Trace, Top-Level Triage metric set, warp-state sampling on, clocks locked to base, vsync off.
- Build: `build/release/bin/lathe`
- Renderer: Vulkan with `VkShaderEXT` shader objects and Slang shaders (entry points `main_task`, `main_mesh`, `main_vs`, `main_fs`, `main_cs`, `late_cs`). Two queues: graphics (family 0) and async compute (family 2). The frame graph puts roughly 7 graphics batches and 5 compute batches into each frame.

| Scene | Command line | GPU time per frame (base clocks) |
|---|---|---|
| rt-poc | `--rt-poc` (render size 1533x1015) | ~2.1 ms |
| shadowcity | `--game=shadowcity` (~1035 entities, dozens of shadowed point and spot lights) | ~30–60 ms |

## Frame structure (same in both scenes)

Graphics-queue command buffers are labelled G and async-compute ones C, in submission order:

1. G: culling and setup. Buffer copies and fills, plus small dispatches.
2. G: depth, phase 1. Indirect mesh-task draws plus an indexed-indirect fallback, both reading the same 40-byte draw records. Takes ~300 µs in shadowcity.
3. C: Hi-Z mip chain, about 26 dispatches.
4. G: a pass that starts with `vkCmdClearAttachments`. This is the only graphics pass that overlaps compute today.
5. G: culled draws. The task shader reads Hi-Z. Writes an image (color + depth attachment) that is handed to compute.
6. C: two full-resolution dispatches (8x8 groups). They read that image, write an intermediate, then write a result image that is handed back to graphics. Takes 336 µs in rt-poc and ~540 µs in shadowcity.
7. G: the **main lit pass**. Indirect mesh draws using `main_task` + `main_mesh` (72 registers, ~7 KB of machine code) + `main_fs` (80 registers, ~35 KB). It reads the compute result in the fragment shader, then draws particles (`vkCmdDraw`, 49152 vertices). This one pass is **97% of the frame in shadowcity**.
8. G: bloom (a compute downsample/upsample chain), a fullscreen composite, and ImGui.
9. rt-poc only: a ray-query compute dispatch on the async queue (~116 µs). It is limited by the ray-tracing hardware, which is expected.

---

## P0 — The lit-pass mesh shader is starved on memory (the main problem)

### Evidence

- **shadowcity:** the lit-pass `main_mesh` gets **80% of all GPU warp samples**, and the lit-pass `main_fs` gets another 18%.
  - In `main_mesh`, ~99% of samples are stalled on long scoreboard, meaning the warps are waiting on memory loads. Only ~0.2% of samples are issuing instructions.
- **Same geometry, cheap shader:** the depth pass draws the same 38 draw records with its own mesh shader (70 registers) in **0.27 ms**, issuing instructions ~10% of the time.
  - So the cost comes from what the lit-pass mesh shader reads or computes that the depth pass's mesh shader doesn't.
- **rt-poc:** the same lit-pass `main_mesh` costs ~470 µs per frame, against ~42 µs for the depth pass's mesh shader (11x). In shadowcity it is ~200x.
  - The cost grows sharply with scene complexity, and the obvious difference is the number of lights.
- **It's mostly one instruction.** At byte offset `0x6e0` in the lit-pass `main_mesh` machine code sits **68%** of that shader's samples. Adding offsets `0x660` and `0x610` brings it to ~83%. All three are long-scoreboard stalls, i.e. waiting on the result of an earlier load.
- **The fragment shader is starved too.** In shadowcity the lit-pass `main_fs` is 84% long-scoreboard with ~1.6% instruction issue. In rt-poc it issues instructions ~12% of the time.
  - So the data the lit pass reads gets slow in both stages as the scene grows.

### Hypotheses (most likely first)

1. **The lit-pass mesh or task shader does per-light work per vertex or per meshlet.** For example, computing shadow-map coordinates for every shadowed light, or looping over the light buffer, with each iteration waiting on a load.
2. **The light, shadow-matrix, instance, material or vertex-attribute buffers read by the lit pass aren't device-local memory.** For example `HOST_VISIBLE | HOST_COHERENT` without `DEVICE_LOCAL`, or an allocator fallback when VRAM was full. Every read would then cross PCIe. Host RAM was 92% used during the shadowcity captures.
3. **Less likely: global atomics in the mesh shader**, such as stats counters or append buffers, contending on a few addresses.

### Tasks

1. Find the lit pass's task, mesh and fragment shaders: the pipeline bound before the indirect mesh draws in the lit-pass frame graph node, i.e. the pass that samples the async-compute output.
2. **Diff the lit-pass `main_mesh` against the depth pass's `main_mesh`.** List every buffer read, every loop over lights, shadows or instances, and every atomic. Anything per-light in the mesh or task stage is the prime suspect.
3. **Move per-light work out of the mesh stage.**
   - Output only what the rasterizer needs: position, normal, tangent, uv, world position, instance or material id.
   - Do light and shadow evaluation in the fragment shader with a bounded light list, either clustered/tiled or a per-meshlet light mask computed once in the task shader.
   - Shadow coordinates should be computed per fragment, and only for lights affecting that cluster.
4. **Audit the memory flags of every buffer the lit pass binds.**
   - Buffers read by shaders must be `DEVICE_LOCAL`.
   - Per-frame data should be uploaded through a staging copy, or placed in `DEVICE_LOCAL | HOST_VISIBLE` (ReBAR) memory.
   - Log the memory type and heap at allocation time, and assert/warn if a shader-read buffer falls back to a non-device-local heap.
5. If atomics exist in the mesh or task stage, aggregate them per wave or workgroup (subgroup ops) before touching global memory.
6. Compile the Slang shaders with debug info (`-g`) so Nsight's Shader Pipelines view maps offset `0x6e0` to a source line. The machine code in the capture is encrypted, so the instruction couldn't be read from the file.

### Acceptance

- shadowcity: the lit pass drops from ~30–58 ms to the low single-digit ms. The lit-pass mesh shader should cost about the same order as the depth pass's (~0.3–1 ms).
- rt-poc: the lit-pass mesh shader drops from ~470 µs to under ~100 µs.
- In Nsight, the lit-pass `main_mesh` is no longer >90% long scoreboard. During the lit pass, the PCIe throughput row is near idle and VRAM throughput is not saturated.

---

## P1 — Async compute is wired correctly but has nothing to overlap with

### Evidence

- **Little overlap.** Async compute overlaps graphics only **6%** of its busy time in rt-poc and **20%** in shadowcity.
  - The one real overlap is the Hi-Z build running alongside the pass with the clear, which has no semaphore wait. That proves the queue setup works.
- **The expensive compute (step 6) sits between two waits.**
  - It waits on the culled draws (step 5), because it reads their output image.
  - The lit pass (step 7) waits on the compute result.
  - Nothing independent is queued on graphics in between, so the graphics queue sits idle for 336–540 µs every frame.
- **All semaphore waits use `VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT`.** The lit pass only needs the compute result in `FRAGMENT_SHADER`, and the culled draws only need Hi-Z in `TASK_SHADER_EXT`.
- **Each hop between queues costs ~5–10 µs.** That's the semaphore plus the queue ownership transfer, and there are 8 waits and 4 ownership transfers per frame.
- **Redundant waits.** Some waits are on timeline values already covered by an earlier wait on the same queue:
  - The second Hi-Z-related compute submit re-waits the value the first one waited on.
  - The final graphics submit of the frame waits on an old compute timeline value.

### Tasks

1. **Change how the frame graph scheduler orders passes.**
   - Schedule passes that produce async-compute inputs as early as possible.
   - Schedule graphics passes that don't depend on async outputs as late as possible, so they fill the gap.
   - Concretely: submit step 6 as soon as step 5 finishes, then run independent graphics work while it executes. Candidates are the pass with the clear (step 4, if it's shadow maps), particles, or anything not reading the compute result.
2. **Make semaphore wait stages match the first real use.**
   - `FRAGMENT_SHADER` for the lit pass's wait on the compute result.
   - `TASK_SHADER_EXT` for the culled draws' wait on Hi-Z.
   - `COMPUTE_SHADER` for compute's waits on graphics outputs.
   - NVIDIA may not exploit narrow wait stages, so measure.
3. **Fall back to the graphics queue when nothing can overlap.** If a compute batch's only consumer is the very next graphics batch and no independent graphics work exists, run it as plain dispatches on the graphics queue. That removes the semaphores and ownership transfers.
4. **Drop redundant waits.** Skip a wait when that queue has already waited on an equal or higher value of the same timeline semaphore.

### Acceptance

- In the Nsight timeline, the step-6 compute overlaps graphics work.
- Async busy time overlapped with graphics goes above 50%, and the graphics queue has no idle gap during step 6.
- Expected gain: up to ~0.35 ms of 2.1 ms in rt-poc, and ≤0.6 ms in shadowcity. That gain is small until P0 is fixed.

---

## P2 — Smaller items

1. **The lit pass starts with a full GPU flush.** Its first barrier batch contains a global memory barrier, `ALL_COMMANDS/MEMORY_WRITE → ALL_COMMANDS/MEMORY_READ`. Replace it with precise buffer and image barriers for what the pass actually reads.
2. **The frame graph appears to be rebuilt every frame.** The in-app console logs `Frame graph plan: 12 batches …` and `Frame graph transients: …` every frame, alternating between 14 and 16 graphics passes.
   - Cache the compiled plan and transient allocation, and rebuild only when the graph changes.
   - Find out why the pass count flips between 14 and 16.
   - This costs CPU time, not GPU time.
3. **rt-poc has an ~0.85 ms GPU-idle gap at the start of every other frame.** The first submit of the frame has no semaphore waits, yet the GPU starts it ~0.85 ms after it is issued. The cause is likely presentation or the compositor (KDE's KWin).
   - Check the swapchain present mode (MAILBOX or IMMEDIATE versus FIFO) and image count.
   - Test fullscreen with direct scanout.
4. **Particles take ~57 µs** for 49152 vertices (8192 quads) in rt-poc, with most samples in an unlabelled stall category. That's low priority, but instancing or a mesh-shader particle path would be cheaper.
5. **Name your Vulkan objects** with `vkSetDebugUtilsObjectNameEXT` and `vkCmdBeginDebugUtilsLabelEXT` per frame graph pass. The captures have no labels, which made pass identification guesswork. Nsight will then show pass names.

## Re-measuring

- Capture the same scenes and settings again: base clocks, 1 frame, Top-Level Triage, warp-state sampling.
- Record GPU frame time per scene, lit-pass duration, and the lit-pass `main_mesh` share of samples and its stall breakdown.

## Limits of this analysis

- **Stall-reason names are inferred.** They were decoded from Nsight's undocumented trace format, using the order of Nsight's own warp-state metric list. Verify them in Nsight.
- **Pass names are inferred** from draw and dispatch structure, barriers and shader stages. The trace contains no debug labels.
- **The hardware counters weren't decoded**, so the memory-placement hypothesis (P0, hypothesis 2) is unverified. That includes PCIe throughput, VRAM throughput and SM throughput.
- **Command-buffer start timestamps lag behind real GPU work.** Durations were cross-checked with sampled shader activity, not timestamps alone.
