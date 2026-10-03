# Frame graph: implementation status and handoff

This document says what is built, what is verified, what is not, and how to continue. It is written for someone (or an
agent) picking the work up on a machine with a GPU and the Docker build. The design lives in `docs/frame-graph.md`;
read that first for the *why*. This file is the *what happened*: it records where the implementation follows the
plan, where it deviates, and what to do next.

- **Branch:** `claude/youthful-carson-5f71a0` (stacked on `frame-graph-plan`, which holds only `docs/frame-graph.md`).
- **Pull request:** <https://github.com/edvn0/lathe/pull/47>, a draft, titled "Frame graph: compiler (phase 1), queue
  infrastructure (phase 2) and recording backend (phase 3)".
- **Head at the time of writing:** `2fd46af`. CI is green on it: `linux-native / Debug`, `RelWithDebInfo` and
  `Debug / ASan+UBSan / -Werror` all build `lathe-tests` and run `ctest` (see `.github/workflows/build.yml`).
- **Nothing in this branch has been run on a GPU.** Read [Verification status](#verification-status) before trusting
  any runtime behaviour.

## 1. Where we are in one page

| Phase | What | State |
|---|---|---|
| 0 | Baseline harness (occlusion statistics in the benchmark JSON, `compare_screenshots.py`, stored baselines) | **Not started.** Every later acceptance check depends on it. |
| 1 | Device-free compiler core (`frame_graph::compile`) | **Done**, unit and property tested. |
| 2 | Queue discovery, timeline semaphores, `QueueSet`, swapchain split, per-queue Tracy, smoke path | **Done in code, not run on a GPU.** |
| 3 | Recording backend (`vk_translate`, `PassProfiler`, executor) and the frame wrapped as one legacy pass | **Done in code, not run on a GPU.** `RenderingDesc` is translated but nothing produces it yet. |
| 4 | Migrate passes out of the legacy body, back to front | **Not started.** First change that alters renderer behaviour. |
| 5 | Transient allocation and aliasing | Not started. |
| 6 | Enable async compute, one pass at a time, by measurement | Not started. |
| 7 | Delete `RenderStage` / `write_empty_stage` | Not started. |
| 8 | Concurrent sharing for read-mostly buffers | Not started (optional). |
| 9 | Parallel batch recording | Not started. |

What exists end to end today: every frame, `Renderer::record_frame` declares a frame graph containing **one**
`legacy()` graphics pass ("`frame_legacy`") that wraps the entire old recording body. The graph is compiled and executed
through the new backend, and `main.cxx` submits the resulting batches through `QueueSet` with timeline semaphores. With
one pass the plan is one graphics batch, so the *result should be the same GPU work as before*; the value of the
change is that all the infrastructure (queues, submission, profiling, translation) is now live and exercised every
frame, and the migration in phase 4 can proceed one pass at a time.

The compiler is already queue-aware (single, same-family and dedicated topologies, ownership transfers, semaphore waits,
list scheduler), so phases 4 to 6 only need to *declare* passes, not extend the compiler, except for the gaps in
[section 8](#8-known-gaps-and-deviations-from-the-plan).

## 2. How to build and test

### Canonical (the repo's own flow)

`CLAUDE.md` is authoritative. In short, build through `cargo xtask` (it runs CMake and Ninja inside the `cross-build`
Docker image):

```
cargo xtask configure
cargo xtask build
cargo xtask test            # builds lathe-tests and runs CTest
cargo xtask tidy            # clang-tidy; the nightly workflow fails on any finding
SANITIZE=1 WERROR=1 cargo xtask test     # mirror the strict CI job
```

**Run `cargo xtask tidy` early.** I could not run clang-tidy where this was written, and the nightly `tidy.yml` fails
on any finding, so expect some in the new code.

Tests are `test/*_test.cxx` (globbed into `lathe-tests`; no CMake edit needed), doctest, `TEST_SUITE("unit")`,
compiled with `DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS`. **With that config a failing `REQUIRE` does not
throw; it aborts the test case (often as a crash/SIGSEGV report).** Prefer `CHECK` after a `REQUIRE`d precondition and
read the first failure.

### Quick standalone loop for the pure parts (optional)

The frame graph compiler, scheduler, `vk_translate`, queue selection and submission planning are device-free, so they
can be built without Docker in a couple of seconds. This is how most of the work was developed:

```
# Needs Vulkan headers and doctest (e.g. apt: libvulkan-dev doctest-dev), g++ 13 or newer.
mkdir -p /tmp/fg && printf '#pragma once\n#include <vulkan/vulkan_core.h>\n' > /tmp/fg/volk.h   # volk.h shim
printf '#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN\n#include <doctest/doctest.h>\n' > /tmp/fg/main.cxx
g++ -std=c++23 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wno-missing-field-initializers -Wno-old-style-cast \
    -Werror -fno-exceptions -fno-rtti -fsanitize=address,undefined -fno-sanitize=alignment \
    -DDOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS -Iinclude -isystem /tmp/fg \
    /tmp/fg/main.cxx src/rendering/frame_graph/{compiler,frame_graph,scheduler,vk_translate}.cxx \
    src/gpu/{queue_selection,submission_plan}.cxx test/frame_graph_*_test.cxx \
    test/queue_selection_test.cxx test/submission_plan_test.cxx -o /tmp/fg/tests && /tmp/fg/tests
```

That is **74 test cases**. The flag set is the project's (`-Wconversion -Wshadow -Wpedantic -Werror`): the first CI
failure on this branch was a `-Wconversion` that a looser local build had hidden, so use the strict flags.

### Conventions that matter here

- **snake_case** for functions, variables, files and namespaces; PascalCase only for classes and structs.
- Use `git clang-format` on **staged** changes (it formats only changed lines). Do not run `clang-format -i` over large
  existing files (`renderer.cxx`, `main.cxx`, `application.cxx`): it reformats unrelated code. If it touches an
  unrelated hunk (it twice collapsed an `on_ui` block in `main.cxx` and reordered an include in `renderer.hxx`),
  put that hunk back.
- The repo's clang-format indents namespace contents, so the frame graph files are indented inside
  `namespace frame_graph { ... }`.
- **Designated initializers must follow declaration order.** `ImportedDesc` and `ResourceDesc` have their physical
  handles (`image`, `buffer`) deliberately at the *end*; put new fields at the end too.

## 3. Architecture as built

```
declare              compile                       record                         submit
-------              -------                       ------                         ------
FrameGraph           compile(graph, topology,      frame_graph::record(           QueueSet::submit(
 .import_*()   --->   options)                ---> ExecuteInfo)             --->   batches, acquire,
 .add_pass()          = CompiledGraph               = vector<SubmitBatch>           render_finished)
 (PassBuilder)        batches/barriers/waits/       + command buffers filled        one vkQueueSubmit2
                      transfers/timestamp slots                                     per batch, timeline
                                                                                    values from
                                                                                    plan_submissions()
```

Today the declaration is built in `Renderer::record_frame` (`src/rendering/renderer.cxx`), recompiled every frame
(there is no plan cache yet), executed into the frame's already-begun prologue command buffer, and the resulting
`SubmitBatch` list is exposed as `Renderer::submit_batches()` for `main.cxx` to submit.

### File map

**Frame graph (`include/rendering/frame_graph/`, `src/rendering/frame_graph/`)**

| File | Role |
|---|---|
| `types.hxx` | `LogicalQueue`, `QueueAffinity`, `PassType`, `Use`, `LoadOp`/`StoreOp`, `ShaderStage(s)`, versioned handles (`ImageId`/`BufferId`/`PassId`, generation = resource version), `ResourceState`, `Sharing`. |
| `use_table.hxx` | The single source of truth for what a `Use` means in Vulkan terms (stages, access, layout, reads/writes), `discards_contents`, `write_access_mask`. Matches the plan's use table. |
| `frame_graph.hxx/.cxx` | `FrameGraph` (the per-frame builder), `PassBuilder`, `GraphDesc` (what the compiler consumes), imports, `ExitUse`, `records()`. Declaration-time validation lives here. |
| `frame_graph_error.hxx` | `FrameGraphError{type, pass, resource}` plus `std::formatter`s. **Not** wired into `RendererError` (deferred). |
| `compiled_graph.hxx` | `CompiledGraph`, `Batch`, `CompiledPass`, `BarrierSet`, `ImageBarrier`/`BufferBarrier`/`MemoryBarrier`, `SemaphoreWait`, `OwnershipTransfer`. |
| `compiler.hxx/.cxx` | `compile()`: validation, culling, queue resolution, the barrier/ownership `Tracker`, batch splitting, waits, hash. The heart of the work (about 880 lines). |
| `scheduler.hxx/.cxx` | `dependency_successors()` and the optional overlap list scheduler. |
| `physical.hxx` | `PhysicalImage` / `PhysicalBuffer`: plain Vulkan handles, ignored by the compiler and the hash. |
| `rendering_desc.hxx` | `RenderingDesc` / `AttachmentDesc` (what a raster pass renders into). Translated but **not yet produced**. |
| `vk_translate.hxx/.cxx` | Pure translation: `BarrierSet` to `VkDependencyInfo` storage, `RenderingDesc` to `VkRenderingInfo` storage, `physical_resources_of(GraphDesc)`, `image_aspect`. |
| `pass_context.hxx` | `PassContext` handed to record lambdas (`command_buffer`, `frame_index`, `queue`, `image(id)`, `buffer(id)`). |
| `pass_profiler.hxx/.cxx` | Per (queue, slot) timestamp pools, readback into `PassTiming`, interned Tracy source locations. |
| `executor.hxx/.cxx` | `frame_graph::record(ExecuteInfo)`: walks batches, records barriers, zones, timestamps and the record lambdas, returns `SubmitBatch`es. |

**GPU layer (`include/gpu/`, `src/gpu/`)**

| File | Role |
|---|---|
| `queue_selection.hxx/.cxx` | Pure queue-family discovery (`choose_queue_families`), `QueueFamilies` (moved here from `context.hxx`), `AsyncComputeMode`, `QueueTopologyKind`, `queue_requests`, `parse_async_compute_mode`. |
| `submission_plan.hxx/.cxx` | Pure `plan_submissions()`: relative signal indices to absolute timeline values, wait resolution and ordering checks. |
| `queue_set.hxx/.cxx` | `QueueSet`: timeline semaphores, per-slot command pools, `begin_slot`, `command_buffer`, `submit`, `topology()`. |
| `swapchain.hxx/.cxx` | No longer owns a fence or command buffers; `acquire(slot)` / `present(frame)`. |
| `host_query_context.hxx/.cxx` | Per-queue Tracy GPU context (`initialize(ctx, queue, family, name)`). |
| `context.hxx/.cxx` | `VulkanContext` gained `compute_queue`, `async_compute_mode`, `sync_validation`, `async_compute_smoke`, `queue_set`, `compute_host_query_context`. |

**Elsewhere:** `src/vulkan_bootstrap.cxx` (discovery, queue requests, timeline feature, layer settings, queue set creation),
`src/main.cxx` (flags, `begin_gpu_frame` / `end_gpu_frame`), `src/rendering/renderer.cxx` + `include/rendering/renderer.hxx`
(the wrapper and `record_frame_legacy`), `src/app/application.cxx` (the "Frame graph" timings table).

**Tests (`test/`)**

| File | Covers |
|---|---|
| `frame_graph_compiler_test.cxx` | Single-queue barriers (RAW/WAR/WAW, read coalescing, imports, swapchain), culling, every validation error, hash, serialize mode. |
| `frame_graph_queue_test.cxx` | The queue shapes from the plan (async shape, same-family, prologue release, epilogue acquire, concurrent sharing, ping-pong, wait merging, demotion, swapchain wiring, timestamps, one-code-path equivalence) and the **random-graph property test** (500 seeds on each of three topologies). |
| `frame_graph_scheduler_test.cxx` | Hoisting, fences, determinism, property test (topological order, fences respected, never more waits, plan still sound). |
| `frame_graph_hiz_test.cxx` | The occlusion chain from the plan (Hi-Z leaves the frame owned by graphics, same plan frame to frame). |
| `frame_graph_legacy_test.cxx` | The legacy-swapchain shape the renderer uses, legacy fences, `ExitUse`, handles excluded from the hash, `records()`. |
| `frame_graph_vk_translate_test.cxx` | Barrier and rendering translation. |
| `frame_graph_isolation_test.cxx` | Includes only `compiler.hxx` (enforces "the compiler builds without a device"). |
| `frame_graph_test_support.hxx` | The **independent happens-before checker**, `describe()` (plan dump for failure messages) and `build_random_graph()` (seeded generator). |
| `queue_selection_test.cxx`, `submission_plan_test.cxx` | Phase 2 pure logic. |

## 4. The compiler in detail

This is the part most likely to need reading when something looks wrong. `compile(GraphDesc, QueueTopology,
CompileOptions)` does, in order:

1. **Validate** (`validate()` plus errors recorded while declaring): duplicate pass names; invalid or **stale handles**
   (using a version older than the latest); reading an unwritten transient; writing a read-only import; image use on a
   buffer (and vice versa); sampled/storage/shader uses without shader stages; one resource used twice in a pass;
   attachments in a non-raster pass; a raster pass with compute affinity; token use on a compute-affinity pass; an import
   whose exit is not on graphics.
2. **Cull.** Roots: `side_effect()`, `legacy()`, `pinned()`, any pass touching the swapchain import, any pass that writes
   an imported resource's final version. Liveness walks producer edges through resource versions; a discarding access
   does not keep its previous producer alive.
3. **Resolve queues.** One queue, or `async_compute = false`: everything on graphics. Otherwise `compute_required` goes to
   compute and `compute_preferred` goes to compute unless it cannot overlap anything (every other live graphics-affinity
   pass is its ancestor or descendant in the dependency DAG, computed with bitset reachability), in which case it is
   demoted.
4. **Schedule.** `declaration_order` (the default) or `overlap` (see 4.4). The compiler builds both and keeps the overlap
   plan only if it has no more semaphore waits than the declared order.
5. **Build the plan over *nodes*.** A node is a position in the schedule; the virtual **prologue** is node `-1` (owns every
   import's entry state, on graphics) and the virtual **epilogue** is node `pass_count` (applies every import's exit
   state, on graphics). Every cross-queue edge runs from a lower node to a higher one; this is what guarantees no
   wait-before-signal.
6. **Split into batches and wait.** Per queue, a node with an incoming cross-queue edge starts a batch and a node with an
   outgoing one ends it. Batches are sorted by first node (so batch 0 is always the graphics prologue batch). Each batch
   keeps at most one wait per other queue (max signal index, union of the first-use stages). The first batch to touch the
   swapchain waits on image acquisition; the last graphics batch signals `render_finished`.
7. **Hash** the declarations (names, types, affinities, accesses, resource descriptors, topology, options). Physical
   handles are excluded.

### 4.1 The state tracker (`Tracker::apply`)

Per resource it tracks: layout, owner queue, the last write (queue, stages, access, node), the stages read since that
write *per queue*, what is already visible *per queue*, the last access node per queue, and bookkeeping for the chain
barrier below. For each access it decides edges, then either an ownership transfer or a plain barrier:

- **Ownership transfer** when the topology has different families, the resource is `Sharing::exclusive`, the access is on
  a different queue than the owner, and the access does not discard. The release goes at the end of the batch holding the
  owner's last access, the acquire at the start of the batch holding this access; both halves carry the same layouts and
  families so the transition happens once. A transfer back to graphics is also forced at the epilogue for any import that
  ended on compute (the **exit-on-graphics rule**).
- **Cross-queue edges without a transfer** (same family, concurrent sharing, or a discarding access): a semaphore edge
  only, plus a layout-only barrier on the receiving queue (from UNDEFINED when discarding).
- **Same-queue barriers:** a write waits for everything before it (last write plus reads since) and makes the previous
  write's *write bits* available; a read gets a barrier only if the layout changes or its stages/access are not already
  visible.

### 4.2 Rules that were found the hard way (do not "simplify" these away)

The happens-before property tests found each of these as a real bug in an earlier version:

1. **A layout transition counts as a write.** A read that changes the layout is recorded as the last writer (so later
   readers chain from it) and creates cross-queue edges like a write.
2. **A discarding write still makes the previous write available.** `src_access` is the prior write's bits (masked with
   `write_access_mask`) even when the new write discards; otherwise sync validation reports WAW. A transition recorded as
   a "write" with only read bits correctly yields an execution-only barrier.
3. **A timeline-semaphore wait only orders the commands of its own batch, not later batches on that queue.** So when a
   reader on a queue relies on a cross-queue write and it is not itself the waiting access, the compiler emits an
   execution-only **chain barrier** from the first reader's stages (`cross_ordered` / `chain_src` / `chained` in the
   tracker). One such barrier orders every later reader at those stages.
4. **A write that discards the whole resource reaches another queue with a semaphore and an UNDEFINED-layout transition but
   no ownership transfer** (this is why `hiz_build` over Hi-Z needs no transfer into compute).
5. **A read-only import entry state starts visible**, and a **read-only `ExitUse` leaves the resource visible** to readers
   in that scope (the pass's own internal barriers are responsible for that), so neither costs a barrier.
6. **Edges from the prologue are skipped when the import's entry state has no stages or access**, except for the
   ownership release, which always needs them.

### 4.3 Other compiler features

- **`ExitUse`** (`PassBuilder::write(image, use, stages, ExitUse{use})`): the pass enters in one use and leaves the
  resource in another; the graph tracks whole-resource state only. Used by the legacy swapchain pass (enter as colour
  attachment, leave as PRESENT) and intended for bloom/Hi-Z/AO style passes that keep their per-mip transitions inside.
- **`legacy()` passes** get a global `ALL_COMMANDS` memory barrier before them and another at the start of the next pass
  on their queue (or in the epilogue), and act as scheduling fences.
- **Tokens** (`import_token`, `Use::token_write/read`) compile to `VkMemoryBarrier2` only; graphics-only.
- **`CompileOptions::serialize`** replaces precise barriers with `ALL_COMMANDS` ones between passes: a debug A/B tool.
  **No CLI flag exposes it yet** (the plan names `--frame-graph-serialize`).

### 4.4 The scheduler

`SchedulerMode::declaration_order` is the baseline and the default. `overlap` is a deterministic list scheduler: a pass
that depends on a pass from the *other* queue is held back while that producer was scheduled fewer than
`overlap_window` (4) passes ago, so independent passes fill the gap. `pinned()`/`legacy()` passes never move and are
never crossed; ties break by declaration index. It is only enabled by the caller (`CompileOptions::scheduler`);
nothing turns it on in the renderer yet (the plan enables it in phase 6).

### 4.5 The independent checker (`test/frame_graph_test_support.hxx`)

`check_happens_before(graph, compiled, topology)` re-derives, from only what the plan records (barrier scopes, semaphore
waits, ownership transfers), whether every RAW/WAR/WAW pair on every resource is ordered: same-queue pairs by a covering
barrier in slot order, cross-queue pairs by batch-level semaphore reachability, transitively. It also checks that every
transfer has matching release/acquire halves (families, layouts) and that the acquire is ordered after the release; that
signal indices are consecutive and waits refer to earlier batches; and that `render_finished` is signalled exactly once,
by the last graphics batch. Known limits: it models only *declared* accesses (buffer-device-address use that was not
declared is invisible to it, exactly as the plan's Risks section warns); it does not check layout consistency between
consecutive barriers; and there is no two-frame mode yet (the Hi-Z test checks the single-frame properties that make the
steady state work, not a joined two-frame plan).

## 5. Phase 2 as built (queues and submission)

- **Discovery** (`choose_queue_families`): graphics and present are chosen as before (the first family completing the
  pair). Compute: a family with COMPUTE and without GRAPHICS (preferring `timestampValidBits > 0`), else a second queue
  of the graphics family if `queueCount >= 2`, else the graphics queue. `AsyncComputeMode::same_family` forces the second
  option when the graphics family has two queues and otherwise falls back. `queue_requests()` turns that into one
  `VkDeviceQueueCreateInfo` per unique family (two queues of the graphics family for same-family).
- **Device:** selection now requires `timelineSemaphore`; `create_device` enables it and creates the requested queues; the
  compute queue is `VulkanContext::compute_queue` (equal to `graphics_queue` for the single topology). The startup log
  reads `Queues: graphics family 0, compute family 2 queue 0 (dedicated)`.
- **Flags:** `--async-compute=auto|off|same-family` (`off` keeps the discovered queue but makes the compiler put
  everything on graphics, so the fallback is testable on any hardware), `--sync-validation` (sets `validate_sync` and
  `syncval_submit_time_validation` through `VK_EXT_layer_settings` on the instance; **Debug builds only**, ignored with a
  warning otherwise or if the extension is missing), `--async-compute-smoke`.
- **`QueueSet`:** one timeline semaphore per *physical* queue; if compute is the graphics queue the two logical queues
  alias (one pool, one timeline). Per frame slot and physical queue there is a transient `VkCommandPool` plus a growable
  list of primary buffers. `begin_slot(slot)` waits (2 s, `VK_TIMEOUT` and `DEVICE_LOST` both map to `device_lost`) for
  the slot's last signalled value on each timeline, then resets its pools. `command_buffer(queue)` returns a begun
  `ONE_TIME_SUBMIT` primary. `submit(batches, acquire, render_finished)` calls `plan_submissions()` for absolute values
  and then issues one `vkQueueSubmit2` per batch in order. A wait with no stages is widened to `ALL_COMMANDS` (a
  semaphore wait cannot name no stages).
- **`plan_submissions`** (pure): signal value = timeline base + number of signals planned so far on that timeline + 1, so
  it also handles aliased queues; it rejects a signal index that is not the next for its queue and a wait for a signal no
  earlier batch makes.
- **Swapchain:** the fence, command pool and command buffers are gone. It keeps `image_available[slot]` and
  `render_finished[image]` and exposes `current_slot()`, `acquire(slot)` (handles pending and out-of-date recreation,
  reporting `Kind::recreated` with no image), `present(frame)`, `image_available(slot)`, `render_finished(image)`.
- **`main.cxx draw()`:** `begin_gpu_frame` = `QueueSet::begin_slot` then `Swapchain::acquire` then the graphics command
  buffer; `end_gpu_frame` = `vkEndCommandBuffer`, submit the renderer's batches, `present`. The rule "always retire the
  frame we began" is kept.
- **Tracy:** `HostQueryContext::initialize(ctx, queue, family, name)`. A second context named "compute" exists only when
  compute is a separate queue; `record_frame` collects both.
- **Smoke path** (`--async-compute-smoke`, only with a separate compute queue and a single planned batch): the frame
  becomes G0 (the recorded work, waits on the swapchain acquire) then an empty compute batch that waits on G0 then an
  empty graphics batch that waits on that and signals `render_finished`. It exercises timeline values, multi-batch
  submission and slot waits with no data dependency.

## 6. Phase 3 as built (recording and the legacy wrapper)

- **`record_frame` now** (a) validates its arguments, (b) calls `PassProfiler::begin_slot(frame_index)` (reads back and
  resets that slot's pass timestamps; the slot's earlier work is finished because `QueueSet::begin_slot` waited),
  (c) builds the graph: the swapchain imported with entry UNDEFINED and exit PRESENT_SRC, and one `legacy()` raster pass
  `frame_legacy` that declares `write(swapchain, Use::color_attachment, 0, ExitUse{Use::present})`, (d) compiles with
  `QueueSet::topology()` and `async_compute = (mode != off)`, (e) records with `frame_graph::record`, keeping the
  `CompiledGraph` and `SubmitBatch` list as members, (f) returns the old body's result. The old body is untouched,
  renamed `record_frame_legacy`; the pass's record lambda calls it with the executor's command buffer.
- **Why the swapchain is declared that way:** the acquire wait must keep its `COLOR_ATTACHMENT_OUTPUT` stage, so the
  pass declares a colour-attachment use; the old body does its own UNDEFINED-to-attachment-to-PRESENT transitions, so the
  declared *exit* is PRESENT and the epilogue emits nothing. The one cost is a redundant UNDEFINED-to-attachment barrier
  before the pass, harmless because the body transitions from UNDEFINED anyway.
- **What is still outside the graph:** everything the old body does, **and** `prepare_frame` (which still records the
  culling and light-clustering dispatches and their barriers into the prologue command buffer before `record_frame`),
  terrain uploads, `upload_frame_data`, and all existing `RenderStage` timestamps. Both timing systems run side by side
  until phase 7.
- **Executor** (`frame_graph::record`): for each compiled batch, batch 0 appends to the prologue command buffer (left
  open: `main.cxx` ends it) and later batches get fresh buffers from `QueueSet`. Per batch: acquire barriers, then each
  pass (its `before` barriers, a CPU `tracy::ScopedZone` and GPU `tracy::VkCtxScope` from the interned source location,
  a `TOP_OF_PIPE` begin timestamp, the record lambda, a `BOTTOM_OF_PIPE` end timestamp), then releases and the epilogue;
  non-prologue buffers are ended. A batch with no content and no buffer yields a `SubmitBatch` with a null command
  buffer. The executor only fails if a barrier cannot be translated (a resource with no handle) or a buffer cannot be
  obtained; a record lambda that fails reports through its own channel.
- **`PassProfiler`:** per (queue, slot) `VK_QUERY_TYPE_TIMESTAMP` pool sized `2 * max_passes` (64); results are read with
  `VK_QUERY_RESULT_64_BIT` on slot reuse and masked to `timestampValidBits`. A queue whose family has
  `timestampValidBits == 0` records no timestamps and its passes report no time. Source locations are interned per
  (label, colour) in `std::deque`s that are never freed, which is what makes runtime strings legal as Tracy zone names.
- **UI:** the timings panel has a "Frame graph" table (label, id, queue, GPU ms) next to the overlay timings.
- **`vk_translate`:** image barriers cover every mip and layer with the aspect derived from the format (combined
  depth/stencil formats yield both aspects, which needs separate depth/stencil layouts to be valid with
  `DEPTH_ATTACHMENT_OPTIMAL`; the engine's depth formats are depth-only today); buffer barriers are whole-buffer;
  ownership halves keep their family indices, everything else is `VK_QUEUE_FAMILY_IGNORED`; `RenderingDesc` becomes a
  `VkRenderingInfo` with `COLOR_ATTACHMENT_OPTIMAL` / `DEPTH_ATTACHMENT_OPTIMAL` and the resolve target in the same layout.

## 7. Verification status

### Verified

- **Pure logic:** 74 unit/property test cases pass locally with the strict warning flags under ASan+UBSan, and in CI
  (`ctest` in the normal and sanitizer jobs). The happens-before property tests were additionally run ad hoc over
  20,000 seeded random graphs per topology with no failures (average about 5 ownership transfers and 5 waits per
  dedicated-topology graph, so the cross-queue paths are exercised); the scheduler property test over 20,000 compiles
  (42% actually reordered, 888 ending with strictly fewer waits).
- **Compilation of everything else:** CI builds the whole engine in Debug, RelWithDebInfo and the `-Werror` sanitizer
  configuration on every commit of this branch. Before pushing, the new engine files were also syntax-checked locally
  against the real Vulkan, volk, GLFW, VMA, Tracy, Slang and (docking) ImGui headers.

### Not verified (needs hardware)

Nothing from phases 2 and 3 has executed. In particular:

- the frame submission path (`QueueSet`, timeline waits, swapchain `acquire`/`present`, resize/recreate);
- the legacy pass wrapper: image-layout correctness of the swapchain through the new barrier plus the old body's own
  transitions, validation-layer cleanliness, and that output is bit-identical to before;
- the Tracy zone "Frame (legacy)", the per-queue Tracy contexts, and the "Frame graph" timings table;
- `--async-compute-smoke`, `--async-compute=off|same-family`, `--sync-validation` (including the layer-settings path);
- clang-tidy (`cargo xtask tidy`) on the new code.

### Hardware checklist (do these before phase 4, or at least before merging)

The plan's acceptance checks, concretely:

1. **Baselines.** Phase 0 does not exist yet, so first capture "before" output from `main` (or the commit before this
   branch): the four variants are occlusion off, occlusion on, occlusion on with `--meshlet-occlusion=on`, and the
   `always_defer` stub, using `--benchmark-screenshots` and `--occlusion-culling=on|off`. Then confirm this branch is
   bit-identical (a byte compare of the PNGs in keyframe order is enough until `compare_screenshots.py` exists) and the
   occlusion statistics match.
2. **Validation.** Debug build, `--sync-validation`, 300 frames including a window resize and toggling occlusion, meshlet,
   AO and bloom: no new validation messages. Pay particular attention to swapchain image layout messages and anything
   about queue submission, timeline values, or command pool reset.
3. **Tracy.** A named, coloured GPU zone "Frame (legacy)" and a `frame_legacy` row (queue `graphics`, a time) in the
   Frame graph table; the `RenderStage` timings unchanged.
4. **Smoke path** on a dedicated-compute GPU: `--async-compute-smoke`, both Tracy GPU contexts ("graphics" and
   "compute") present, validation clean.
5. **Fallbacks:** `--async-compute=off` and `--async-compute=same-family` behave identically; the startup log line shows
   the topology on a dedicated-family GPU and on a single-queue device (lavapipe/llvmpipe works).
6. **`cargo xtask tidy`.**

### If something fails, where to look

- *Validation about the swapchain image's old layout:* `Renderer::record_frame` (the `frame_legacy` declaration) and the
  `legacy_fence`/exit-use handling in `compiler.cxx` (`Tracker::apply`, the `spec.exit` block).
- *Device lost or hang at the first frame:* `QueueSet::begin_slot` (a wait on a value never signalled) and
  `plan_submissions`; check `QueueSet::submit` updated `last_value` for the slot.
- *Missing or unnamed Tracy zones:* `HostQueryContext::initialize` (naming), `PassProfiler::source_location`, and that
  the build defines `TRACY_ENABLE`.
- *Empty or zero timings:* `PassProfiler::begin_slot` ordering (it must run after `QueueSet::begin_slot` and before the
  executor) and `timestampValidBits`.
- *Recreate/resize misbehaving:* `Swapchain::acquire` (recreation handling) versus `begin_gpu_frame`'s order: the slot
  wait happens before the acquire/recreate decision.
- *A compile-time surprise in a plan:* build the graph in a test and print `test::describe(graph.description(), compiled)`.

## 8. Known gaps and deviations from the plan

Deliberate or forced differences, so they are not mistaken for oversights:

1. **No `RendererError` wiring.** `FrameGraphError` is a standalone type, and `QueueSetError` is a small struct local to
   `queue_set.hxx` (the plan said `DeviceError`). `RecordFn` is `std::move_only_function<void(PassContext&)>`, not one
   returning `std::expected<void, RendererError>`: the plan's version would make the compiler headers depend on
   renderer errors (and `slang.h`), which the isolation rule forbids, so the legacy adapter captures its
   `std::expected` result by reference. Revisit when a real migrated pass needs to fail.
2. **`RenderingDesc` is not produced and not consumed by the executor.** The type, its translation and
   `CompiledPass::rendering` exist and are tested, but `PassBuilder::color`/`write_depth` record only a load-derived
   discard flag (store ops, clear values and resolve targets are dropped), and `executor.cxx` does **not** call
   `vkCmdBeginRendering` / `vkCmdEndRendering`. Phase 4's first raster pass needs both: the builder must fill
   `RenderingDesc` (including resolve targets as `color_resolve` / `depth_resolve` accesses) and `record_pass` must wrap
   the record lambda with begin/end rendering when `compiled_pass.rendering` is set.
3. **No plan cache.** `CompiledGraph::hash` is computed but unused; the graph is recompiled every frame. The plan lists the
   hash-skip cache under phase 3; it is cheap for one pass but should land before the pass count grows.
4. **Transients are images only and have no memory.** `PassBuilder::create` makes transient *images* in the declaration,
   but there is no allocator (phase 5) and no transient buffers; `CompileOptions::alias_transients` does not exist yet.
5. **Missing CLI flags:** `--frame-graph-serialize` (maps to `CompileOptions::serialize`), `--frame-graph-alias`,
   `--frame-graph-dump`. Phase 4's acceptance check uses the first.
6. **`QueueSet` does not warn when the present family differs from graphics** (the plan says it should log a warning;
   the existing swapchain gap is unchanged).
7. **`--async-compute=off` is compile-time only:** the compute queue is still created and, with `--async-compute-smoke`,
   used; only the compiler's placement changes.
8. **Hi-Z two-frame check is not modelled** in the checker (see 4.5).
9. **Phase 0 does not exist** (benchmark statistics, `compare_screenshots.py`, baselines), although `tools/perf/` already
   has `run_benchmark.sh` and `compare_benchmarks.py` that phase 6 will use.
10. **`docs/frame-graph.md` has not been updated** with an "As built" section (`docs/ibl-and-skybox.md` is the model for
    one). Do it when phase 4 settles the pass-declaration API.

## 9. What to do next

**Before writing more code, run the hardware checklist in section 7.** Phases 2 and 3 changed how every frame is
submitted and the legacy pass changes how the swapchain image is transitioned; a defect there would otherwise get buried
under phase 4.

### Suggested order

1. Fix whatever the checklist and `cargo xtask tidy` find.
2. **Phase 0** (small, and everything after needs it): add the occlusion and cluster statistics to the benchmark JSON,
   `tools/perf/compare_screenshots.py` (bit-identical PNG compare in keyframe order, report the first differing pixel),
   and capture and store the four baselines outside the repo. See `docs/frame-graph.md`, Phase 0.
3. **Close the gaps phase 4 needs** (section 8, items 2, 3, 5): fill `RenderingDesc` from the builder
   (load/store/clear/resolve), add `vkCmdBeginRendering`/`vkCmdEndRendering` to `executor.cxx`, add the plan cache keyed
   by `CompiledGraph::hash`, and add `--frame-graph-serialize`. Add unit tests for the builder's `RenderingDesc`
   (the compiler tests are the model) before using it on a real pass.
4. **Phase 4, step 1 of the plan: composite, `ui_only` and screenshot.** Create `src/rendering/renderer_frame_graph.cxx`
   (add it to `src/rendering/CMakeLists.txt`) holding `Renderer::build_frame_graph` and one `add_*_pass` per pass. Keep
   `frame_legacy` as the first pass but shrink it: it now covers "everything except composite/ui/screenshot", declares
   the layouts it expects and leaves for every graph-visible resource it touches (use `ExitUse`), and the new passes
   follow it (`legacy()` already fences both sides). Delete the hand barriers that the graph now derives in the same
   commit. One commit per migrated pass, each with the acceptance check (bit-identical baselines, `--sync-validation`,
   `--frame-graph-serialize` identical).
5. Continue in the plan's order (bloom, forward, AO, depth prepass/Hi-Z/occlusion, shadows, culling and clustering out of
   `prepare_frame`, environment, overlays and UI), then phases 5 to 9.

### Pitfalls to remember

- Every buffer a shader reaches by device address must be *declared* by the pass that dereferences it; sync validation
  cannot see BDA accesses, so the compiler's own tests are the guarantee and an undeclared dependency is silent.
- Imports must leave the frame on the graphics queue (validated). That is what keeps cross-frame history such as Hi-Z
  correct without a cross-frame timeline wait.
- The viewport target and Hi-Z stay imported and per slot (ImGui samples them by bindless index); do not make them
  transient.
- Do not split device-address buffers into sub-range graph resources; transfers and barriers are whole-buffer on purpose.
- Keep `SchedulerMode::declaration_order` as the default until phase 6; reordering can expose undeclared dependencies.
- The strict CI flags (`-Wconversion -Wshadow -Wpedantic -Werror`) are what failed first. Build with them locally.
- If a new test uses `REQUIRE`, remember the no-exceptions doctest configuration (section 2).

## 10. Glossary

- **Node:** a position in the schedule used as the compiler's unit of ordering; the prologue is `-1` and the epilogue is
  the pass count.
- **Batch:** the work of one queue between cross-queue dependencies; one `vkQueueSubmit2`. Batch 0 is always the graphics
  *prologue* batch.
- **Prologue / epilogue:** virtual passes that own every import's entry state and apply its exit state.
- **Ownership transfer (QFOT):** a release barrier on one queue family and an acquire on the other, with identical
  layouts, ordered by a semaphore. Needed for exclusive resources across different families when the contents matter.
- **Exit use:** the use a pass leaves a resource in, when it manages the resource's subresources internally.
- **Legacy pass:** a pass whose body does its own barriers against everything outside the graph, fenced by global
  barriers; the migration shim for phase 4.
- **Fence (scheduler):** a `pinned()` or `legacy()` pass; never moved, never crossed.
- **Chain barrier:** an execution-only barrier from the first reader's stages that orders later readers on the same queue
  after a cross-queue write (rule 3 in 4.2).
- **Token:** a graph resource with no Vulkan object behind it, compiled to a memory barrier only.
