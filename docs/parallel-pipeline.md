# Parallel Pipeline Registration — Implementation Plan

## Objective

`Renderer::initialize()` currently registers 9 pipelines sequentially through
`PipelineGraphRepository::register_pipeline()`, each of which synchronously
compiles 1–3 Slang shader stages before building a `VkPipeline`. Shader
compilation dominates this cost. This plan adds a batched, parallel
registration path (`register_pipelines_parallel`) that:

1. Introduces a `VkPipelineCache` persisted to disk across runs (Task 0),
   independent of parallelization but done first since it changes the
   synchronization rules the parallel path has to follow.
2. Compiles every distinct dirty shader stage across the whole batch
   concurrently on `Renderer::thread_pool()`.
3. Builds every pipeline/shader-object concurrently as well, now
   synchronized around the shared pipeline cache from Task 0 (see
   Task 3's caveat).
4. Falls back to the existing sequential `register_pipeline()` for
   hot-reload (`process_dirty`), which is unchanged and out of scope.

**Task order matters:** do Task 0 before Task 3. Building the parallel
path against a null cache first and retrofitting synchronization once
the cache lands is more error-prone than building it against the cache
from the start.

Net effect: `Renderer::initialize()`'s pipeline setup time should drop from
"sum of 9 sequential compiles+builds" to roughly "the slowest single
stage compile + build," modulo pool contention and shared-stage
deduplication.

## Confirmed constraints (do not relitigate these — verified against source)

- `renderer::SlangCompiler::compile()` creates a fresh `slang::ISession` per
  call from a *shared* `slang::IGlobalSession`. `createSession()` is not
  documented as reentrant-safe; everything after session creation
  (module load, entry point resolution, linking, codegen) only touches
  that call's own `ISession` and is safe to run concurrently.
  → **Mutex required around `createSession()` only.**

- `PipelineGraphRepository::find_or_create_stage` /
  `find_or_create_source_file` / `link_stage_source_files` mutate shared
  maps (`stage_lookup_`, `source_file_lookup_`, `stage_nodes_`,
  `source_files_`). **Must stay single-threaded.**

- `PipelineStorage::create_graphics` / `create_compute` and
  `Pipeline::create_graphics` / `create_compute` currently pass
  `VK_NULL_HANDLE` as the `VkPipelineCache` to `vkCreateGraphicsPipelines` /
  `vkCreateComputePipelines`. Per the Vulkan spec, these commands only
  require external synchronization on the `pipelineCache` parameter, and
  with no cache in play there's nothing to synchronize.
  → **As of today, legal to call concurrently on the same `VkDevice`.**
  **This changes once Task 0 (pipeline cache) lands** — see the note in
  Task 3 for how the two tasks interact. Do Task 0 first, then build
  Task 3's parallel path against a cache that's already in place, rather
  than parallelizing against null and re-adding synchronization later.
  `ShaderObjectStorage::create_linked` / `create_compute` were not
  reviewed — audit them the same way (check for a shared cache or shared
  mutable builder state) before assuming they're equally safe; if unsure,
  serialize just the `use_shader_objects == true` path and parallelize only
  the `VkPipeline` path.

- `Renderer::thread_pool()` returns a function-local static
  `BS::priority_thread_pool`, lazily constructed on first call — safe to
  call from `initialize()`.

## Task 0 — `VkPipelineCache` with disk persistence (do this first)

Right now every pipeline is built cold — no `VkPipelineCache` at all —
which costs full driver compile time on every run, not just on the first
one, and independently of the parallelization work below. Add a cache
owned at the `PipelineGraphRepository` level, loaded from disk at
startup and saved back on shutdown.

**Files:** `pipeline_storage.hxx`, `pipeline_storage.cxx`,
`pipeline.hxx`, `pipeline.cxx`, `pipeline_graph_repository.hxx`,
`pipeline_graph_repository.cxx`, `renderer.hxx`/`renderer.cxx` (create_info
plumbing + shutdown hook only)

1. **Cache creation.** Add a `VkPipelineCache cache_ = VK_NULL_HANDLE;`
   member to `PipelineStorage` (not to `Pipeline` — one cache is shared
   across every pipeline the storage owns, unlike layout/pipeline handles
   which are per-`Pipeline`). Create it in `PipelineStorage::create()`
   via `vkCreatePipelineCache`, seeded with on-disk data if present (see
   step 2), or an empty `VkPipelineCacheCreateInfo{.initialDataSize = 0,
   .pInitialData = nullptr}` otherwise. Destroy it in
   `PipelineStorage::destroy()` with `vkDestroyPipelineCache` after all
   pipelines using it are destroyed (order matters — destroy pipelines
   first, cache last, same as today's shutdown ordering for other
   resources in `Renderer::destroy()`).

2. **Loading from disk.** Add a helper (e.g. free function in
   `pipeline_storage.cxx`, or a small new `pipeline_cache_io.hxx/.cxx` if
   you want it reusable) that:
   - Reads the cache file (path passed in via
     `PipelineStorageCreateInfo`, e.g. `.cache_file_path =
     "cache/pipeline_cache.bin"`) into a `std::vector<std::byte>` if it
     exists.
   - Validates the Vulkan pipeline cache header before trusting it:
     check `VkPipelineCacheHeaderVersionOne` — `headerSize`,
     `headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE`, `vendorID`
     and `deviceID` against `VkPhysicalDeviceProperties` (available via
     `context.physical_device`), and `pipelineCacheUUID` against
     `VkPhysicalDeviceProperties::pipelineCacheUUID`. The driver will
     silently discard stale/mismatched data per spec regardless, but
     checking yourself means you can log
     `"Pipeline cache stale (driver/device changed), starting cold"`
     instead of silently eating the mismatch, and skip feeding a
     corrupt/truncated file to `vkCreatePipelineCache` at all.
   - On any read/validation failure, log and fall back to an empty cache
     — never treat this as a hard error that blocks `initialize()`.

3. **Saving to disk.** Add
   `PipelineStorage::save_cache_to_disk() ->
   std::expected<void, PipelineStorageError>` (or similar) that calls
   `vkGetPipelineCacheData` twice (once to query size, once to fill a
   buffer) and writes the result to `cache_file_path`, atomically if
   convenient (write to a `.tmp` path, then rename over the real file, so
   a crash mid-write doesn't corrupt next run's cache). Expose a matching
   `PipelineGraphRepository::save_pipeline_cache()` that forwards to it.

4. **Wiring the shutdown hook.** Call
   `pipeline_graph_.save_pipeline_cache()` from `Renderer::destroy()`,
   before `pipeline_graph_.destroy()` runs (the cache must still be valid
   — and every pipeline built through it should already be done being
   used — when you call `vkGetPipelineCacheData`). Also consider calling
   it on any other clean-exit path your engine has (e.g. an explicit
   "save and quit" command), but `Renderer::destroy()` alone covers the
   normal shutdown path and is sufficient for this task.

5. **Plumb the cache handle down to `Pipeline::create_graphics`/
   `create_compute`.** Both currently hardcode `VK_NULL_HANDLE` as the
   `pipelineCache` argument to `vkCreateGraphicsPipelines`/
   `vkCreateComputePipelines`. Thread `VkPipelineCache` through from
   `PipelineStorage::create_graphics`/`create_compute` (which now owns
   `cache_`) down into `Pipeline::create_graphics`/`create_compute` as an
   extra parameter, replacing the hardcoded null.

**Acceptance check:**
- Delete the cache file, run once (cold), time `Renderer::initialize()`
  with Tracy.
- Run again (warm, cache file now populated) and confirm both a faster
  `initialize()` time and, if you can inspect it, evidence the driver
  actually used the cache (some drivers/tools report cache hit stats;
  at minimum, confirm the file grows on first run and stays roughly
  stable on subsequent runs rather than the driver ignoring it).
- Corrupt the cache file deliberately (truncate it, flip some header
  bytes) and confirm `initialize()` still succeeds — logs the fallback,
  doesn't crash, doesn't propagate an error up through
  `std::expected`.
- Delete/rename the cache directory entirely and confirm first-run
  behavior (directory gets created, no crash on missing path).

## Task 1 — `SlangCompiler`: make `compile()` thread-safe

**Files:** `slang_compiler.hxx`, `slang_compiler.cxx`

1. Add `#include <mutex>` to `slang_compiler.hxx`.
2. Add `std::mutex session_creation_mutex;` to `SlangCompiler::Impl`.
3. In `compile()`, wrap only the `impl_->global_session->createSession(...)`
   call in `std::lock_guard<std::mutex>`. Everything before (validation,
   file read, macro/search-path vector construction) and everything after
   (module load, entry point, composite, link, codegen, SPIR-V validation)
   stays outside the lock.
4. Verify `Impl` remains movable: `std::mutex` is not movable/copyable, but
   `Impl` is held via `std::unique_ptr<Impl>` in `SlangCompiler`, so moving
   `SlangCompiler` moves the pointer, not the mutex — no change needed to
   `SlangCompiler`'s own move constructor/assignment.

**Acceptance check:** write a small throwaway test (or reuse an existing
test harness if one exists in the repo) that calls `compiler.compile()`
for 8–10 distinct stage requests concurrently from
`std::async`/a thread pool and confirms no crash, no corrupted SPIR-V, and
results match single-threaded compilation of the same requests
byte-for-byte.

## Task 2 — `ShaderObjectStorage` audit (blocking dependency for Task 4)

**Files:** wherever `shader_object_storage.hxx`/`.cxx` live (not yet
reviewed in this conversation — locate and read them first)

Before parallelizing pipeline *building*, confirm
`ShaderObjectStorage::create_linked` and `::create_compute` have no shared
mutable state analogous to a pipeline cache (e.g., a shared descriptor
pool being written to per-call without synchronization, a shared staging
buffer, etc.). If they call `vkCreateShadersEXT` with no shared cache
object, the same Vulkan-spec argument from Task 4 applies. If they share
any mutable resource, keep the `use_shader_objects == true` branch of
`build_node` serialized (see Task 4's fallback note) until that's
resolved.

## Task 3 — `PipelineGraphRepository::register_pipelines_parallel`

**Files:** `pipeline_graph_repository.hxx`, `pipeline_graph_repository.cxx`

Add the batched API. Three phases, in this order, matching the structure
already drafted in this conversation:

**Phase 1 (sequential):** For each `PipelineRegisterInfo` in the input
span, reserve a free `PipelineNode` slot and call `find_or_create_stage`
for each of its shader stage requests. This phase only exists because
`find_or_create_stage`/`find_or_create_source_file` mutate shared
lookup maps — do not attempt to parallelize it. On `pipeline_free_head_ ==
0` or `info.stages.empty()`, record a per-index failure and skip node
allocation for that entry (don't abort the whole batch at this stage —
only a compile or build failure aborts the batch, per the design already
established for `register_pipelines_parallel`).

**Phase 2 (parallel):** Collect every stage index across the whole batch
that is currently `.dirty`, de-duplicated (shared stages across pipelines
in the same batch must compile exactly once — mirror the existing
single-call dedup behavior in `register_pipeline`). Submit one
`thread_pool.submit_task(...)` per distinct dirty stage calling
`compiler.compile(stage.request)`. Collect futures, `.get()` them all,
write `spirv`/`entry_point`/`dirty`/`has_compiled_once` back onto
`stage_nodes_[stage_index]` sequentially after all futures resolve (this
write-back is sequential but cheap — no compilation happens in it). If
any stage fails to compile, record the first error, free every node
slot reserved in Phase 1 for this batch, fill every result with that
error, and return early — do not proceed to Phase 3.

**Phase 3 (parallel, with one caveat introduced by Task 0):** For each
successfully-compiled node, submit `thread_pool.submit_task(...)` calling
`build_node(node)`. `build_node` reads `stage_nodes_[stage_index].spirv`
(already written in Phase 2) and calls into `storage_.create_graphics` /
`create_compute` or `shader_object_storage_.create_linked` /
`create_compute` — gate the `ShaderObjectStorage` path on Task 2's audit
result.

**Caveat from Task 0:** once `PipelineStorage` owns a shared
`VkPipelineCache` (`cache_`), that handle *is* shared mutable state
across every `create_graphics`/`create_compute` call, and the Vulkan spec
requires external synchronization on a `pipelineCache` parameter used
concurrently by multiple threads — unlike the null-cache case, this one
needs a lock. Add a `std::mutex` to `PipelineStorage` guarding just the
`vkCreateGraphicsPipelines`/`vkCreateComputePipelines` call inside
`Pipeline::create_graphics`/`create_compute` (pass the mutex in, or wrap
the call at the `PipelineStorage` level around `Pipeline::create_graphics`
— either works, but keep the locked region to just the create call, not
the SPIR-V/layout setup around it, so you don't serialize away the
benefit of Phase 3 being parallel in the first place). This is a much
shorter critical section than shader compilation, so parallel building
still wins over sequential — it just isn't lock-free anymore the way the
null-cache version was.

Collect futures, `.get()` them, and sequentially assign
`node.live_handle` / `node.live_shader_object_handle` and unlink/relink
free-list pointers (`pipeline_free_head_`, `node.next_free`) since those
are shared mutable state independent of the cache. On any build failure
for a given node, free that node's slot and record its error — but a
build failure for one node in the batch should **not** invalidate other
nodes in the batch (unlike a compile failure, which is all-or-nothing
because compile failures are rare and typically indicate a broken shared
shader file affecting the whole batch — a build failure is more likely
pipeline-specific, e.g. a bad format combination).

**Signature (already agreed):**
```cpp
[[nodiscard]]
auto register_pipelines_parallel(renderer::SlangCompiler const &compiler,
                                  BS::priority_thread_pool &thread_pool,
                                  std::span<PipelineRegisterInfo> register_infos)
        -> std::vector<std::expected<PipelineNodeHandle, PipelineGraphError>>;
```

Add `#include <future>` and the appropriate `BS::priority_thread_pool`
header to `pipeline_graph_repository.hxx`.

**Acceptance check:** unit/integration test registering the same 9
pipelines used in `Renderer::initialize()` via
`register_pipelines_parallel` in one call, and via 9 sequential
`register_pipeline()` calls, in two separate `PipelineGraphRepository`
instances against the same device. Confirm:
- Same number of successful handles.
- Each resulting `Pipeline`/`ShaderObjectSet` is `.valid()`.
- Compiled SPIR-V for shared stages (there are none shared across your
  current 9 pipelines today, but exercise this with a synthetic case
  that shares a stage across two `PipelineRegisterInfo` entries in one
  batch) is compiled exactly once, not once per referencing pipeline.
- A deliberately broken shader (bad Slang syntax) in the batch causes
  every node in that batch to fail with `PipelineGraphErrorType::
  compiler_error`, and all reserved slots are returned to the free list
  (verify via a subsequent successful `register_pipeline()` call
  succeeding at the same slot index).

## Task 4 — `Renderer::initialize()` call-site change

**File:** `renderer.cxx`

Replace the 9 sequential `{ ... auto registered_X =
pipeline_graph_.register_pipeline(compiler, PipelineRegisterInfo{...});
if (!registered_X) { return ...; } X_ = *registered_X; ... }` blocks with:

1. Build a `std::vector<PipelineRegisterInfo>` containing all 9
   `PipelineRegisterInfo` structs, in the same order/content as today
   (forward, forward_blend, light_icon, shadow, shadow_mask,
   depth_prepass, depth_prepass_mask, composite, frustum_cull).
2. One call: `auto results = pipeline_graph_.register_pipelines_parallel(
   compiler, thread_pool(), pipeline_infos);`
3. Loop over `results`, and on the first `!result`, `return
   std::unexpected(make_pipeline_graph_error(result.error()));` — preserve
   existing rollback behavior (the `FinalAction` rollback guard already in
   `initialize()` still fires via the early return, unchanged).
4. Assign each `*results[i]` to its corresponding `X_` member in index
   order.
5. Leave the `light_icon_texture_` creation block (the
   `image_storage_.create_image(...)` call and its error handling)
   exactly where it is today, immediately after `light_icon_pipeline_` is
   known-valid — do not fold it into the batch.

**Do not touch:** `process_dirty()`'s call site in `prepare_frame()`
(hot-reload path) — that stays on the existing sequential
`register_pipeline`-adjacent logic inside `PipelineGraphRepository`
itself (`process_dirty` recompiles one stage at a time already, and
batching hot-reload recompiles is a separate, lower-priority
optimization not in scope here).

## Task 5 — Validation pass

1. Build in both Debug and Release (or your CMake preset equivalents) with
   validation layers enabled.
2. Run with Vulkan validation layers on; confirm no new
   `VUID`-flagged errors around pipeline layout/creation during startup.
3. Confirm engine boots to the same visual output as before this change
   (same 9 pipelines resolve to valid handles, forward/shadow/composite
   passes render identically) — a simple frame screenshot diff against a
   pre-change baseline is sufficient.
4. Time `Renderer::initialize()` before/after with Tracy (you already have
   it wired in) and record the wall-clock delta in the PR description —
   this is the actual point of the change, so it should be measured, not
   assumed. Measure cache cold vs. warm separately from parallel vs.
   sequential, so the PR description attributes the win to the right
   change (Task 0 vs. Tasks 1/3) rather than one combined number.
5. Run TSan (or equivalent, given your WSL/Clang toolchain) once over the
   new code path specifically — cheap insurance given this is the first
   genuinely concurrent section of the pipeline graph, and now doubly
   important given Task 3's added mutex around the shared pipeline cache.
6. Confirm a delete-cache-file run and a warm-cache run both still
   produce byte-identical rendered output — the cache must never change
   *behavior*, only compile time.

## Explicitly out of scope for this plan

- Parallelizing `process_dirty()` (hot-reload). Left sequential
  intentionally — it's already cheap (one changed file at a time in
  practice) and mixing it with the new batch path adds risk for no
  measured benefit yet.
- Parallelizing model/texture loading (`load_model_cpu`,
  `DecodedImage::load_from_file`) — good candidate for a future pass on
  `thread_pool()`, but unrelated to pipeline registration and should be
  its own plan.
