# Engine review follow-ups

This is a handoff for work explicitly deferred out of a large modernization
pass, so it can be picked up cold in a later session. Each section is meant
to be actionable on its own — read the section for the item you're picking
up, you shouldn't need the others.

None of this was verified visually (no live GPU/display was available when
it was written) beyond `TARGET=linux-native ./compile.sh` builds and
`ctest`. Treat "compiles and passes the test suite" as necessary, not
sufficient, for anything that touches rendering or physics.

---

## What's already done (for orientation, not re-doing)

A prior session did a large pass across the engine: sanitizers wired into
CI (`SANITIZE=1`, `.github/workflows/build.yml`'s `sanitize` job),
`-Werror` available and CI-enforced (`WERROR=1`), a `.clang-tidy` config
(not yet CI-enforced, not yet triaged — see "clang-tidy first pass"
below), pinned previously-floating CPM deps, a real
`BULLET2_MULTITHREADING` fix (the project's own `"BT_THREADSAFE OFF"` CPM
option was a no-op — Bullet's actual gate is `BULLET2_MULTITHREADING`,
which defaulted off, silently compiling `btSpinMutex` as an
assert-and-abort stub despite the project fully using Bullet's `*Mt`
classes), `PhysicsWorld` rebuilt to store live bodies as a
`Components::PhysicsBody` ECS component instead of an internal hash map,
async model-CPU-loading (`Renderer::load_model_cpu_async`,
`ModelStorage::create_pending_model`/`upgrade_pending_model`,
`Renderer::finish_model_load`, a new `ModelStreamer`), `main.cxx` split
into `main.cxx` + `vulkan_bootstrap.cxx`, and the hardcoded sample level
extracted from `Application::recreate_entities()` into
`demo_scene.cxx`/`build_demo_scene()`. `Renderer::resize()`,
`ArenaAllocator`, `Scene`'s entity-destroy safety net, `shoot_bullet()`,
and `Renderer::load_model`'s cache key were all hardened against specific
dangling-pointer/collision bugs found during review.

The Docker build image runs as root with no `--user` mapping in
`compile.sh`'s `run_container()` — this was fixed to run as the invoking
host user (see "CI cache permissions" below for the still-open half of
this).

---

## 1. `GeometryArena`: no free-list, upload buffer sized 1:1 with device buffer

**Files**: `include/geometry_arena.hxx`, `src/geometry_arena.cxx`.

**Current behavior** (verified by reading the code, not assumed): `GeometryArena`
is a pure bump allocator. `allocate_bytes()` (`src/geometry_arena.cxx:96-138`)
only ever advances `next_offset`; there is no operation anywhere in the class
that decreases it or reclaims a range. `GeometryArena::create()`
(`src/geometry_arena.cxx:53-94`) allocates `upload_buffer` with
`.size = create_info.capacity` — i.e. a host-visible staging buffer the exact
same size as the entire device-local geometry buffer — even though it's only
ever needed transiently, during the copy commands `write()` records.

**Why it matters**: `Renderer::destroy_mesh()` →
`MeshStorage::destroy_mesh()` (`src/mesh_storage.cxx:36`) only removes the
CPU-side `MeshSlotData` bookkeeping record. It has no way to tell
`GeometryArena` "this range is free now" because that operation doesn't
exist. Every mesh ever created — including ones created and then rolled
back on a later failure inside `Renderer::create_model_common()`
(`src/renderer.cxx`, look for `rollback_meshes`) — permanently consumes
arena space for the process's lifetime. This is fine for a level that only
ever grows; it is not viable for any future streaming/unload scenario
(swapping models in and out, level transitions without a full restart).
Separately, the upload buffer being sized to match the *entire* device
buffer's capacity means the engine permanently reserves however many
hundreds of MB `GeometryArenaCreateInfo::capacity` is set to in host-visible
upload-heap memory, when a much smaller rotating staging ring would do.

**What "done" looks like** (two independent pieces, can be done separately):

1. **Free-list / sub-allocation.** `allocate_bytes()` needs to track freed
   ranges and reuse them, and a `free_bytes(GeometrySlice)` (or similar)
   needs to exist and be called from wherever a mesh/model is actually torn
   down. The natural free-list unit is probably a `GeometrySlice`
   (offset+size) as already defined in `include/geometry.hxx` — check that
   header before introducing a new range type. Watch out for two things:
   (a) `VertexSlice`/`IndexSlice` have different alignment requirements
   (`allocate_vertices` takes an explicit `alignment` param,
   `allocate_indices` derives it from `VkIndexType` via
   `index_element_size()`) — a naive free-list keyed only on size without
   alignment awareness will fragment badly; (b) a freed range can't
   actually be reused until the GPU is done with whatever draw commands
   still reference it, so this needs the same "defer until N frames have
   passed" discipline `PipelineGraphRepository::tick_retirement()` and
   `TextureStreamer`'s `retiring_staging_` already use elsewhere in this
   codebase — don't invent a third pattern, follow one of those two.
2. **Rotating staging buffer instead of a full-size upload buffer.**
   Replace the `.size = create_info.capacity` upload buffer with a small,
   fixed-size ring (a few MB, tunable) that `write()` copies through in
   chunks when a single write is larger than the ring slot, or just
   allocates from round-robin per call otherwise. This needs the same
   in-flight-safety reasoning as above: don't reuse a ring slot until the
   GPU has consumed the copy command that read from it.

**Suggested order**: do the rotating staging buffer first — it's the
smaller, more contained change (touches `create()` and `write()` only,
no public API change) and de-risks understanding the buffer-lifetime
discipline before attempting the free-list, which is the bigger change
(new public API, touches every mesh/model destruction call site).

**Verification**: there's no existing test for `GeometryArena` at all
(check `test/` — as of this writing there's `fly_string_test.cxx`,
`object_pool_test.cxx`, `object_pool_holder_test.cxx`,
`texture_pipeline_test.cxx`, `physics_world_test.cxx`; none touch
geometry). This is GPU-dependent (needs a real `VulkanContext`), so it may
not be unit-testable the way `PhysicsWorld` was — check whether a headless
Vulkan device (`VK_EXT_headless_surface` + lavapipe, or just a real GPU in
CI) is feasible before assuming this needs a live window. At minimum,
manually exercise repeated model load/unload in the running app once a
free-list exists and confirm memory stays bounded rather than growing
without limit.

---

## 2. Data-driven config layer

**Current state**: `include/config.hxx` exists but only holds
compile-time rendering-tuning constants (`frames_in_flight`,
`shadow_cascade_count`, `lod_count`, LOD distance/simplification tables).
Runtime-ish tunables are scattered as `constexpr` literals across
unrelated files, e.g.:

- Window size: `constexpr std::int32_t default_width = 1280;` /
  `default_height = 720;` in `src/vulkan_bootstrap.cxx:255-256`
  (inside `initialize_glfw`'s `ScreenType::windowed` case).
- Camera/player move speed: `move_speed = 5.0F` (plus
  `min_move_speed`/`max_move_speed`) duplicated separately in both
  `include/editor_camera.hxx` and `include/player_controller.hxx` — these
  are two independent structs with the same tuning values, not one shared
  source of truth.
- Demo-scene layout constants: `physics_grid = 5`, floor/grass field sizes,
  light colours/intensities, etc., all inline in `src/demo_scene.cxx`.

**Why this was deferred rather than attempted**: unlike the other items
here, this isn't a bug fix — it's a genuine design decision with no
single obviously-correct answer. Before writing code, decide:

1. **Scope**: is this actually about a *file-loaded* config (TOML/JSON/INI
   parsed at startup), or just about *consolidating* the scattered
   constants into one named struct/namespace so there's one place to look,
   without necessarily adding file I/O yet? The original review flagged
   the *scattered* part as the real smell ("recompile to change a window
   size") — consolidation alone (no new dependency, no parser) is a much
   smaller, safer, still-valuable first step, and could be done without
   resolving (2) or (3) below at all.
2. **Format and dependency**: if file-loaded, this project has no
   config-parsing library today. Pulling one in via CPM is itself a choice
   (TOML via `toml++`, JSON via a lightweight header-only parser, or a
   hand-rolled INI reader if the surface area is small enough not to
   justify a dependency at all) — don't default to JSON just because it's
   familiar; this project already uses `.slang`/`.gltf`/etc. and has no
   existing JSON dependency to reuse (fastgltf embeds simdjson internally
   but it's not exposed as a general-purpose parser).
3. **Reload semantics**: startup-only, or hot-reloadable like the shader
   watcher (`ShaderHotReloadWatcher`, already wired via `efsw`)? Given the
   project already has hot-reload infrastructure and a stated interest in
   iteration speed (see the shader hot-reload feature itself), hot-reload
   is plausible but adds real scope — confirm intent before building it.

**Recommended minimal first step** if picking this up without further
direction: do the consolidation (option 1 above) only. Create a single
`EngineConfig`-shaped struct (name it whatever fits the codebase's
conventions — check how `PhysicsWorldSettings`, `BloomSettings` etc. are
named/structured as precedent) holding at least window size and the
player/camera movement tuning (deduplicating the two copies), constructed
with the current hardcoded defaults, and thread it through
`initialize_glfw`/`Application`/`EditorCamera`/`PlayerController`'s
construction sites. This alone fixes the "one place to look" problem and
sets up a natural seam for file-loading later without committing to a
format now.

---

## 3. `VkPipeline` → `VK_EXT_shader_object` migration: decide, don't guess

**File**: `docs/pipeline_to_shader_objects.md` (already exists, detailed
6-phase plan with exit criteria and a risk register — read it in full
before touching anything here).

**Current state**: mid-migration. `src/renderer.cxx` has a documented
"Phase 3 of docs/pipeline_to_shader_objects.md" section (search for that
exact string) containing dynamic-state helpers
(`default_shader_object_vertex_input`,
`set_mesh_shader_object_dynamic_state`) that compile but are deliberately
uncalled — marked `[[maybe_unused]]` with a comment pointing back to the
migration doc rather than deleted, specifically so a future `-Werror`
build doesn't trip on intentionally-staged code. `bind_graphics_node()`
already supports both a `VkPipeline` and a `ShaderObjectSet` path
side-by-side (resolved via `PipelineGraphRepository::resolve_shader_objects`
vs `resolve`).

**What's actually needed here is not code — it's a decision**: does this
migration continue, or does the project commit to keeping
`VkPipeline` as the permanent path and the dual-backend code gets
*removed* instead (deleting `ShaderObjectSet`, the Phase-3 helpers, and
the `resolve_shader_objects` branch)? Dual-backend code that never
finishes migrating is worse than either endpoint — it's the thing that
makes `bind_graphics_node()` and friends harder to reason about with no
corresponding benefit.

Note the doc's own checkboxes are **all still unchecked** (every `- [ ]`
in the file, none marked `- [x]`) despite the code showing real evidence
of Phase 1-3 progress (`select_physical_device`'s
`shader_objects_supported` feature-detection in
`src/vulkan_bootstrap.cxx`, the `ShaderObjectSet`/dual-backend plumbing in
`renderer.cxx`/`pipeline_graph_repository.hxx`) — the tracking checklist
was never kept in sync with the actual work. Don't trust the checkboxes as
a status signal; re-derive actual progress from the code itself (or ask)
before resuming. And either way, don't restart or extend this migration
without the continue-vs-revert decision being made explicitly by the
project owner, not inferred from what code happens to exist.

---

## 4. Adopt `ModelStreamer` in `build_demo_scene`

**File**: `src/demo_scene.cxx`.

**Current state**: `build_demo_scene()` still calls the fully synchronous
`Renderer::load_model()` (via the local `load_or_fallback` lambda near the
top of the function) for every model, even though `ModelStreamer` (added
this session — see `include/model_streamer.hxx`,
`Renderer::model_streamer()`) now exists and does the CPU-heavy parse work
off the render thread.

**Why this wasn't just switched over**: `build_demo_scene` immediately
uses a just-loaded model's bounds to lay out content — most concretely,
`cube_half_extents` is computed from `renderer->model_bounds(cube_model)`
right after loading (`src/demo_scene.cxx:216`), and that value then drives
the entire physics-cube grid spacing (`spacing`, `physics_grid` loop,
`src/demo_scene.cxx:213-231`) plus the floor entity's scale later in the
same function. `ModelStreamer::request()` returns a handle immediately,
but the *real* bounds aren't available until `process_ready()` promotes it
on a later frame — `model_bounds()` on a still-pending handle returns the
fallback's bounds, not the real ones, so the grid layout would silently
use wrong spacing for however many frames the load takes.

**What "done" looks like**: this needs the grid-layout logic to either (a)
defer running until the streamed model is confirmed ready (e.g. poll
`model_streamer()`'s state, or have `ModelStreamer` support a completion
callback so `build_demo_scene` can schedule the grid-spawning part for
later), or (b) stop depending on a *specific* model's runtime bounds for
layout at all — e.g. hardcode the grid spacing as a config value (ties in
with item 2 above) rather than deriving it from `cube_model`'s bounds.
(b) is simpler and arguably more correct anyway (layout shouldn't silently
change if someone swaps the cube model asset), but changes the demo
scene's behavior slightly (spacing no longer auto-adapts to the model's
actual size) — confirm that trade-off is acceptable before doing it.
`house_model`/`tree_model`/`helmet_model` loads don't have this problem as
acutely (their bounds usage is more local/per-entity), so this is really
just about `cube_model`.

---

## 5. CI cache permissions — worth checking, not confirmed broken

**File**: `.github/workflows/build.yml`.

This session fixed `compile.sh`'s local Docker invocation to run as the
host user instead of root (`run_container()`), because the image's
`Dockerfile` has no `USER` directive and files written into bind mounts
were coming out root-owned on the host — this broke `ctest` (couldn't
write `Testing/Temporary/LastTest.log`) and blocked direct edits to
`CMakeCache.txt`.

**CI's own `docker run` invocations were *not* updated** (there are 8 of
them across the `build` and `sanitize` jobs, none pass `--user`) — this
was out of scope for this session and was not verified against a real CI
run. The theoretical risk: CI writes `~/.cache/CPM` and `~/.cache/ccache`
from inside the root-running container, then `actions/cache@v4`'s save
step runs as the GitHub Actions runner's own (non-root) user *outside* the
container — if that step can't read root-owned files to tar them up, cache
saves could be silently failing (or already are, unnoticed, since a
restore-miss just means a slower rebuild rather than a hard failure).
**This needs someone to actually check a CI run's cache-save step output**
(or reproduce locally with `act` or similar) before deciding whether it's
worth mirroring the `--user`/`HOME` fix from `compile.sh` into the
workflow file too. Don't assume it's broken and "fix" it speculatively —
confirm first, since getting the `HOME`/`--user` combination wrong in CI
specifically (where the runner's UID may differ from what's assumed) has
its own failure modes.

---

## 6. `.clang-tidy`: first full run + triage

**File**: `.clang-tidy` (added this session, not yet run project-wide).

The config enables `bugprone-*`, `clang-analyzer-*`, `performance-*`, and
a small curated set of `modernize-*`/`readability-*` checks, deliberately
excluding checks that would fire constantly on this codebase's deliberate
patterns (raw Vulkan/Bullet pointers, `reinterpret_cast` for handle
interop, literal graphics/physics constants). It is **not** wired into CI
and no one has run it against the full source tree yet. Next step: run
`clang-tidy` (via `run-clang-tidy` or similar, using the project's
`compile_commands.json`) across `src/` and `include/`, triage the
findings into "real bug, fix it" vs "false positive for this codebase,
add a targeted `NOLINT` or adjust the config's `Checks:` list", and only
then consider adding it to CI (as a non-blocking report first, most
likely, given the size of a first-run finding list on a ~27k-line
codebase is unknown).

---

## 7. Broader test coverage (renderer, scene)

**Current state**: `PhysicsWorld` got real unit tests this session
(`test/physics_world_test.cxx`) because it needs no GPU — only a thread
pool and Bullet. `Renderer` and `Scene` (beyond what `PhysicsWorld`'s
tests indirectly exercise) still have zero tests, and both are tightly
coupled to a live `VulkanContext`/GLFW window today, which is why they
weren't touched this session.

**Before writing any renderer tests**, answer: is a headless Vulkan
context (`VK_EXT_headless_surface` + a software rasterizer like lavapipe,
or a real GPU made available to CI) actually feasible in this project's
CI environment? If yes, the highest-leverage first test is probably a
smoke test: bring up a headless `Renderer`, run a handful of frames
(`prepare_frame`/`record_frame`/present-equivalent), tear down clean,
assert no validation-layer errors. If a headless context isn't feasible,
consider whether any of `Renderer`'s *pure logic* (cache-key computation,
error-path branching, the `create_model_common` mesh-rollback logic) can
be extracted and tested without a device at all, similar to how
`build_model_slot_data`-shaped logic was already factored out for reuse
between `create_model` and `finish_model_load` this session.

---

## 8. `windows-mingw` target — unverified for everything in this backlog's parent session

Every change from the session that produced this backlog was verified
only against `TARGET=linux-native` (per explicit instruction not to
spend time on the mingw cross-compile target that session). Before
considering any of that work fully "done," someone should run
`TARGET=windows-mingw ./compile.sh --rebuild` and confirm it still builds
clean — particularly the `vulkan_bootstrap.cxx`/`main.cxx` split (new
source file added to the executable target in `CMakeLists.txt`) and the
`-Werror`/`SYSTEM`-include fixes (`glm`, `stb` — mingw-w64's GCC may
surface different or additional warnings than native GCC 14 did).

---

## Lower-priority / explicitly-not-worth-doing-yet

- **Unused-`#include` cleanup**: clangd flags a long tail of
  not-used-directly includes across `main.cxx`, `application.cxx`, and
  others (pre-existing, not introduced this session). Genuinely low
  value relative to effort — only worth doing as part of an IWYU-tool-driven
  pass across the whole codebase at once, not file-by-file.
- **Mesh-level async pending/upgrade**: considered and deliberately not
  built. `MeshHandle`s are only ever created as an internal step of
  loading a *model* (`Renderer::create_model_common`), never requested
  standalone by external code, so there's no scenario today where a
  "pending mesh handle" would need to be handed out before its real
  geometry exists — the model-level pending/upgrade support
  (`ModelStorage::create_pending_model`/`upgrade_pending_model`) covers
  the actual need. Revisit only if something starts requesting meshes
  independently of a model load.
- **`btSetTaskScheduler` is process-global**, so exactly one
  `PhysicsWorld` can be stepping at a time by construction — noted in
  `src/physics_world.cxx`'s comments. Not a bug, but blocks any future
  "simulate a scene in the background while editing" or multi-world
  feature without a scheduler-ownership rework. No action needed unless
  such a feature is actually planned.
