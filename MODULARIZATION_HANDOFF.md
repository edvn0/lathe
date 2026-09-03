# Engine modularization — handoff / continuation notes

Full plan lives at `/home/edwin/.claude/plans/i-d-like-to-make-spicy-sun.md` (approved
plan, on this machine only). This file is the **execution status** and the
**remaining work**, self-contained so it survives moving machines. Delete this
file once the split is finished and merged.

## Status: Phase 1, 2, 3 & 4 all done and verified. Nothing left but the
## runtime smoke-test and merging.

Phases 1-3 are committed (`feat: Modularise`, `feat: Some moves`). Phase 4
(the 8-library split) is implemented but **not yet committed** as of this
writing — `git status --short` will show the new `src/<module>/CMakeLists.txt`
files plus edits to the root `CMakeLists.txt` and `cmake/renderdoc_config.cmake`.
Every file now lives under its `include/<module>/` / `src/<module>/`
directory, and `mingw-vulkan-core` is a thin `INTERFACE` target aggregating
8 real `engine_<module>` static libraries (`engine_core`, `engine_gpu`,
`engine_assets`, `engine_physics`, `engine_scene`, `engine_terrain`,
`engine_rendering`, `engine_app`).

Verified green:
- `TARGET=linux-native CMAKE_BUILD_TYPE=Debug   ./compile.sh --rebuild` — clean parallel build, 681 targets.
- `TARGET=linux-native CMAKE_BUILD_TYPE=Release ./compile.sh --build`   — clean build, 680 targets (no `mingw-vulkan-tests` filter difference; `global_new_delete.cxx` correctly excluded).
- `TARGET=linux-native` `--test` passes all 123 tests, both Debug and Release.
- `TARGET=windows-mingw CMAKE_BUILD_TYPE=Debug  ./compile.sh --rebuild` — clean cross-compiled build, 648 targets, links `bin/mingw-vulkan.exe`. (Not built earlier in the previous session; built and verified in this one.)
- Back-edge verification script (Phase 2's version, `#include`-based) still
  reports the single expected edge and nothing else — Phase 4 didn't touch
  any `#include`.

**Runtime smoke-tested this session** (the previous session's environment
had no display; this one does — a real NVIDIA GPU behind an X11 display at
`:1`). Ran `build/linux-native-debug/bin/mingw-vulkan` directly for 15s
uninterrupted: selects the real physical device, creates the logical device
and swapchain, initializes the renderer and registers all 13 pipelines,
loads `test_cube.glb`, streams its textures through the async
`texture_streamer` (including a genuine, pre-existing, unrelated content bug
— `dirt_rough_1k.exr` fails to decode with "unsupported feature" and
correctly falls back to its fallback texture instead of crashing), renders
ImGui, and survives a swapchain recreation. No crash, no Vulkan validation
error, no assertion/segfault/terminate in the log. This exercises the
`IMeshSink`/`IModelSink` paths from Phase 2 end-to-end.
**Not exercised by this particular scene**: terrain streaming (this sample
scene has no terrain), physics debug-draw toggle, shader hot-reload, and the
in-app Memory panel — the sample app here just doesn't happen to touch those
paths. Worth a manual look before merging if the real content set exercises
them.

### A real problem Phase 2's back-edge analysis couldn't catch: link-time circularity

Splitting into 8 real static libraries surfaced something the `#include`-graph
verification script is structurally blind to: **two deliberate cross-module
*symbol* dependencies that only become a problem once each module is its own
archive.**

1. `error_describe.cxx` (**core**) defines the generic `describe(variant<...>)`
   dispatcher (Phase 2 point 9), which calls per-subsystem `describe()`
   overloads defined in **gpu** (`gpu_error_describe.cxx`), **assets**
   (`assets_error_describe.cxx`), and **rendering**
   (`rendering_error_describe.cxx`) — i.e. core calls upward into every
   higher layer, by design, for exactly one function.
2. `renderer.cxx` (**rendering**) instantiates
   `Renderer::record_frame<ApplicationOverlayPolicy>`, which calls
   `ApplicationOverlayPolicy::render_debug`/`render_ui`, defined in
   `renderer_application_policy.cxx` — which Phase 3 deliberately placed in
   **app**, not rendering, precisely because it needs the complete
   `Application` type (see "one deliberate header/impl module split" above).
   So rendering calls upward into app, also by design.

Neither of these is visible to a `#include`-based cycle check — there's no
header cycle, only a link-time symbol dependency. With everything in one
archive (pre-split), the linker's normal multi-pass-within-one-archive
symbol resolution papered over it invisibly. Split into 8 archives linked in
strict layering order, GNU ld's single left-to-right pass genuinely can't
resolve it: by the time it reaches `libengine_core.a` needing a symbol from
`libengine_gpu.a`, it has already moved past that archive and won't revisit
it.

**Fix:** wrap the whole `mingw-vulkan-core` INTERFACE target's link list in
`$<LINK_GROUP:RESCAN,...>` (CMake ≥3.24 — this repo requires ≥3.25, so it's
free to use), CMake's portable equivalent of `-Wl,--start-group` /
`--end-group`. This makes the linker keep revisiting the whole group of 8
until every symbol resolves, regardless of direction. Confirmed working on
both GNU ld (linux-native) and mingw-w64's ld (windows-mingw cross-compile).
If you ever see "undefined reference" errors that look like they *should*
resolve — the defining `.o` is right there in the link line — check whether
it's this exact shape (a symbol only reachable via a later-linked archive
needing an earlier one) before assuming something is actually missing.

### Other Phase 3 corrections found only once Phase 4 forced a real rebuild

Phase 3's own text claimed "the existing single `mingw-vulkan-core`
`CMakeLists.txt` should still work unmodified" because the `file(GLOB_RECURSE
...)` calls are already recursive. That's true for the *glob* calls, but
**five explicit path references elsewhere in the root `CMakeLists.txt` still
pointed at the old flat locations** and needed updating during the Phase 3
checkpoint, before Phase 4 even started:
- The two `platform_sources` globs (`src/platform/{linux,windows}/*.cxx` →
  `src/assets/platform/{linux,windows}/*.cxx`).
- The two `configure_renderdoc()` source paths
  (`src/renderdoc/{renderdoc,stub}.cxx` → `src/gpu/renderdoc/{renderdoc,stub}.cxx`).
- The two `target_precompile_headers(... include/pch.hxx)` calls
  (→ `include/core/pch.hxx`).
If you're redoing this split elsewhere, don't trust "the globs are recursive
so nothing else needs to change" — grep the whole root `CMakeLists.txt` for
every literal `src/` or `include/` path, not just the `file(GLOB_RECURSE)`
calls, before considering a Phase 3 checkpoint build actually green.

### Phase 1 (done): `engine_options` / `engine_deps`

- `cmake/compiler_options.cmake` rewritten: no longer a `configure_compiler_options()`
  function, now defines an INTERFACE target `engine_options` carrying every
  warning/exception/RTTI/sanitizer flag (previously applied per-target via the
  function). Also folds in the `-Wno-missing-field-initializers -Wno-old-style-cast`
  pair that used to be `PRIVATE` to `mingw-vulkan-core` only.
- `cmake/engine_deps.cmake` (new): INTERFACE target `engine_deps` carrying the
  include dirs (`include/`, generated shader-push-constants dir), the SYSTEM
  include dirs for stb/bullet3, the 20-entry third-party `target_link_libraries`
  list, and the `VK_NO_PROTOTYPES`/`GLFW_INCLUDE_NONE`/`GLM_*`/`SLANG_ROOT_PATH`/
  `MINGW_VULKAN_ENABLE_VALIDATION` defines. All previously lived directly on
  `mingw-vulkan-core`.
- `mingw-vulkan-core`, `mingw-vulkan` (executable), `mingw-vulkan-tests` each now
  do `target_link_libraries(<target> PUBLIC engine_deps PRIVATE engine_options)`
  (core) or `PRIVATE engine_options` (exe/tests) instead of calling
  `configure_compiler_options(<target>)`.
- `test/CMakeLists.txt` also switched from linking raw `mingw-vulkan-core` to
  the `mingw-vulkan::core` alias, for consistency with the executable.

This step is a clean, low-risk foundation — it doesn't move anything, just
gives every future module library a one-line way to pick up the same compiler
flags and third-party deps.

### Phase 2 (done): broke every real dependency cycle, still flat layout

Measured the `#include` graph over all ~140 files against the target layering
`core < gpu < assets < physics < scene < terrain < rendering < app` and found
it was **almost already acyclic**. Every fix below is applied and compiles.
Final verification script (see "Verification script" section below) reports
**zero back-edges** except one, which resolves by file *placement* alone in
Phase 3 (no further code change needed):

```
rendering -> app (1): src/renderer_application_policy.cxx -> application
```//
(`renderer_application_policy.hxx` stays a rendering-facing declaration —
only its `.cxx` needs the complete `Application` type. Once `.cxx` moves to
`src/app/` in Phase 3, this stops being a back-edge.)

Changes made:

1. **`Renderer::thread_pool()` → free function.** New `include/thread_pool.hxx`
   + `src/thread_pool.cxx`, core module. Was a `static` method with zero
   dependency on `Renderer` state. Updated every call site
   (`renderer.cxx`, `texture_streamer.cxx`, `load_model.cxx`,
   `pipeline_graph_repository.cxx` — **two** call sites in one function, easy to
   miss, `terrain_streamer.hxx`, `scene.cxx`).

2. **`IMeshSink` interface** (new `include/mesh_sink.hxx`, assets module).
   Pure-virtual `create_mesh`/`submit_mesh`/`geometry_arena()`. `Renderer`
   inherits it (`struct Renderer final : public IMeshSink, public IModelSink`
   — note `final`, added specifically to silence a real `-Wnon-virtual-dtor`
   warning GCC gives non-final classes with virtual functions; `DebugRenderer`
   is `final` for the same reason and didn't warn). `terrain_streamer.hxx`,
   `terrain_slot_pool.hxx`/`.cxx`, `terrain_world.hxx`/`.cxx` now take
   `IMeshSink &` (renamed the parameter to `mesh_sink`) instead of `Renderer &`.
   Callers (`application.cxx`, `main.cxx`) pass `*renderer` unchanged — implicit
   upcast, no call-site changes needed there.

3. **`SubmeshCreateInfo`/`MeshCreateInfo` extracted** from `renderer.hxx` into
   new `include/mesh_create_info.hxx` (assets) — `IMeshSink::create_mesh` needs
   them and they have zero dependency on `Renderer` itself.

4. **`IModelSink` interface** (new `include/model_sink.hxx`, assets module).
   `create_pending_model`/`finish_model_load`/`sampler_storage()`.
   `ModelStreamer::request`/`process_ready` (in `model_streamer.hxx`/`.cxx`)
   now take `IModelSink &sink` instead of `Renderer &renderer`. **Note:**
   `ModelStreamer::request` turned out to be dead code (never called anywhere
   in the tree) but still had to compile correctly.

5. **`load_model_cpu_async` moved from `Renderer` to a free function** in
   `load_model.hxx`/`.cxx` (assets), signature gained an explicit
   `SamplerStorage &sampler_storage` parameter (previously read
   `Renderer::sampler_storage_` via captured `this`). `ModelStreamer::request`
   now calls `load_model_cpu_async(path, sink.sampler_storage())`.

6. **`ShaderHotReloadWatcher::start` takes `ShaderChangeQueue &` instead of
   `Renderer &`.** `Renderer::notify_shader_file_changed` (a one-line wrapper
   around `shader_change_queue_.push`) was removed and replaced with a
   `shader_change_queue()` accessor. `application.cxx` call site updated to
   `shader_watcher_.start(renderer->shader_change_queue(), ...)`.

7. **Physics/scene component split.** `components.hxx` used to define
   everything under `namespace Components` AND `#include "renderer.hxx"`
   just for `MaterialHandle`/`ModelHandle` typedefs. Split into three:
   - `include/transform.hxx` (**core**): `Transform`, `Lifetime`, `Parent` —
     pure data, zero project deps.
   - `include/physics_components.hxx` (**physics**): `BodyShape`,
     `HeightfieldShape`, `RigidBody`, `PhysicsBody` — physics needed these
     directly and previously got them (plus everything else) via
     `components.hxx`, which pulled in `renderer.hxx` transitively. This was
     the only *forced* split (physics really does need these, just not the
     rest of `Components`).
   - `include/components.hxx` (**scene**, now just an umbrella + the
     remaining light/material-override/model/tag components): includes
     `transform.hxx` + `physics_components.hxx`, plus `material.hxx`/`model.hxx`
     directly (not `renderer.hxx`) for `MaterialHandle`/`ModelHandle`.
   `physics_world.hxx`/`.cxx` now include `physics_components.hxx` +
   `transform.hxx` directly, not `components.hxx`.

8. **`IDebugLines` interface** (new `include/debug_lines.hxx`, physics
   module). 3 pure virtuals: `bullet_debug_draw()`, `begin_frame()`,
   `clear_lines()`. `debug_draw::DebugRenderer` (in `debug_renderer.hxx`,
   rendering module) now inherits it (`class DebugRenderer final : public
   IDebugLines`). `PhysicsWorld::attach_debug_drawer` takes `IDebugLines &`
   instead of `debug_draw::DebugRenderer &`. **Gotcha hit:** `scene.cxx` calls
   `physics_world->attach_debug_drawer(renderer)` where `renderer` is a
   `debug_draw::DebugRenderer &` — this needs `DebugRenderer` to be a
   *complete* type at that call site to resolve the upcast to `IDebugLines &`,
   but `scene.cxx` only had it forward-declared via `forward.hxx`. Had to add
   `#include "debug_renderer.hxx"` to `scene.cxx` directly.

9. **`error_describe.cxx` split into 4 files**, all still physically flat in
   `src/` (destined for 4 different modules in Phase 3):
   - `src/error_describe.cxx` (**core**): `describe(ErrorContext)`,
     `describe(ErrorCause)`, `describe(RendererError)` — kept together because
     `RendererError`'s own describe() only touches the generic `ErrorCause`
     dispatcher, nothing subsystem-specific.
   - `src/gpu_error_describe.cxx` (**gpu**): `describe()` for `DeviceError`,
     `ImageError`, `ImageStorageError`, `PipelineError`, `PipelineStorageError`,
     `ShaderObjectError`, `ShaderObjectStorageError`, `GpuResourceTableError`,
     `SamplerStorageError`.
   - `src/assets_error_describe.cxx` (**assets**): `GeometryArenaError`,
     `MaterialStorageError`, `ModelLoadError`, `TexturePipelineError`,
     `SlangLibraryError`, `renderer::ShaderCompileError`.
   - `src/rendering_error_describe.cxx` (**rendering**): `PipelineGraphError`,
     `ForwardTargetError`.
   All declarations stay in `error_describe.hxx` (unchanged, core) — this
   works because `describe()` overloads only need to be *declared* where
   `std::visit`'s generic dispatcher calls them; the linker resolves the
   actual definition from whichever module provides it. **`RendererError`
   itself moved from being a "renderer.hxx" concept to living in
   `renderer_error.hxx`, reclassified as core module** (see point 11) —
   this is what let `IMeshSink`/`IModelSink` (assets) use it without a
   back-edge into rendering.

10. **Two *genuine* (non-accidental) cross-layer data-type couplings found and
    fixed** — these were NOT accidental over-inclusion, they were real:
    - `ImageStorage::upgrade_pending_image` (gpu, `image_storage.hxx`) takes a
      `CompressedTexture const &`, previously defined inside
      `texture_pipeline.hxx` (assets). Extracted `TextureRole`,
      `CompressedMipLevel`, `CompressedTexture` into new
      `include/compressed_texture.hxx` (**gpu** — zero deps beyond
      `<volk.h>`/stdlib). Both `image_storage.hxx` and `texture_pipeline.hxx`
      now include it.
    - `Pipeline::create_graphics`'s default vertex-input state (gpu,
      `pipeline.cxx`) needs `ModelVertex`/`default_vertex_description()`,
      previously defined inside `load_model.hxx` (assets). Also
      `CompressedModelVertex`/`compress_vertex`/`compress_vertices`/
      `encode_octahedral`/`decode_octahedral` were in the same file and used
      by terrain (assets-adjacent). Extracted all of it into new
      `include/model_vertex.hxx` + `src/model_vertex.cxx` (**gpu** module —
      vertex format is fundamentally a GPU-pipeline concern).
      `load_model.hxx` now includes `model_vertex.hxx`; `pipeline.cxx`
      includes it directly instead of `load_model.hxx`.
    **Lesson for Phase 3**: don't trust the original back-edge analysis
    blindly when a "fix" claims something is accidental over-inclusion —
    verify by checking if the including file actually *uses* a type from the
    header before deleting the include. Two of my initial four "accidental"
    fixes (`image_storage.hxx` dropping `texture_pipeline.hxx`, and
    implicitly `pipeline.cxx` dropping `load_model.hxx`) turned out to be real
    dependencies once the build was attempted, caught immediately by the
    compiler (`'CompressedTexture' has not been declared`,
    `'default_vertex_description' was not declared`). One (`pipeline.cxx`
    dropping `load_model.hxx` for `MaterialCreateInfo`/`to_gpu_material`) was
    genuinely accidental — `load_model.cxx` (not `pipeline.cxx`) needed
    `material_storage.hxx` directly, added it there.

11. **`RendererError`/`RendererErrorType` reclassified core, not rendering.**
    `include/renderer_error.hxx` unchanged in content except gaining the
    `std::formatter<RendererErrorType>` specialization, which used to live
    inline inside `renderer.hxx` (a ~65-line switch statement at the bottom of
    the file). Moved it into `renderer_error.hxx` next to the enum, matching
    every other `*ErrorType` formatter's pattern in the codebase (each lives
    beside its own enum). **Gotcha hit:** `model_streamer.cxx` used to get this
    formatter transitively via `#include "renderer.hxx"`; once that include
    was replaced with `model_sink.hxx` (which only pulls in
    `renderer_error.hxx`, not the formatter), `error("...{}", finished.error().type)`
    failed to compile with a wall of `std::format` template errors plus a
    confusing secondary "consteval function" error on the enclosing
    `std::erase_if` call — both vanished together once the formatter moved.
    If you see `std::formatter must be specialized` errors while doing Phase 3
    moves, check whether the type's formatter used to live in a *different*
    file than its enum and ask whether that file is still transitively
    included at the new location.

12. **Accidental-inclusion cleanups** (no interface needed, just deleted
    unused includes): none survived removal blind — see point 10's lesson.
    `pipeline.cxx` did have one genuinely-unused include removed
    (`load_model.hxx` itself, once `model_vertex.hxx` was split out of it —
    what `pipeline.cxx` actually needed was `model_vertex.hxx`, not all of
    `load_model.hxx`).

### Verification script (re-run this after Phase 3 moves)

This Python snippet (run from repo root) computes the `#include` graph and
reports any edge pointing from a lower layer to a higher one. It treats
`stub.cxx`/`renderdoc.cxx` as gpu and ignores `vma.cxx` (no project
`#include`s, never matches any group — that's fine, harmless to leave
ungrouped). **Update the `groups` dict to reflect the final Phase 3 directory
layout** (this version encodes the flat-file names from *before* the move):

```python
import re,os,collections
groups={
 'core':'allocator arena_allocator object_pool handle holder fly_string logger memory_tracker memory_tracking_ui error_context error_describe error_types config forward global_new_delete pch thread_pool transform renderer_error'.split(),
 'gpu':'context buffer image sampler pipeline pipeline_storage shader_object shader_object_storage swapchain vk_barrier vk_object_name device_error vulkan_bootstrap gpu_resource_table image_storage sampler_storage host_query_context renderdoc compressed_texture model_vertex gpu_error_describe stub'.split(),
 'assets':'load_model model model_storage model_streamer texture_pipeline texture_streamer slang_compiler slang_library shader_change_queue shader_hot_reload_watcher material material_storage mesh_storage geometry geometry_allocator geometry_arena primitive_meshes stb mesh_sink model_sink mesh_create_info assets_error_describe'.split(),
 'physics':'physics physics_world physics_components debug_lines'.split(),
 'scene':'components editor_camera input_events'.split(),
 'terrain':'terrain_chunk terrain_mesh terrain_quadtree terrain_slot_pool terrain_streamer terrain_world noise'.split(),
 'rendering':'renderer render_passes render_stage forward_target debug_renderer imgui_renderer shadow_cascades screenshot pipeline_graph_repository terminal_widget renderer_application_policy engine_models rendering_error_describe scene entity'.split(),
 'app':'application game main'.split(),
}
g={}
for k,v in groups.items():
    for n in v: g.setdefault(n,k)
order=['core','gpu','assets','physics','scene','terrain','rendering','app']
rank={n:i for i,n in enumerate(order)}
back=collections.defaultdict(list)
unk=set()
for d in ('include','src'):
    for root,_,fs in os.walk(d):
        for f in fs:
            p=os.path.join(root,f)
            b=re.sub(r'\.(hxx|cxx|def)$','',f)
            src=g.get(b)
            if src is None:
                unk.add(p); continue
            t=open(p,encoding='utf8',errors='replace').read()
            for m in re.findall(r'#include\s+"([a-z_0-9/]+)\.(?:hxx|def)"',t):
                bb=os.path.basename(m); dst=g.get(bb)
                if dst is None or dst==src: continue
                if rank[dst]>rank[src]:
                    back[(src,dst)].append((p,bb))
print("UNGROUPED:", sorted(unk))
print("\nREMAINING BACK-EDGES:")
tot=0
for (a,b),v in sorted(back.items()):
    print(f"{a} -> {b} ({len(v)})")
    for p,m in v: print("   ",p,"->",m); tot+=1
print("TOTAL:",tot)
```

Last run (before Phase 3) reported exactly one back-edge, the
`renderer_application_policy.cxx` one noted above — expected to resolve by
file placement alone.

## Phase 3 (not started): physically move files into module directories

Move everything from flat `include/*.hxx` / `src/*.cxx` into
`include/<module>/` / `src/<module>/`, using `git mv` so history follows,
then rewrite every `#include "name.hxx"` to `#include "<module>/name.hxx"`
based on the mapping below.

### File → module mapping (authoritative, reflects Phase 1+2 state)

**core** (`include/core/`, `src/core/`):
allocator, arena_allocator, object_pool, handle, holder, fly_string (+.cxx),
logger (+.cxx), memory_tracker (+.cxx), memory_tracking_ui, error_context,
error_describe (+.cxx — the trimmed core-only one), error_types.def, config,
pch, thread_pool (+.cxx), transform, renderer_error

**gpu** (`include/gpu/`, `src/gpu/`):
context (+.cxx), buffer (+.cxx), image (+.cxx), image_storage (+.cxx),
sampler, sampler_storage (+.cxx), pipeline (+.cxx), pipeline_storage (+.cxx),
shader_object (+.cxx), shader_object_storage (+.cxx), swapchain (+.cxx),
vk_barrier (+.cxx), vk_object_name, device_error, gpu_resource_table (+.cxx),
host_query_context (+.cxx), renderdoc (+.cxx from `src/renderdoc/renderdoc.cxx`
and `src/renderdoc/stub.cxx` — **both** stay conditionally compiled, see
Phase 4 notes on `configure_renderdoc`), compressed_texture, model_vertex
(+.cxx), gpu_error_describe (.cxx only, no header), vma (`src/vma.cxx` only,
no header — VMA implementation TU)

**assets** (`include/assets/`, `src/assets/`):
load_model (+.cxx), model, model_storage (+.cxx), model_streamer (+.cxx),
material, material_storage (+.cxx), mesh_storage (+.cxx), geometry,
geometry_allocator (+.cxx), geometry_arena (+.cxx), primitive_meshes (+.cxx),
texture_pipeline (+.cxx), texture_streamer (+.cxx), slang_compiler (+.cxx),
slang_library (+ **two** platform `.cxx`: `src/platform/linux/slang_library.cxx`,
`src/platform/windows/slang_library.cxx` — keep the `platform/` subdirectory
inside `src/assets/platform/{linux,windows}/`, CMake still needs to pick only
the matching one), shader_change_queue, shader_hot_reload_watcher (+.cxx),
mesh_sink, model_sink, mesh_create_info, assets_error_describe (.cxx only),
stb (`src/stb.cxx` only, no header — stb_image implementation TU)

**physics** (`include/physics/`, `src/physics/`):
physics, physics_world (+.cxx), physics_components, debug_lines

**scene** (`include/scene/`, `src/scene/`):
components, editor_camera (+.cxx), input_events
(Note: **not** `entity.hxx`/`scene.hxx`/`scene.cxx` — see rendering below.)

**terrain** (`include/terrain/`, `src/terrain/`):
terrain_chunk (+.cxx), terrain_mesh (+.cxx), terrain_quadtree (+.cxx),
terrain_slot_pool (+.cxx), terrain_streamer, terrain_world (+.cxx), noise (+.cxx)

**rendering** (`include/rendering/`, `src/rendering/`):
renderer (+.cxx), renderer_error was **moved to core**, so just: render_passes
(+.cxx), render_stage, forward_target (+.cxx), debug_renderer (+.cxx),
imgui_renderer (+.cxx), shadow_cascades (+.cxx), screenshot (+.cxx),
pipeline_graph_repository (+.cxx), terminal_widget (+.cxx),
renderer_application_policy.hxx (header only — its `.cxx` moves to **app**,
see below), engine_models (+.cxx), rendering_error_describe (.cxx only),
**and** `entity.hxx`, `scene.hxx`, `scene.cxx` — see the writeup under
"Scene/Entity reclassification" below for why these two moved out of a
naive "scene" module.

**app** (`include/app/`, `src/app/`):
application (+.cxx), game, main.cxx (stays a direct executable source, not
part of the app *library* — see Phase 4), and
**`renderer_application_policy.cxx`** (only the `.cxx` — the header stays in
rendering). This is the one deliberate header/impl module split in the
codebase; comment it clearly when you do the move so a future reader isn't
confused about why the `.hxx` and `.cxx` live in different libraries.

**Not part of any of the 8 libraries** (stay as direct executable sources,
exactly as today): `src/main.cxx`, `src/vulkan_bootstrap.cxx`,
`src/global_new_delete.cxx` (currently compiled into the core library and
excluded only in Release — Phase 4 trap #3 below says move it to the
executable outright), `game/src/*.cxx`, `game/include/*.hxx` (untouched,
already separate).

**Not modularized at all** (leave alone): `test/*.cxx`, `test/CMakeLists.txt`
(still links whichever umbrella target Phase 4 produces).

### Scene/Entity reclassification — read before moving `scene.hxx`

This was the one place my initial module assignment was wrong and worth
understanding before you move files. `components.hxx`, `editor_camera.hxx`,
`input_events.hxx` are pure ECS/data types with no dependency on `Renderer` —
genuinely a low "scene data" layer, positioned (as originally planned) below
terrain/rendering.

**But** `Scene` (`scene.hxx`/`scene.cxx`) and `Entity` (`entity.hxx`, which
`#include`s `scene.hxx` and needs the *complete* `Scene` type in its template
bodies) are a different animal: `Scene::Scene(Renderer &renderer)` genuinely
connects an entt signal to `Renderer::mark_lights_dirty` and needs the
complete `Renderer` type in its `.cxx`. This is a **real**, inherent,
one-directional dependency (`Scene → Renderer`, never the reverse — confirmed
`renderer.hxx`/`renderer.cxx` have zero references to `Scene`/`Components`).
Since `entity.hxx` needs the complete `Scene` type too, and only 3
consumers use `entity.hxx` at all (`main.cxx`, `application.cxx`,
`game/basic_game.cxx` — all app-layer), the clean fix was to fold `Scene` +
`Entity` into the **rendering** module (Scene sits right above Renderer,
consuming it plus `PhysicsWorld` — architecturally it's "engine glue just
below app", the same category as `imgui_renderer`/`terminal_widget` already
in that module), and keep the genuinely-low-level `Components`/`EditorCamera`/
`InputEvents` in "scene". This required zero further code changes (already
done) — it's purely a Phase 3 directory-placement decision, captured in the
mapping above.

### Mechanics for the move

1. `git mv include/X.hxx include/<module>/X.hxx` for every file (and
   `src/X.cxx` → `src/<module>/X.cxx`), including `error_types.def`. Keep
   `game/` untouched. Watch the two `platform/{linux,windows}/slang_library.cxx`
   and two `renderdoc/{renderdoc,stub}.cxx` files — preserve their
   subdirectory nesting under the new module dir.
2. Rewrite every `#include "name.hxx"` (bare, no path) across
   `include/**`, `src/**`, `game/**`, `test/**` to
   `#include "<module-of-name>/name.hxx"`, by building a `{basename: module}`
   lookup table from the mapping above and doing a mechanical per-file
   rewrite (a small Python script reading each `#include "..."` line and
   substituting is enough — this codebase has no relative-path includes to
   worry about, everything is a bare filename today).
   - **Two exceptions, do not prefix these:**
     - `error_types.def` — included as `#include "error_types.def"` with
       `X`/`NX` macros predefined by the includer, from both
       `error_context.hxx` and `error_describe.hxx` (both core, after the
       move, so `#include "core/error_types.def"` — this one *should* get
       prefixed since it's a real project header with a fixed module home).
       Actually — re-check this at move time: since both includer and
       includee end up in `core/`, a same-directory relative include
       (`#include "error_types.def"`) still works even after prefixing
       becomes the norm elsewhere, but for consistency prefix it too:
       `#include "core/error_types.def"`.
     - `shader_push_constants.hxx` — **do not prefix this one**, it's a
       *generated* header (see Phase 4) living under the build directory's
       `generated/include/`, not under `include/<module>/`. Its two includers
       (`renderer.cxx`, `render_passes.cxx`, both rendering) already just do
       `#include "shader_push_constants.hxx"` — leave that line exactly as is.
3. After the moves + rewrite, **the existing single `mingw-vulkan-core`
   CMakeLists.txt should still work unmodified** — it uses
   `file(GLOB_RECURSE ... "src/*.cxx")` and `"include/*.hxx"`, which are
   already recursive and will pick up the new nested directories
   automatically. **Build and test at this checkpoint before touching CMake**
   (`TARGET=linux-native ./compile.sh --rebuild && ./compile.sh --test`) — if
   this doesn't pass, the include rewrite has a bug; find it before moving on
   to Phase 4 rather than debugging two changes at once.

## Phase 4 (done, verified — see status header above): split into 8 static libraries

Only attempt this after Phase 3's checkpoint build is green.

1. One `CMakeLists.txt` per `src/<module>/`, added via `add_subdirectory`,
   each with an **explicit source list** (not `file(GLOB)` — the point of the
   split is for the source list itself to document the module's contents;
   also globs stop making sense once directories already partition sources).
2. Each module library:
   - `add_library(engine_<module> STATIC <explicit .cxx list>)`
   - `target_include_directories(engine_<module> PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../include")`
     (all 8 share the single `include/` root — see the plan's note on
     compile-time enforcement being "by convention" given a shared include
     root; that was an accepted tradeoff, not an oversight)
   - `target_link_libraries(engine_<module> PUBLIC engine_deps PRIVATE engine_options <downstream modules per the layering>)`
     — e.g. `engine_gpu` links `PUBLIC engine_core`; `engine_rendering` links
     `PUBLIC engine_terrain engine_scene engine_physics engine_assets engine_gpu engine_core`.
3. `mingw-vulkan::core` becomes an INTERFACE (or plain) target that links all
   8 `engine_<module>` libraries, so `test/CMakeLists.txt` and the executable
   don't need to know the module list — this alias already exists today,
   just repoint what it aliases.

### Traps — each one is a real, verified-relevant risk, not speculative

- **PCH.** Compile `pch.hxx` once on `engine_core`
  (`target_precompile_headers(engine_core PRIVATE include/core/pch.hxx)`),
  then `target_precompile_headers(engine_<other> REUSE_FROM engine_core)` on
  the rest. `REUSE_FROM` requires **identical compile flags** between the
  reusing target and the one that compiled the PCH — this is exactly why
  Phase 1's `engine_options` (shared INTERFACE target, not a per-target
  function) had to happen first. If REUSE_FROM errors about mismatched
  flags, check that every module links `engine_options` and nothing adds
  target-specific flags on top.
- **Generated `shader_push_constants.hxx`.** Only `renderer.cxx` and
  `render_passes.cxx` (both `engine_rendering`) include it. That target needs
  both the `${shader_reflect_generated_dir}/include` dir (currently added via
  `engine_deps` — fine, no change needed there since it's harmless on every
  module) **and** its own
  `add_dependencies(engine_rendering generate_shader_push_constants)`.
  Today this `add_dependencies` line is on `mingw-vulkan-core`; move it to
  `engine_rendering` specifically. Missing it produces a race that only shows
  up on a clean parallel build (`--rebuild`), not incremental ones — test
  with `--rebuild`, not just `--build`.
- **`global_new_delete.cxx`.** A global `operator new`/`delete` override
  inside a static archive can be silently dropped by the linker if nothing
  else in that archive member is referenced (this is a real, not
  hypothetical, linker behavior for unreferenced archive members). Today it's
  in the core archive with a `Release`-only `list(FILTER ... EXCLUDE)` and a
  paired `PRIVATE MINGW_VULKAN_TRACK_MEMORY` define. **Move it to the
  executable's own sources** (alongside `main.cxx`/`vulkan_bootstrap.cxx`,
  which already aren't part of any of the 8 libraries), keeping the `Release`
  exclusion logic. Verify Debug memory tracking still works after this move
  (the in-app Memory panel, `memory_tracking_ui.hxx`/`Application::on_ui`).
- **jemalloc.** Currently linked `PUBLIC` on `mingw-vulkan-core` for
  `Linux + Release`. Its `malloc`/`free` interposition depends on final
  link-line position — attach it to the **executable** target instead of any
  leaf library, for the same reason `global_new_delete.cxx` moves there.
- **`set_source_files_properties` on `vma.cxx`** (`-w` on GNU/Clang, `/W0` on
  MSVC) is **directory-scoped** in CMake. Once `vma.cxx` lives in
  `src/gpu/` with its own `CMakeLists.txt`, this property call must move into
  that new file — if left in the root `CMakeLists.txt`, it silently stops
  applying and VMA's (extensive) warnings return, which fails the build under
  `-Werror`/`WERROR=1`.
- **`configure_renderdoc()`** (in `cmake/renderdoc_config.cmake`) does
  `target_sources(${target} PRIVATE ...)` to attach exactly one of
  `renderdoc.cxx`/`stub.cxx`, and sets `HAS_RENDERDOC` **`PRIVATE`** on that
  same target. Must be called with `engine_gpu` (which owns
  `renderdoc.hxx`/`.cxx`) as `${target}` now, instead of `mingw-vulkan-core`.
  **Also change `HAS_RENDERDOC` to `PUBLIC`** in that cmake file —
  `application.cxx` (app module) and `vulkan_bootstrap.cxx` (direct exe
  source) both `#include "renderdoc.hxx"` and check `HAS_RENDERDOC`; today
  they get the define transitively because everything is one archive, but
  once gpu is its own library, `PRIVATE` means those other targets see it
  *undefined*, which is a silent logic bug (`#if HAS_RENDERDOC` evaluates
  false-via-undefined instead of erroring, so this won't even fail loudly —
  verify by grepping `HAS_RENDERDOC` usage sites after the split, not just by
  building).
- **Output directories.** Currently set per-target via
  `set_target_properties(... ARCHIVE_OUTPUT_DIRECTORY ...)` /
  `RUNTIME_OUTPUT_DIRECTORY`. Switch to the global
  `CMAKE_ARCHIVE_OUTPUT_DIRECTORY` / `CMAKE_RUNTIME_OUTPUT_DIRECTORY` cache
  variables (set once near the top of the root `CMakeLists.txt`) instead of
  adding a `set_target_properties` block for each of the 8 new libraries.
- **`file(GLOB)` for `mingw_vulkan_headers`** (the IDE-listing glob for
  `include/*.hxx`, appended as sources to `mingw-vulkan-core` purely so IDEs
  show headers in the project tree) — once split, either drop it (small IDE
  regression, low cost) or repeat a per-module version of it. Not
  functionally load-bearing either way, lowest-priority item on this list.

## Verification checklist for the finished split

```
TARGET=linux-native  ./compile.sh --rebuild   # done: clean, parallel, 681 targets, green
TARGET=windows-mingw ./compile.sh --rebuild   # done: clean, parallel, 648 targets, green (cross-compiled)
TARGET=linux-native  ./compile.sh --test      # done: 123/123 passing, both Debug and Release
```

Then:
- ✅ Re-ran the "Verification script" above (unchanged — Phase 4 didn't touch
  any `#include`): still exactly the one expected back-edge.
- ✅ Confirmed `HAS_RENDERDOC` is visible (not silently `0`-via-undefined) in
  both `application.cxx` and `vulkan_bootstrap.cxx` after the `PUBLIC` fix —
  checked `compile_commands.json` for `-DHAS_RENDERDOC=` on both TUs; both
  show `HAS_RENDERDOC=0` (RenderDoc disabled in this environment, as
  expected — the point was confirming the define reaches them at all, which
  it now does).
- ✅ Confirmed Release-mode Linux attempts to link jemalloc (falls back to a
  warning + system allocator in this environment, since libjemalloc-dev
  isn't installed here — that's a pre-existing environment gap, not a
  regression; the `find_library` / `target_link_libraries` logic itself is
  unchanged in intent, just retargeted from `mingw-vulkan-core` to
  `mingw-vulkan`) and that Debug-mode `global_new_delete.cxx` is compiled
  into the executable while Release excludes it (checked via
  `compile_commands.json` / build output, not just visual inspection) — both
  `CMAKE_BUILD_TYPE=Debug` and `CMAKE_BUILD_TYPE=Release` built and tested
  for `TARGET=linux-native`.
- ✅ Ran the app directly (`build/linux-native-debug/bin/mingw-vulkan`, 15s,
  real NVIDIA GPU): device/swapchain/renderer init succeed, model + textures
  load and stream async, ImGui renders, swapchain recreation survives, no
  Vulkan validation errors or crashes. See the status header above for
  details. Terrain streaming, physics debug-draw, and shader hot-reload
  weren't exercised by this particular sample scene — worth a manual look
  with real content before merging, but the async-loading interfaces from
  Phase 2 (`IMeshSink`/`IModelSink`) are now confirmed wired correctly at
  runtime, not just at compile time.
