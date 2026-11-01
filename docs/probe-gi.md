# Probe GI (DDGI on ray query): plan and handoff

Status: **phases 0 to 6 done** except the items under "Phase 6 (what is left)" at the end. The sections up to "Progress" are
the original plan and design notes, kept for the reasoning; "Using it" below is what exists. Read `docs/ray-tracing.md`
and `docs/adding-a-pass.md` for the pieces it builds on.

## Using it

`lathe --gi` (needs `VK_KHR_ray_query`; off otherwise). Emissive surfaces, the sun, the sky, local lights and bounce light
all light the scene, dynamically, with no baking. Two cascades of 32x8x32 probes follow the camera: the near one 1 m
apart, the far one 4 m apart (updated every other frame), blended by coverage at shade time.

| cvar | meaning |
|---|---|
| `--gi` | create the volume (and the scene TLAS it needs) |
| `--gi-view off\|irradiance\|state` | debug views |
| `--gi-probe-spacing M`, `--gi-rays-per-probe N`, `--gi-intensity S` | near cascade settings (the far one is 4x the spacing) |
| `--gi-emitter-intensity I` | strength of small emissive pieces (bulbs) as lights, default 200 |
| `--gi-far` | the coarse cascade (on by default) |
| `--gi-synth-lights auto\|on\|off` | the point lights models get from emissive meshes; `auto` is off while GI is on |

**The test town**: `tools/blender/gi_town.py` builds a 554 m hilly town (10 x 10 terraced blocks joined by sloped streets
inside a ring of hills, hollow houses with skylights and coloured rooms, lit towers with neon signs, a covered passage,
a raised plaza with stairs, emissive pillars and market stalls, a climbable watchtower, parked cars, street lamps and
strings of bulbs; 1174 small emitters) with
`~/dev/git/build_linux/bin/blender -b --python tools/blender/gi_town.py -- ~/Assets/Meshes/gi_town` (writes the `.blend` and
the `.glb`, about 35 MB). `lathe --game=lua --script=assets/scripts/gi_test/town_walk.lua --player --gi` walks it: WASD,
Shift, Space, mouse, F free flight (Q/E down/up), 1 noon, 2 dusk, 3 night. Lathe's world is the model with Blender's Y
and Z swapped (Blender (x, y, z) is (x, z, y) here, not the usual glTF -Z); keep that in mind for spawn points.

The editor's frame graph panel has a "Probe GI" stage with the same settings. `assets/scripts/gi_test/` has the test
scenes (Cornell room, sun boxes, the dynamic scene, Bistro shots) and the `gi_emitters` benchmark scenario measures it.
Code: `ProbeVolume` (`probe_volume.*`), shaders `gi_probes`, `gi_probe_trace`, `gi_probe_classify`, `gi_probe_blend`,
the TLAS in `scene_acceleration_structure.*`, the CPU copy of the addressing maths in `probe_addressing.hxx` (tested).

## Goal

Diffuse global illumination that is dynamic and needs no baking:

1. **Emissive surfaces light the scene themselves.** The Bistro's string lights, lanterns and street lamps (emissive
   materials, no `KHR_lights_punctual`) should illuminate the street without point lights standing in for them. This is
   the motivating case.
2. Bounce light: a sunlit wall tints the shaded side of the street; sky light reaches covered areas through openings
   instead of the flat environment SH term.
3. Works with moving lights, moving rigid geometry and a moving sun, converging in a fraction of a second.

Out of scope for the first version: specular GI / reflections (stay on the prefiltered environment cubemap), caustics,
translucency, skinned-mesh and terrain geometry in the trace (Phase 5), probes for the editor viewport's second view.

## The technique, and why this one

**DDGI** (Majercik et al., "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields", JCGT 2019; the
RTXGI SDK is the reference implementation): a 3D grid of probes. Each frame every probe (or a rotating subset) traces
some rays with ray query, shades the hits, and blends the result into two octahedral atlases (irradiance, and distance
moments for visibility). Surface shading samples the 8 surrounding probes, weighted by trilinear weight, a
backface term and a Chebyshev visibility test, so light does not leak through walls.

Alternatives considered:

| Option | Verdict |
| --- | --- |
| Probes captured by rasterising cube faces through the forward path | Needs no RT, but each probe costs a scene render, leaks badly, and the forward path is meshlet/GPU-driven with no cheap "render from here" entry. Rejected. |
| Voxel cone tracing / SDF GI | A voxelisation or SDF pipeline the engine does not have, and large memory. Rejected. |
| Screen-space GI | Off-screen emitters contribute nothing. Bistro lamps are routinely off-screen. Rejected as the base; may complement it later. |
| **DDGI with ray query** | The engine already has ray query, a working TLAS/BLAS PoC and persistent buffers in the graph. Chosen. |

Fallback: without `VulkanContext::ray_query_supported` the feature is off and shading uses today's environment SH
ambient. There is no non-RT GI path, deliberately.

## What exists today (verified in this session)

- Ambient diffuse is `sh9_eval(ubo.environment_sh, ...)` times AO times `environment_intensity` in
  `assets/shaders/forward_geom.slang` (about line 616 to 640). GTAO (`gtao.slang`) multiplies it. This is the term GI
  replaces.
- Ray query: `VulkanContext::ray_query_supported`, features and extensions in `src/runtime/vulkan_bootstrap.cxx`
  (about 757 to 776, 863), TLAS in bindless binding 6 (`gpu_resource_table.cxx`, `set_acceleration_structure`). Only a
  PoC exists: `src/rendering/ray_tracing_poc.cxx`, `assets/shaders/rt_poc.slang`. One hardcoded sphere, built once,
  synchronously. See `docs/ray-tracing.md` "Limits of the PoC".
- Frame graph: engine passes live in `src/rendering/renderer_frame_graph.cxx`; `GameGraph` passes cannot record
  acceleration structure builds (no command buffer). `docs/adding-a-pass.md` "In the engine".
- Persistent buffers: `graph.persistent_buffer` keeps contents across frames, zero-filled on creation or resize, and
  forces the graphics queue. Images are the transient allocator's; **a persistent image resource may not exist**. Check
  `frame_graph` for imported persistent images before designing the atlases around buffers (Phase 2, question 2).
- Geometry: `src/assets/geometry_arena.cxx` holds resident vertex/index data with `SHADER_DEVICE_ADDRESS` usage but
  **not** `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY`. Vertices are `CompressedModelVertex` (20 bytes, quantised
  position, octahedral normal; `include/gpu/model_vertex.hxx`).
- Materials: `src/assets/material_storage.cxx` has a device-address material buffer; textures are bindless
  (`sampled_2d[...]`), ray shading must use `SampleLevel`.
- Lights: directional light `Renderer::DirectionalLight`, point/spot via the clustered light path
  (`light_cull.slang`, `light_cluster.slang`) and shadow atlases. Local light shadows are shadow-map based, which a ray
  cannot use; ray hit shading shadows its direct light with shadow rays instead.
- Emissive: `ModelCpuMaterial::emissive_factor`, `emissive_strength`, `emissive_image`.
- Interim: `synthesize_emissive_lights` in `src/assets/load_model.cxx` makes point lights for emissive mesh pieces and
  `Application::spawn_model_lights` spawns them. Once GI lights emissives for real these double-light the scene: gate
  them behind a cvar (default off when GI is on) in Phase 6.

## Design

### Probe volume

- A camera-following grid, **toroidally addressed** (a probe keeps its texel slot while the volume scrolls; probes
  entering the far side are reset and relit). Start with one cascade: `32 x 8 x 32` probes, spacing about 1 m near the
  camera in world units, both configurable. Add a second, coarser cascade (spacing x4) in Phase 6 for distance.
- Per probe state: a `classification` (active / sleeping / off) and a `relocation offset` (move a probe out of
  geometry). Probes inside geometry or with no nearby surface are switched off after the first rays; this is what makes
  dense scenes cheap and avoids dark/bright leaks from probes buried in walls.
- Rays per probe per frame: 64 at first (tunable 32 to 256). Fibonacci sphere directions, rotated by a random rotation
  per frame (a seeded matrix, not per ray) so the blend converges to a uniform sampling.
- Update budget: all probes every frame at first; later spread updates over N frames by distance and by "dirty" state
  (nearby light or geometry change), the same way the shadow atlas rations updates.

### Per-ray work (compute shader, `RayQuery`, one probe per workgroup row)

1. Trace `TraceRayInline` against the scene TLAS, `RAY_FLAG_CULL_BACK_FACING` off (need backface hits to detect probes
   inside geometry), `tMax` bounded.
2. Miss: radiance from the environment cubemap (the existing prefiltered/specular-mip 0 cube) times
   `environment_intensity` and exposure, so the sky lights the volume.
3. Hit: fetch the 3 vertices via the per-instance address table (`instanceCustomIndex` indexes it), interpolate normal
   and UV, sample base colour and emissive with `SampleLevel` (fixed LOD bias from ray length), then shade:
   `albedo * (direct + indirect) + emissive`.
   - `direct`: sun with a shadow ray; point/spot lights from a capped list of lights near the probe (use the same
     light buffers as the clustered path, but select by distance to the hit, at most 4 to 8 with shadow rays);
   - `indirect`: sample **the probe volume itself** at the hit (previous frame's atlases). This is multi-bounce for free
     and is what makes the volume converge to proper GI.
4. Write `(radiance, hit distance)` into a ray-data buffer, `probes x rays` entries (`RGBA16F` packed).

### Blend (second compute pass)

- Irradiance atlas: octahedral `8x8` interior + 1 texel border per probe (so `10x10`), `RGB10A2` or `RGBA16F`.
  Each texel integrates the cosine-weighted radiance of all rays whose direction is near the texel direction
  (`max(0, dot)^k` weight), blended into the old value with hysteresis `0.97` (lower, about 0.85, for the first frames
  of a probe and when a large change is detected, to cut lag).
- Distance atlas: octahedral `16x16` + border, `RG16F`, holding mean and mean-square distance (`max(0, dot)^64` weight)
  for the Chebyshev visibility test.
- Border texels copied each frame so bilinear sampling across the octahedron edge is valid (a third small pass or the
  tail of the blend pass).
- Use the perceptual encoding the paper uses (store `irradiance^(1/5)`-ish) to avoid banding in dark areas, decode when
  sampling.

### Sampling at shade time (`forward_geom.slang`)

- A shared Slang module (`gi_sample.slang`): given world position, shading normal and view direction, find the 8 probes,
  compute trilinear weights, multiply by backface weight `((dot(dir_to_probe, n) + 1) * 0.5)^2 + 0.2`, Chebyshev
  visibility, and the "crush tiny weights" step; offset the sample position along the normal and towards the camera
  (`normal_bias`, `view_bias`) to fight self-shadow/leak.
- Result **replaces** the `sh9_eval` diffuse term inside the volume, and fades to it over the outer probe layer, so
  nothing pops at the volume boundary. Multiply by albedo and (1 - metallic) as the SH term is today; keep the AO
  multiplier but reduce GTAO's weight when GI is on (it double-counts occlusion; make `ao_intensity` scale, decide
  by eye).
- Specular ambient stays the prefiltered environment.

### Acceleration structure (the large prerequisite)

Follow `docs/ray-tracing.md` migration items 1, 2, 3 and 5; item 4 (skinned/deforming) is Phase 5.

- BLAS per **mesh LOD chosen for tracing** (a coarse one: tracing quality does not need the raster LOD), built when the
  mesh is uploaded, with `ALLOW_COMPACTION`. The arena buffers need `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY`
  added. Positions are quantised: either apply the dequantisation in the instance transform or keep a float position
  stream just for the BLAS (decide after reading `geometry_arena.cxx`; the second is simpler and costs memory).
- TLAS rebuilt each frame from the instance list the renderer already uploads, by a kernel that writes
  `VkAccelerationStructureInstanceKHR` records (`instanceCustomIndex` = index into the per-instance geometry/material
  address table). One TLAS per frame in flight.
- Frame graph: add `Use` values for acceleration structure build and ray query read, translated in
  `frame_graph/vk_translate.cxx`; the build is an engine pass in `renderer_frame_graph.cxx`. Check the existing queue
  and ownership derivation handles it, do not special-case.
- Mask bit on instances so GI can skip things that should not bounce light (decals are not geometry; particles, text and
  water are excluded in v1).

### Settings and debug

- cvars (Display or a new "GI" group): `gi` (on/off, default on when ray query is supported), `gi_rays_per_probe`,
  `gi_probe_spacing`, `gi_hysteresis`, `gi_intensity`, `gi_normal_bias`, `gi_debug`.
- Debug view modes (existing F1..F8 pattern, add after "Lighting only"): "GI only" (the sampled irradiance as the
  diffuse term), plus a probe-visualisation overlay (spheres coloured by irradiance, off probes in red, relocated
  probes marked) drawn with the debug renderer.
- Editor: a section in the lighting settings panel (`draw_lighting_settings`) with the cvars; show the probe count,
  active count, and ms for the trace and blend passes (the frame graph panel already shows per-pass times).

## Phases

Each phase ends with something visible and a perf number. Do not start a phase before the previous one is verified with
`drive-engine` screenshots; build in Release for the heavy scenes. The point of the ordering is that nothing is a guess
by the time the probes arrive.

### Phase 0: setup (small)

- Branch `feat/probe-gi` from `main` (after PR for `feat/editor-ui-and-emissive-lights` merges, or from it).
- Find or build a **test scene** and write down the recipe: (a) Bistro, with `--scene` and a camera shot script (a Lua
  game script like `tmp_bistro.lua` in this session, which must live inside the data dir); (b) a small closed "Cornell"
  room (a Lua or C++ demo game in `game/`) with one coloured wall and a ceiling emissive quad, for light-leak and
  convergence checks. Add a Cornell game as `game/src/gi_test_game.cxx` with 1 to N camera shots (keys), per the
  drive-engine skill advice.
- Capture a baseline in `docs/perf/` (RTX 4070 Ti SUPER numbers exist there to compare against).

### Phase 1: scene acceleration structure (the prerequisite, large)

1. Arena buffers get the BLAS-input usage; add BLAS build on mesh upload (async, compute queue, with compaction).
2. Per-instance table buffer: for each TLAS instance, the device addresses of vertex/index data, the material index,
   and the transform (for normals).
3. Per-frame TLAS build pass (kernel writes instances, then `vkCmdBuildAccelerationStructuresKHR`), as an engine pass.
4. Frame graph `Use` additions and barriers.
5. Verify by porting the PoC: replace the hardcoded sphere with the real scene and keep the "shadow ray towards the sun"
   result; compare with the shadow-map shadows on the Bistro. **Exit criterion**: ray-traced sun shadows from the real
   TLAS look right and the TLAS build costs under about 0.3 ms for the Bistro. This phase is useful on its own
   (ray-traced shadows, AO, reflections later).
6. Handle: device-lost recovery (`test/gpu/device_lost_recovery.sh`) must rebuild BLAS/TLAS; mesh unload frees BLAS;
   model streaming (`ModelStreamer`) must not leave instances pointing at freed geometry.

### Phase 2: probe volume and ray trace, no blend yet

- Probe grid, toroidal addressing, classification buffer, ray-data buffer.
- Trace pass shading hits with **sky and sun only** first (no emissive, no local lights, no recursion).
- Debug: dump a ray-data slice to a screenshot (view probe radiance); probe overlay.
- Decision to settle here (read `frame_graph` first): where the atlases live. Options: persistent buffers (works with
  today's `persistent_buffer`, sample via buffer loads and do bilinear in the shader, which costs but is simple), or add
  a persistent/imported image resource to the engine frame graph (cleaner, hardware bilinear, more work). Recommend
  starting with imported images owned by the renderer (like `shadow_atlas` in `renderer_frame_graph.cxx`, which is
  already a persistent engine image) rather than game-graph buffers.

### Phase 3: blend and shade

- Irradiance and distance blend passes, border copy, hysteresis.
- `gi_sample.slang`, wired into `forward_geom.slang` replacing the SH diffuse inside the volume, with the boundary fade.
- Probe relocation and classification.
- **Exit criterion**: the Cornell room shows colour bleeding from the coloured wall with no leaking through the room's
  walls; the Bistro's shaded side of the street is lit by the sky and a bounce, not flat.

### Phase 4: emissive and local lights, multi-bounce

- Hit shading adds emissive (`emissive_factor * emissive_strength * emissive texture`) and direct light from
  point/spot (shadow rays) and recursion through the volume.
- **Exit criterion**: turn the synthesised point lights off (`gi_synth_lights=0`, see Phase 6) and the Bistro's string
  lights, lanterns and street lamps visibly light the cobbles and walls near them. Colour is right (the bulbs' own
  texture colour comes from the emissive texture now, so the name-based colour guess in `emissive_light_colour` is no
  longer needed for GI).
- Tune intensity: emissive radiance is a texture times a factor; bulbs are tiny, so their power is concentrated in a few
  texels. Importance can be low variance if rays rarely hit a bulb: consider boosting by sampling emissive instances
  explicitly (a list of emissive instance bounds in the per-instance table, a few rays per probe aimed at nearby
  emitters). This is the main quality risk for the Bistro; see Risks.

### Phase 5: dynamic scenes

- Moving rigid instances (TLAS only, already works), moving lights, moving sun (the sky cube changes; probes just
  reconverge, check lag and use the faster hysteresis on large change).
- Skinned meshes: BLAS refit from `skin.slang`'s scratch buffer (`ALLOW_UPDATE`, `MODE_UPDATE`).
- Terrain: a coarse BLAS per terrain tile or exclude it and treat as sky-lit; decide with the terrain owner.

### Phase 6: polish and ship

- Second, coarser cascade; update-rate budgeting by distance; probe "sleeping" when nothing nearby changed.
- Gate `synthesize_emissive_lights` / `spawn_model_lights` behind a cvar, default off when GI is active.
- Editor panel, docs (`docs/probe-gi.md` becomes the real documentation; move the plan to a "design notes" section),
  benchmark scenario in `src/benchmark/benchmark_scenarios.cxx`, and a perf entry in `docs/perf/`.
- Tests: see below.

## Testing

- **Unit tests** (`test/`, run `cargo xtask test`): the CPU-side maths only. Toroidal probe addressing and scrolling
  (which probes are reset when the volume moves by k probes), octahedral encode/decode round trip, Fibonacci
  direction set coverage, hysteresis blend function, the weights helper if mirrored on the CPU. Put the pure functions in
  a header (`include/rendering/gi_math.hxx`) so the tests link without a device.
- **GPU checks** with the `drive-engine` skill: screenshots of the Cornell game and the Bistro with GI on, off, and
  "GI only" view, at fixed frames (allow ~200 frames for convergence; verify with two shots 30 frames apart that they
  match). Validation layer errors (VUID) must be clean in Debug.
- **Light leak test**: Cornell room with a thin wall; irradiance behind the wall must be ~0.
- **Temporal stability**: a shot sequence while the sun moves; no flicker (compare consecutive frames' mean luminance).
- **Device lost**: `test/gpu/device_lost_recovery.sh` must still pass with GI on (probes are persistent state that
  has to be re-created zeroed and reconverge).
- **Perf**: add a `gi` benchmark scenario; record `rtx 4070 Ti SUPER` numbers for trace and blend passes. Budget target
  at 1080p: trace+blend under about 1.5 ms with 64 rays on 8k probes, TLAS build under 0.3 ms.

## Risks and open questions

1. **Emissive sampling noise.** A 7 cm bulb 10 m from a probe is hit by few of 64 rays, so convergence is noisy and
   flickery. Mitigations: explicit emissive sampling (above), larger hysteresis for emissive contribution, or clamping
   the sample radiance and adding emissive as small area lights in the light list. Prototype this early in Phase 4;
   it decides whether the interim synthesised point lights stay as a fallback.
2. **Leaks and probe placement** in thin Bistro geometry (awnings, railings): relocation and the Chebyshev test mostly
   fix it; expose `normal_bias`.
3. **Unit scale.** Models can be in cm (the Bistro is) with the entity scaled to 0.01; the TLAS uses world space, so
   spacing is metres. Confirm instance transforms include the entity scale. The interim light code had to multiply
   range by the parent scale for the same reason.
4. **Light sources the ray cannot shadow-map**: local light shadows are shadow-atlas based; hit shading uses shadow
   rays on a capped light list, so the GI's view of local lights is an approximation of the raster path's.
5. **Material fidelity at hits**: only base colour and emissive; no normal maps, no alpha test in v1 (opaque-flag
   everything except alpha-tested foliage, which needs the any-hit loop from `docs/ray-tracing.md`). Bistro foliage is
   alpha tested; decide to treat it as opaque in the TLAS for GI (cheaper) or run the any-hit loop.
6. **Memory**: 32x8x32 = 8192 probes; irradiance atlas ~ 8192 x 100 texels x 4 B = ~3.3 MB, distance ~ 8192 x 324 x 4 B
   = ~10.6 MB, ray data = 8192 x 64 x 8 B = ~4 MB. Fine.
7. **Frame graph persistent images** (see "What exists today"). May need engine work before Phase 2.
8. **Vulkan project rules** (from the user's CLAUDE.md): the swapchain is `VK_EXT_headless_surface`-backed, and every
   shader is a `VK_EXT_shader_object`. Ray query compute fits this; do **not** introduce a ray tracing pipeline for GI
   (it is a `VkPipeline`, which needs explicit agreement; ray query is enough here).

## Project conventions the implementer must follow

- Branch `feat/probe-gi` (type/name, see `CLAUDE.md`); never a `claude/...` session branch.
- Build with `cargo xtask build`; do not call cmake/ninja; do not rebuild after every small edit; run heavy scenes from
  the Release build; run clang-tidy only when asked.
- Module layering: this is `rendering` (`src/rendering`, `include/rendering`) and may include `core`, `storage`, `gpu`,
  `assets`, `animation`, `physics`, `scene`, `terrain` only. BLAS code belongs next to the mesh (`assets` or `gpu`
  where the geometry arena lives), the pass in `rendering`.
- Tracy zones use the named and coloured macros (`ZoneScopedNC`, `TracyVkZoneC`).
- Commits are GPG signed; check `gpg-agent` has the key unlocked before committing, and ask the user to unlock if not.
- Use `uv` for any Python.
- Verify with `drive-engine` (headless, `--exit-after-frames`), read screenshots, and check the log for `[E]` and VUID.

## Decisions (settled with the user)

- **Atlases**: renderer-owned images imported into the engine graph (like `shadow_atlas`), not game-graph buffers.
  `frame_graph` already supports `import_image` with entry/exit states.
- **Foliage**: alpha-tested (`AlphaMode::mask`) geometry is opaque in the GI TLAS (no any-hit loop in v1).
- **Instance transforms**: the world matrix comes from `get_world_transform` (full TRS including parent scale), so the
  Bistro's 0.01 entity scale is carried by the TLAS instance transform.

## First steps for the new session

1. Read `docs/ray-tracing.md`, `docs/adding-a-pass.md`, `src/rendering/ray_tracing_poc.cxx`,
   `src/rendering/renderer_frame_graph.cxx` (how `shadow_atlas` and the engine passes are declared),
   `src/assets/geometry_arena.cxx`, and `assets/shaders/forward_geom.slang` around line 596 to 645.
2. Settle the two open design questions with the user before writing code: (a) atlases as renderer-owned imported
   images vs game-graph buffers, (b) whether foliage is opaque in the GI TLAS.
3. Do Phase 0 and Phase 1; stop and show the ray-traced sun shadow comparison before starting Phase 2.

## Progress

### Phase 0 (done)

- Test scenes live in `assets/scripts/gi_test/` and step through camera shots by frame count (key injection does not
  reach Lua `on_key`): `cornell.lua` (closed room, red and green walls, emissive ceiling quad, thin divider),
  `sun_boxes.lua` (sun lit boxes and a sphere on a floor, for shadow comparisons), `bistro_shots.lua` (the Bistro, 200
  frames per shot; edit `MODEL`). Recipe: `.claude/skills/drive-engine/shoot.sh -b release -g lua -k 190:301,390:301 -n
  420 -- --script=assets/scripts/gi_test/bistro_shots.lua --rt-poc --rt-poc-view=shadow-mask`.
- Baseline in `docs/perf/results.md` ("Probe GI, phase 0/1").
- Fixed on the way: `--inject-keys` split a temporary string, which ASan (Debug) rejected as stack-use-after-scope.

### Phase 1 (done): the scene acceleration structure

Implemented as described in `docs/ray-tracing.md` ("Scene acceleration structure"), with two deviations from the plan
above, both agreed: instance records are written by the CPU (not a kernel) into per-frame host visible buffers, and BLASes
are built lazily with a per-frame budget inside the engine pass instead of asynchronously on the compute queue at mesh
upload. Compaction is not done (83 MiB of BLAS for the Bistro): a follow-up.

Exit criterion: the ray traced sun shadows match the shadow-map shadows on the Bistro (aerial shot) and on the
`sun_boxes` scene; the TLAS pass (steady state, 1591 instances, includes the barriers) costs 0.31 ms on an RTX 4070 Ti
SUPER, at the 0.3 ms target. Known artefact: tracing the coarsest LOD makes a faceted terminator on spheres (a jagged
crescent in the shadow mask); the GI will want a normal bias or a finer LOD for round things.

### Phase 2 (done): probe volume and ray trace, no blend

- `ProbeVolume` (`include/rendering/probe_volume.hxx`, `--gi`): game graph passes in `after_lighting`, graphics queue,
  behind the scene TLAS build. A persistent `gi_probe_state` buffer (2 float4 per probe) and a transient `gi_ray_data`
  buffer (8 bytes per ray: RGB16F radiance and a signed half distance, negative for a back face hit).
- Grid `32x8x32`, spacing 1 m, 64 rays per probe (`--gi-probe-spacing`, `--gi-rays-per-probe`), centred on the camera
  in whole probe steps. Toroidal addressing: slot = world index mod counts, so a probe keeps its slot while the grid
  scrolls. Rays are Fibonacci directions rotated by a random rotation per frame.
- `gi_probe_trace.slang`: one ray per thread, ray query against the scene TLAS. Hits are shaded with the sun (shadow
  ray) plus sky irradiance from the prefiltered cube, with a flat albedo of 0.5 (textures and emissive come in phase 4);
  misses return the sky. The sky term on a hit is not occluded, which is the phase 2 approximation until probes feed
  back into the trace. Radiance is not multiplied by exposure.
- `gi_probe_classify.slang`: one thread per probe writes the mean radiance, the back face fraction, and switches a probe
  off when more than 25% of its rays hit a back face. Relocation is phase 3 (the offset is zero for now).
- Debug: `--gi-view=radiance|state|back-faces` shows the nearest probe's value on every surface in the volume (state:
  green active, red off).
- Where the atlases live (decision a): renderer-owned imported images, as settled; none exist yet, the blend is phase 3.
- Numbers (RTX 4070 Ti SUPER, Release, Bistro, 8192 probes x 64 rays = 524k rays): `gi_probe_trace` 0.67 ms,
  `gi_probe_classify` 0.02 ms. The TLAS pass still costs 0.32 ms.
- Checked: `sun_boxes` and Cornell (Debug, validation clean); Bistro street level (Release), where probes inside the
  hedges and building shells come out red and the lit/shadowed ground comes out white/blue as expected. Cornell reads
  black by design (no sun, no ambient), which is correct for sky-and-sun-only shading.
- The volume is only 8 probes (8 m) tall, so a camera high above the ground sees no probes on it.

### Phase 3 (done): blend and shade

- **Moved to the renderer.** The phase 2 game graph passes are now engine passes in `renderer_frame_graph.cxx`, because
  the atlases are renderer-owned images (decision a) and a game pass cannot write an engine image. `Renderer` owns the
  `ProbeVolume` (`renderer.probe_volume()`); the trace pass declares `Use::acceleration_structure_read` on the TLAS
  instead of relying on the queue order. Passes: `gi_probe_trace`, `gi_probe_classify`, `gi_probe_blend`,
  `gi_probe_border`. The phase 2 `--gi-view` overlay passes are gone; the views live in the forward shader.
- **Atlases:** irradiance `10x10` tiles (8x8 octahedral plus a border), distance `18x18` (16x16 plus a border), both
  `RGBA16F`, tiles `count.x * count.y` to a row (2560x320 and 4608x576 for 32x8x32). Linear storage, no perceptual
  encoding (half floats have the range). Irradiance is the cosine weighted mean ray radiance (the SH term's units, so the
  forward shader multiplies it by albedo as before); distance holds mean and mean square of the ray distance, capped at
  5 cells, weight cos^64.
- **Blend:** hysteresis 0.97, 0.85 for the first 8 frames of a probe, 0 on its first frame (a scrolled-in probe takes the
  fresh value outright). Border texels are copied by a separate dispatch once the interior is final.
- **Probe state:** three float4 per probe (offset and active flag; mean radiance and back face fraction; world index and
  age). A slot whose world index changed (the grid scrolled) is reset. Classification and **relocation**: a probe with
  more than 25% back face hits moves along its nearest back face ray, a probe closer than 10% of a cell to a surface is
  nudged away, offsets are capped at 45% of a cell.
- **Trace shading:** hits now use the surface's base colour (factor times texture, level from hit distance), the sun
  with a shadow ray, and sky light through one cosine weighted visibility ray, so a hit inside a room or under a roof
  gets no sky. Misses return the sky. Emissive and local lights are phase 4.
- **Sampling:** `gi_probes.slang` (`gi_sample_irradiance`): trilinear over the 8 probes, backface weight, Chebyshev
  visibility from the distance atlas, crushed tiny weights, normal and view bias, a fade over the outermost probe layer,
  and a coverage factor the forward shader blends with the SH term by. Off probes get no weight. `forward_geom.slang`
  replaces the SH diffuse term with it; `environment_intensity.x` now scales the SH term only (the probes carry
  `Settings::intensity`).
- **Debug:** `--gi-view=irradiance` (the sampled irradiance on grey), `--gi-view=state` (green active, red off).
- **UBO:** grew by a 64 byte `GpuGiBlock` (1088 bytes).
- **Test scene:** `assets/scripts/gi_test/cornell_skylight.lua` (a Cornell room open over the left front half, sun at 50
  degrees, no emissive yet; the lamp comes back in phase 4).
- **Numbers** (RTX 4070 Ti SUPER, Release, Bistro, 8192 probes x 64 rays): trace 0.98 ms, classify 0.02 ms, **blend
  1.44 ms**, border 0.10 ms, TLAS 0.30 ms. Blend is the one to optimise (each of the 64 texels of a probe loops over all
  rays and recomputes the Fibonacci directions): cache the ray directions in groupshared memory, one workgroup per probe.
- **Known gaps:** the volume is 8 m tall; no unit tests for the addressing maths yet; probes are traced every frame
  (budgeting is phase 6); the Bistro run under Debug validation is still not possible (streaming too slow).

### Phase 4 (done): emissive, local lights and multi-bounce

- **Hit shading** (`gi_probe_trace.slang`): own emission (factor x strength x emissive texture), plus albedo times the
  incoming light: the sun (shadow ray), sky through one cosine weighted visibility ray, the explicit emitters and local
  lights through **one stochastic shadow ray** (a light picked in proportion to its unshadowed contribution, weighted by
  the sum), and **the probe volume itself** as last frame left it (every ray that hits a surface samples the atlases there,
  which is the multi-bounce). The trace pass reads the atlases (`Use::sampled`), the lights buffer and the UBO by address.
- **Units fix:** the sun and light terms were missing the `1/pi` of a Lambertian surface (the forward shader's
  `albedo / pi * radiance * ndotl`), so phase 3's bounce was pi times too bright. Everything is now "mean radiance over the
  cosine lobe" (E/pi), which is what the atlases and the SH term hold.
- **Explicit emitters.** A bulb is far too small for random rays to find. `Renderer::update_ray_tracing_scene` lists each
  instance of an emissive batch whose bounding radius is at most 0.3 m (`ProbeVolume::small_emitter_radius`) as an emitter
  (centre, radius, material) in a per frame buffer, and flags the instance (`flag_explicit_emitter`) so a ray that does
  hit it does not add its emission a second time. Larger emissive pieces light the volume only by being hit. Emitters
  are lit like point lights (range `80 * radius`, the forward shader's falloff), colour from the material's average
  emission texture colour, strength `gi_emitter_intensity` (default 200). A bulb's physical emission (about 1 to 10 nits
  from a 5 cm sphere) lights nothing, so this is an art knob; the loader's synthesised lights use 25 but have no bounce
  or probe smoothing. The Bistro at night with 200 to 500 shows the lanterns, string lights and street lamps lighting
  cobbles and walls with warm and coloured pools.
- **No environment lighting:** the forward shader's flat ambient branch (no IBL) now takes the probe volume too; before,
  GI only applied when IBL was valid.
- **cvars:** `gi_intensity`, `gi_emitter_intensity`. Test scene: `assets/scripts/gi_test/bistro_night.lua` (no sun, no
  ambient, exposure +1).
- **Numbers** (Release, Bistro daytime, 47 emitters): trace 1.33 ms (was 0.98 before multi-bounce and lights), blend 1.31 ms,
  border 0.07 ms, classify 0.02 ms, TLAS 0.27 ms; full frame 5.2 ms.
- **Known gaps:** one shadow ray for all local lights (noise on lamps near many others, converged by hysteresis); every
  hit loops over all emitters and lights (fine for dozens, select per probe for thousands, phase 6); an emitter's colour is
  its material's average, so a shared texture atlas of coloured bulbs tints them all alike; blend is still the slow pass.
  The interim point lights (`synthesize_emissive_lights`, `spawn_model_lights`) are untouched, so the editor double
  lights emissives until the phase 6 cvar.

### Phase 5 (done, terrain open): dynamic scenes

- **Moving rigid instances, lights and sun** need no new machinery (the TLAS is rebuilt every frame, lights and the sun are
  read from the UBO each frame). What was missing is how fast the probes follow: hysteresis 0.97 fades over seconds.
  `gi_probe_classify` now keeps a slow average of each probe's mean ray radiance (state float4 `[3]`, so a probe is 64
  bytes) and flags a probe whose mean strays by more than 35% (luminance); the blend then uses hysteresis 0.4 for it until
  the average catches up. `assets/scripts/gi_test/dynamic.lua` (a box sliding in front of coloured walls, an orbiting
  point light, a sun sinking from 60 to 8 degrees) shows the lighting following with no visible lag.
- **Skinned meshes.** The skinning output (`CompressedModelVertex` in the per-frame skin scratch buffer, same layout as the
  arena) is read in place, so every skinned instance gets a bottom level structure of its own, rebuilt each frame
  (`SceneAccelerationStructure::add_skinned`, called where `emit_batches` creates the skin job, so it is the LOD the
  instance is drawn at). A full rebuild rather than a refit: nothing to keep between frames, nothing to invalidate. The
  structures live in a per frame slot storage buffer, created and destroyed with the slot. The TLAS pass now declares a
  read of the skin scratch buffer so it runs after the skin pass; the scratch buffer has the build input usage. A
  skinned instance is added only within `0.5 * max(count.x, count.z) * spacing + 8` m of the camera and while the frame's
  budget of 200,000 skinned vertices lasts (`max_skinned_vertices`): without a budget the 49 characters of the `moving` game
  (1.67 M vertices, 2891 tiny structures) cost 7.2 ms in the TLAS pass; with it 1.7 ms. The per structure overhead
  dominates, not the vertices, so merging a character's submeshes into one structure would be the next saving.
  Instances beyond the budget are not in the scene (they neither occlude nor bounce).
- **Terrain** (decided: a BLAS per chunk). The terrain chunks are ordinary meshes in the geometry arena submitted through
  `submit_mesh`, so they already reach the TLAS through the batches (the earlier "not in the TLAS" was wrong). What was
  broken: a recycled chunk slot keeps its offsets and counts while its vertices are rewritten in place, so a cached BLAS
  went stale. `Submesh::revision` (bumped by `update_submesh_geometry`) is now part of the BLAS cache check. Checked in
  Debug on `--game=basic`: BLASes appear as chunks stream in, no validation errors. (Planet chunks, `PlanetSystem`, were
  not looked at.)
- **Numbers** (Release): `moving` with 49 characters and `--gi`: TLAS pass 1.73 ms (7.2 ms without the budget), skin
  0.13 ms. Debug validation clean on `moving` and `dynamic`.

### Phase 6 (done, see what is left below): polish

- **Second cascade.** A second `ProbeVolume` (`Renderer::probe_volume_far`), same counts at 4x spacing and 4x the biases,
  `update_interval = 2`: its passes (`gi_far_probe_*`) are added only on the frames it updates, and it does not scroll on
  the frames it skips, so a probe that scrolls in is never read with another probe's data. `gi_sample_irradiance` takes the
  near cascade first and the far one where near coverage is below 1 (including the near cascade's fade layer); each
  cascade's multi-bounce samples its own atlases (`trace` push `flags` bit 0), so the far cascade needs nothing from the
  near one. Both get the same emitter list. UBO is 1152 bytes (`gi_far` block).
- **Blend pass** is one workgroup per probe with the probe's rays (direction, radiance, distance) staged once in groupshared
  memory: 0.17 ms, from about 1.3 ms.
- **Interim lights.** `--gi-synth-lights auto|on|off`; `auto` skips `spawn_model_lights` while GI is on. The loader still
  synthesises the lights (`synthesize_emissive_lights`), the editor just does not spawn them.
- **Editor panel, benchmark scenario (`gi_emitters`, load = bulbs), unit tests** (`probe_addressing_test.cxx`: slot
  uniqueness on negative windows, `world_index` inverse, only one layer reassigned when the window scrolls).
- **Numbers** (Release, RTX 4070 Ti SUPER, 1920x1080, `gi_emitters`, both cascades, 64 rays): GPU frame 2.2 ms at 16 bulbs,
  2.2 ms at 128 and 1024 (it was 2.5 and 5.3 ms before the emitter grid below). Near cascade passes at 16 / 128 /
  1024 bulbs: trace 0.13 / 0.15 / 0.35 ms (0.14 / 0.33 / 2.7 ms before), classify 0.01, blend 0.17, border 0.06. Debug validation clean on `cornell`, `cornell_skylight`, `sun_boxes`, `dynamic`, `moving` and
  `basic` (terrain) with both cascades. Tests: 898 of 898.

### Phase 6 (what is left)

- **Probe sleeping / update budgeting** was not built: with trace plus blend at about 0.4 ms on the Bistro-sized scenes the
  budget is met without it. It is the next step if the trace pass grows (many emitters, 256 rays): classify would flag a
  probe whose neighbourhood did not change and the trace would skip it (indirect dispatch).
- The far cascade is not measured on its own (the benchmark reports the median over frames, and it runs every other one).
- Not run: `test/gpu/device_lost_recovery.sh` with GI on; the Bistro under Debug (streaming never finishes there).
- Known gaps from earlier phases: the volume is only 8 m tall, an emitter's colour is its material's average, one shadow
  ray is shared by all local lights, merging a character's submeshes would cut the skinned BLAS cost.
- **Loader fix found with the town**: a glTF material with an emissive factor and no emissive texture rendered black,
  because the fallback was the black default image (glTF means a texture of 1). The fallback is white now. The
  `gi_emitters` scenario had the same mistake (its bulbs were dark; the trace cost is the same either way, it is the
  loop over the emitters).
- **Emitter grid.** Every hit used to loop over all emitters (a distance test each, then a material fetch and a texture
  sample for those in range). `ProbeVolume::finish_emitters` now builds a uniform grid over the frame's emitters on the CPU
  (cell size the largest emitter reach, so an emitter touches at most two cells per axis; at most 16384 cells, the cell
  grows if more would be needed) into the tail of the same upload buffer: a header, per cell (first index, count), and the
  emitter indices. A hit reads its own cell only, which holds every emitter that can reach it. Trace at 1024 bulbs: 2.7 to
  0.35 ms. The grid is rebuilt every frame (about 1200 emitters in the town: negligible on the CPU) for both cascades.
