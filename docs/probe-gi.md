# Probe GI (DDGI on ray query): plan and handoff

Status: **plan, nothing implemented.** Written at the end of the session that added crisp editor icons, `ui_font_size`
and point lights synthesised from emissive meshes (commit `e59a6f1`, branch `feat/editor-ui-and-emissive-lights`). A new
session should be able to start from this page alone. Read `docs/ray-tracing.md` and `docs/adding-a-pass.md` first;
this page only repeats what it must.

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
