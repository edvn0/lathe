# Water

Water is a component, the way Unreal has water bodies and Unity HDRP a Water Surface: put `Components::Water`
(`include/scene/components.hxx`) on an entity with a Transform and the engine draws it. `--game=water` is the demo:
keys 1-3 pick a shot (pool from above, across the pool, out to sea), P pauses the orbit, M toggles MSAA.

## The component

`Components::Water` is data only. The entity's position places the body and its yaw turns a lake; its scale is ignored.

- `body`: `ocean`, an open sea that follows the camera to the horizon at the entity's height, clipped to the `extent`
  rectangle about the entity (its water zone, which keeps it off land: the demo's sea starts at the coast); or `lake`, a
  finite rectangle of `extent` metres (a lake, pool or pond).
- Look: `colour` (what light scatters back out of deep water), `roughness` (reflection blur), `absorption` (per metre,
  per channel), `shore_fade` (m of depth over which the surface fades in), `refraction_strength`, `foam`, `whitecaps`,
  `caustics`.
- Waves: `wave_scale` (frequency, 1/m), `wave_speed`, `wave_height` (m the surface moves up and down),
  `normal_strength` (extra shading detail).
- `visible`.

`Water::pool(size)` is a calm, clear pool; `Water::ocean(zone)` a rough sea with whitecaps. The component is saved with
the scene (section `waters`), copied into the play scene, editable in the editor's inspector (Add Component > Water,
with presets) and cloned with the entity.

## The system

`WaterSystem` (`rendering/water_system.hxx`, owned by `Runtime`) runs from the scene submission every frame. For each
Water entity it makes and owns a water material (`MaterialCreateInfo::water`, which sets `GpuMaterial::flag_water`; an
internal detail that games never touch) and a grid mesh, updates the material when the component changes, remakes a
lake's grid when its size needs a different cell count, and submits the surface through the ordinary model path as a
blended batch. When the component or its entity goes away (Ctrl-R, a scene clear, a game switch) the mesh and material
are released `retire_frames` submits later, once no frame in flight still draws them.

- Lake: an even grid, about 12.5 cm cells (8 to 256 a side), scaled to the extent.
- Ocean: one shared grid, 320 x 320 cells over 1200 m, packed under the camera and widening exponentially (about 5 cm
  at the centre, 1.4 m at 30 m, 26 m at the edge). It is re-centred on the camera in 8 m steps
  (`WaterSystem::ocean_snap`), so it does not slide every frame, and the vertex stage folds the vertices outside the
  water zone onto its edge (`GpuMaterial::water_zone`), where their triangles collapse.

## Asking where the water is

`WaterSystem::water_height(registry, xz, time)`, `water_normal` and `is_underwater(registry, point, time)` answer from
the components alone, for gameplay (buoyancy, boats, splashes). They run `water_waves::field`
(`rendering/water_waves.hxx`), a CPU copy of the shader's wave field: the wave directions and octave steps are tables
present in both, and every constant from the table to `water_wave_field` in `water.slang` must match it (both files
say so). The directions used to come from a `sin()` hash, which a CPU cannot reproduce on every GPU. The time is the
shader time, `Renderer::frame_time()`. A lake wins over an ocean where both cover a point, and among lakes the highest.
The height includes the waves the mesh shows close to the camera; the normal is the displaced surface's slope, not the
shading's extra detail. Lua has `scene.water_height(x, z [, time])` and `scene.is_underwater(x, y, z [, time])`.
`test/water_waves_test.cxx` checks the field (determinism, bounds, zero mean, fading) and the queries; it cannot run the
shader, so it does not compare GPU output.

## Frame graph

Water needs the scene behind it, so on frames where a water material is among the blended batches
(`RendererFrame::scene_copy_needed`) the forward pass is split in two (`src/rendering/renderer_frame_graph.cxx`):

1. `forward`: the opaque, double-sided and masked draws and the sky (`ForwardPhase::opaque`).
2. The copy: with MSAA, the first pass resolves its colour into `scene_colour_copy` and keeps the multisampled image
   for the second pass, and the depth copy is the prepass's single-sample resolved depth, so nothing is copied at all.
   Without MSAA a `scene_copy` transfer pass copies the HDR and depth images into `scene_colour_copy` and
   `scene_depth_copy`.
3. `forward_blend`: the blended draws, the overlays and the game's scene draws (`ForwardPhase::blend`). It loads the
   colour, outline and depth attachments and does the final resolve.

The copies reach the shader as bindless indices in the forward push constants (`PC.scene_colour_index`,
`PC.scene_depth_index`, sampled through `sampled_2d` / `sampled_2d_depth`). Everything is at the render extent, so
the render scale needs no special handling. A frame without water keeps the single `forward` pass and pays nothing.

## Shading (`assets/shaders/water.slang`)

Only the blended pipelines are built with `BLEND_PASS`, so the opaque shaders carry none of this.

- **Waves**: eight directional waves (sharpened sines, `exp(sin - 1)`) scattered within about +-50 degrees of one
  wind heading, with frequencies rising about 1.4 per octave, scrolling at the deep-water speed `sqrt(g k)` over a
  slow domain warp. Waves shorter than a few footprints fade out (the pixel footprint in the fragment shader).
- **Displacement**: in the blended pipelines the vertex/mesh stage (`forward_vertex` in `forward_geom.slang`) lifts
  upward-facing water vertices by `wave_height * offset` (`water_displacement`), where `offset` is the same field
  centred on its mean (faded waves add nothing, so the surface does not sink where they fade) and scaled so a lone
  crest reaches `wave_height`. The vertex footprint grows with camera distance (3%, at least 5 cm), roughly matching
  the mesh spacing, so the mesh does not alias the short waves. The fragment shader evaluates the same field at the
  same xz (displacement is vertical only, no Gerstner pull), and its normal is the displaced surface's slope
  (`offset_gradient * wave_height`) plus `normal_strength` detail, which includes the waves too short for the mesh.
  The grids come from `make_water_grid_mesh` (`assets/primitive_meshes.hxx`), whose spacing can widen exponentially
  away from a focus (see The system).
- **Refraction**: the colour copy is sampled at the pixel offset by the wave slope (less with distance). An offset
  sample whose depth is in front of the surface (a pillar standing in the water) is rejected for the unshifted one.
- **Absorption**: the depth copy gives the length of the view ray under water; Beer-Lambert with
  `absorption` (per metre, per channel) fades the scene behind into the scattered colour. The sky behind
  the water counts as infinitely deep.
- **Reflection**: screen-space reflection, marching the reflected ray in world space (32 growing steps with a
  per-pixel jitter, 5 bisection steps; the march follows a flatter normal than the shading) against the depth copy; where it misses, leaves the screen or fades out at the edges, the prefiltered
  environment cube is used. Reflected rays are kept above the horizon. Planar reflection was not used: it needs a
  second scene render per water plane.
- **Fresnel**: Schlick with F0 = 0.02 mixes refraction and reflection, and the sun adds a GGX highlight.
- **Depth**: the water writes no depth, so refraction, the water column, SSR and foam all use the scene copy and
  the fragment's own (displaced) position; where a trough dips below a beach the depth test simply hides it.
- **Shoreline**: the surface fades in over `shore_fade` metres of water depth (the alpha), so it meets the ground
  softly.
- **Foam**: a narrow, noise-broken band where the water is shallower than 0.2 m (shorelines and anything breaking the
  surface), plus whitecaps on high crests where the surface is squeezed together. With vertical displacement only
  there is no real Jacobian, so the field's analytic Laplacian (strongly negative on sharp crests) times the wave
  height stands in for the compression; `whitecaps` scales it. The foam is lit by the ordinary forward lighting
  (sun, spot and point lights, IBL) as a rough white surface and mixed over the water.
- **Caustics**: a domain-warped `sin x sin y` network projected along the sun onto what is seen through the water,
  brightening it multiplicatively, strongest just under the surface and gone with depth. It is applied in the water
  shader, so it only shows where you look through water, not on submerged surfaces seen from elsewhere, and it does
  not know whether the floor point itself is in shadow (it uses the surface's sun shadow).

## Culling

A displaced surface leaves its mesh's bounds, so its batch's cull padding (`GpuCullBounds::wind_padding`) is the
larger of the wind strength and the wave height, and the frustum cull now pads every axis by it (it used to pad xz
only, for wind). Meshlet culling adds the wave height to each meshlet's radius. Water casts no shadows, so the shadow
culls are unchanged, and the blend sort is unaffected (water sorts first anyway).

## Limits

- The ocean grid has no LOD beyond its graded spacing. When it re-centres (every 8 m of camera travel) the far vertices
  land on new points of the wave field, so distant silhouettes can pop slightly; the shading does not move. It
  follows the previous frame's camera (the scene is submitted before the frame's camera is chosen). Coarse far cells
  smooth the waves into a mirror near the horizon.
- The ocean zone is an axis-aligned rectangle; there are no coastline splines or exclusion volumes, so an ocean under a
  pool must be kept out with its zone (as the demo does).
- The edges of a displaced surface move with the waves: against a wall the gap can show for a moment (a few cm in
  the pool, hidden under the terrace and beach for the sea).

- **Blended surfaces and water.** Water draws before every other blended batch (`blend_order::draws_before`): it
  paints the scene copy behind itself, which holds no blended surfaces, so glass behind or under the water would
  otherwise vanish. The cost is that glass under the surface is drawn over the water without being tinted by it.
- Refraction is a screen-space offset, not a bent ray, and the refracted point is assumed to lie along the unbent
  view ray when measuring the water depth. Things outside the screen are not refracted or reflected (the environment
  fills in for reflection).
- The camera is assumed to be above the water: the back faces of a single-sided water mesh are dropped like any
  single-sided blended mesh, and there is no underwater view.
- The water casts no shadows and writes no depth (it is blended), so depth-based effects after the forward pass (AO,
  a game's `after_lighting` passes) see what is under it.
- The pipeline statistics query covers only the opaque pass on split frames.

## Blend ordering

Unrelated to water, but changed with it: blended batches are sorted back to front by camera distance to their bounds
centre, ties go to the larger bounds first and then to a stable id (material, mesh, submesh, LOD), so concentric
meshes keep one order as the camera moves. A double-sided blended batch is emitted as two indirect commands, back
faces then front faces (`DrawCommand::blend_faces`; the fragment shader discards the other side, so the commands
still go in one multi-draw). Single-sided blended batches now drop their back faces. LOD-driven and skinned
double-sided batches are not split (both sides in one draw, as before). The blended commands are drawn in runs of one
draw path (meshlet or indexed), so the order also holds between meshlet and indexed batches, which used to be drawn
all meshlet first, then all indexed.
