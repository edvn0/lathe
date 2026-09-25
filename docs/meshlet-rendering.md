# Meshlet rendering

Every scene-geometry pass (shadow cascades, depth prepass, forward -- and
their mask/blend variants) draws through task + mesh shaders. There is no
vertex-shader geometry path left; ImGui, the debug line renderer and the
fullscreen composite/post passes are not scene geometry and are unchanged.

## Data

- **Build** (`include/assets/meshlet.hxx`, `src/assets/meshlet.cxx`):
  meshoptimizer splits each distinct index buffer into meshlets of at most
  64 vertices / 124 triangles. Bounds (sphere + normal cone) are computed
  from the *half-float decoded* positions the mesh shader reads, so a
  sphere can never be tighter than the geometry actually drawn.
- **Storage**: `MeshGeometry::meshlets` (`MeshletSlice`) holds two
  `GeometryArena` ranges -- `GpuMeshlet` descriptors, and a `uint32` data
  array of meshlet-local vertex indices followed by packed triangles
  (`i0 | i1 << 8 | i2 << 16`). `Renderer::create_mesh` rejects a submesh
  whose LODs lack meshlets; `destroy_mesh` retires them with the rest of the
  geometry.
- **Models**: built per LOD off the render thread by
  `prepare_primitive_gpu_data()` (`load_model.hxx`), together with vertex
  compression -- as the last step of `finalize_primitive_cpu`, which
  `ModelStreamer` runs per primitive on `thread_pool()`; procedural meshes
  get it in `to_model_cpu_data()`. `step_model_gpu_upload` only uploads
  the prebuilt `MeshletBuild`s (and warns if handed an unprepared
  primitive, building it late as a fallback). Aliased LODs alias the
  meshlets too. The synchronous `Renderer::load_model()` still finalizes
  on its caller's thread, like the rest of its CPU work.
- **Terrain**: `TerrainSlotPool` builds one index-order topology for the
  canonical chunk index buffer and shares its data range across every slot;
  each slot has its own descriptor range whose bounds are recomputed in
  `write()` whenever the slot's vertices are rewritten.

## Draws

- One `GpuTaskCommand` per batch (32 bytes: the three
  `VkDrawMeshTasksIndirectCommandEXT` group counts plus `first_instance`,
  `instance_count`, `meshlet_count`), drawn with
  `vkCmdDrawMeshTasksIndirectEXT` at that stride.
- A batch needs `instance_count * ceil(meshlet_count / 32)` task groups,
  spread over X/Y by `set_task_group_counts` (C++ and `frustum_cull.slang`
  must agree) so no dimension exceeds the guaranteed 65535.
- Shadow pass: CPU-built, un-culled commands. Main view: `mainCs`
  frustum-culls instances as before, compacts survivors and rewrites the
  group counts for the survivor count.
- `SV_DrawIndex` restarts at 0 per indirect call, so
  `render_pass::detail::draw_task_commands` re-points `PC::task_commands`
  at the first command of each call.

## Task-shader culling (`assets/shaders/meshlet_task.slang`)

One task workgroup = up to 32 meshlets of one instance, one lane each:

- **Frustum**: world-space bounding sphere (radius scaled by the largest
  axis scale, padded by `sqrt(2) * wind_strength`) against 6 planes -- the
  camera's for prepass/forward, the cascade's own for the shadow pass
  (`frustum_planes_buffer` holds camera + 4 cascades, 30 planes).
- **Backface cone**: opaque prepass + forward only, since those are the only
  back-face-culled draws. Skipped for wind-swayed materials, non-uniform
  scale and mirrored transforms. The prepass and forward pass use identical
  flags because forward depth-tests `EQUAL` against the prepass.
- Survivors are compacted in lane order (deterministic) into the payload,
  then a single `DispatchMesh`.

`Renderer::set_meshlet_culling(false)` (Lighting > Debug > "Meshlet culling
(task shader)") disables per-meshlet culling for A/B debugging. With
`meshShaderQueries` available, Scene stats shows task/mesh shader
invocations for the forward pass.
