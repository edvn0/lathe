# Hardware ray tracing: the proof of concept and the road to the full version

`lathe --rt-poc` traces one hardcoded sphere into the running scene with `VK_KHR_ray_query`. It exists to get
acquainted with acceleration structures and traversal, not to be a renderer feature. This page says what it does, how
it is built, and what the full version has to change.

```
lathe --game=materials --rt-poc
```

Needs a device with `VK_KHR_ray_query` + `VK_KHR_acceleration_structure` (the startup log prints
`VK_KHR_ray_query support: yes|no`). Without it the flag only logs a warning.

## What it draws

One compute pass, `rt_poc`, in `GameSlot::after_lighting`:

- A ray per pixel from the camera against the TLAS: a hit in front of the scene depth draws the sphere (flat shaded
  from the triangle's face normal, lit by the renderer's directional light).
- A ray from every scene pixel towards the sun: a hit means the sphere shadows that pixel, which is darkened.
- The result goes to a new image and `replace_scene_colour` hands it to composition, like a compute effect.

The sphere is a 320 triangle icosphere (`make_sphere`), placed at `RayTracingPoc::sphere_centre` with
`sphere_radius`. Move it by editing those two constants.

## Where the code is

| Piece | File |
| --- | --- |
| Device features and extensions, `VulkanContext::ray_query_supported` | `src/runtime/vulkan_bootstrap.cxx`, `include/gpu/context.hxx` |
| TLAS binding in the bindless set (binding 6, only with ray query) | `src/gpu/gpu_resource_table.cxx` (`set_acceleration_structure`) |
| Mesh, BLAS, TLAS build, per-frame pass declaration | `src/rendering/ray_tracing_poc.cxx` |
| Shader | `assets/shaders/rt_poc.slang` |
| Wiring: flag, camera matrices, hook, teardown | `src/runtime/engine_main.cxx`, `src/runtime/runtime.cxx` |

## How it works, in the order it happens

1. **Device.** `rayQuery` and `accelerationStructure` features are enabled (with `VK_KHR_deferred_host_operations`,
   which the acceleration structure extension requires). `bufferDeviceAddress` was already on.
2. **Geometry.** Vertices (`float3`) and `uint32` indices go into host visible buffers with
   `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` and `SHADER_DEVICE_ADDRESS` usage.
3. **BLAS.** `vkGetAccelerationStructureBuildSizesKHR` gives the storage and scratch sizes; a storage buffer
   (`ACCELERATION_STRUCTURE_STORAGE`) backs the `VkAccelerationStructureKHR`; the build runs on the device.
4. **TLAS.** One `VkAccelerationStructureInstanceKHR` (a 3x4 row major transform and the BLAS device address), built the
   same way. A barrier (`ACCELERATION_STRUCTURE_BUILD` write to build read/write) sits between the two builds, which
   also lets them share one scratch buffer. The scratch address is rounded up to
   `minAccelerationStructureScratchOffsetAlignment`.
5. **Binding.** The TLAS handle is written once into binding 6 of every frame's bindless set.
6. **Per frame.** `RayTracingPoc::declare` adds the compute pass through `GameGraph::add_compute`. Push constants
   (104 bytes): the inverse view-projection, the vertex and index buffer addresses (to rebuild the face normal in the
   shader), the three bindless image indices, and the sun direction.
7. **Shader.** `RayQuery<RAY_FLAG_NONE>`, `TraceRayInline`, `Proceed()`, then `CommittedStatus()`,
   `CommittedPrimitiveIndex()` and `CommittedRayT()`. One `Proceed()` is enough because the geometry is opaque (no
   candidate hits to accept or reject).

## Why ray query and not the ray tracing pipeline

`VK_EXT_shader_object` has no ray tracing stages: raygen, hit, miss, intersection and callable shaders only exist as
`VkPipeline` shader groups. Ray query is traversal inside an ordinary compute (or fragment) shader, so the pass stays a
shader object, needs no shader binding table, and fits `GameGraph` unchanged. It is the same hardware and the same
acceleration structures.

## Things the PoC found out the hard way

- **A device address cannot become an acceleration structure here.** Slang compiles
  `RaytracingAccelerationStructure(uint64)` to `OpConvertUToAccelerationStructureKHR`, which needs
  `SPV_KHR_ray_tracing`, which Vulkan only allows with `VK_KHR_ray_tracing_pipeline`. Validation rejects the shader
  object. Hence the descriptor in binding 6. With the pipeline extension enabled the push constant address works again.
- **The depth buffer is reversed.** Depth 1 is the near plane, 0 the far plane and the sky; NDC z is `1 - depth`
  (see `view_pos_from_depth` in `gtao.slang`).
- **The viewport is flipped.** The renderer uses negative height viewports, so NDC y points up while pixel rows point
  down: rebuild NDC as `(2u - 1, 1 - 2v)`. Getting this wrong mirrors the sphere and every shadow ray origin vertically
  about the screen centre, which looks plausible until you view the `depth-test` debug view from a low camera.
- **Descriptor writes.** Binding 6 is not update-after-bind, so it is written once before frames use the set.

## Limits of the PoC

- Static: built once, never rebuilt, refit or compacted. One BLAS, one instance.
- The instance is a translation and a uniform scale, which is what lets the shader use the object space face normal as
  the world one.
- Not part of the scene: no entity, `Model` or material is involved, and it does not respond to the editor.
- Only `after_lighting`, only compute, no alpha test, no tests in `test/`.
- The build is synchronous (`one_time_submit`), at startup.

## Migrating to the full version

Roughly in the order it pays off.

### 1. Real meshes in the BLAS

- Each engine mesh needs a BLAS built from its own vertex and index buffers, which must be created with
  `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` and `SHADER_DEVICE_ADDRESS` usage (check what the mesh and resident
  geometry buffers are created with today; the first flag is probably missing).
- Vertices are `CompressedModelVertex` (20 bytes, quantised position, octahedral normal; `gpu/model_vertex.hxx`).
  `VK_FORMAT_R16G16B16_UNORM` positions need the dequantisation scale applied through the instance transform, or keep a
  second float position stream just for the BLAS. Check which the meshlet path leaves resident.
- Build BLASes when a mesh is uploaded (asynchronously, on the compute queue), compact them
  (`ALLOW_COMPACTION`, a query pool for compacted size, `vkCmdCopyAccelerationStructureKHR`), and store the handle and
  address next to the mesh.

### 2. Instances from the scene

- Build the TLAS every frame from the same instance list the renderer already uploads (`source_transforms`,
  `source_draws`). A kernel writes `VkAccelerationStructureInstanceKHR` records into a device buffer, then
  `vkCmdBuildAccelerationStructuresKHR` reads them (`arrayOfPointers = false`).
- `instanceCustomIndex` (24 bits) carries the draw or material index so a hit can find its material, and
  `mask` / `instanceShaderBindingTableRecordOffset` are there when needed.
- Keep one TLAS per frame in flight, and its scratch buffer. Rewriting the descriptor each frame means either one
  binding element per frame slot or `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT` on binding 6.

### 3. Make the frame graph aware of acceleration structures

The PoC gets away without this because its builds finish before the first frame. A per-frame TLAS must be a frame graph
resource, otherwise barriers and queue ownership are not derived:

- Add `Use` values for acceleration structure build (write) and ray query (read) in `frame_graph`, translated in
  `frame_graph/vk_translate.cxx` to `PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR` with
  `ACCELERATION_STRUCTURE_READ/WRITE_BIT_KHR`, and to `COMPUTE_SHADER` + `ACCELERATION_STRUCTURE_READ` for the trace.
- The build pass needs a command buffer: `GameGraph` does not expose one, so this becomes an engine pass in
  `renderer_frame_graph.cxx` (see `docs/adding-a-pass.md`, "In the engine"), not a game pass. Ray traced effects can
  then declare a read of the TLAS resource and stay `GameGraph` passes.
- Builds run on a compute capable queue; a pass that reads the TLAS from another queue needs the usual ownership
  transfer, which the graph already derives for buffers.

### 4. Moving and deforming geometry

- Rigid movers: only the TLAS changes.
- Skinned meshes (`skin.slang` already writes skinned vertices to a scratch buffer): build the BLAS with
  `ALLOW_UPDATE` and refit it each frame (`MODE_UPDATE`, same source and destination), rebuilding when the pose drifts
  far.
- Terrain and meshlet LODs: pick the LOD level the BLAS represents deliberately (usually a coarse one), since the
  traced geometry does not have to match what is rasterised.

### 5. Shading what a ray hits

With ray query the "closest hit shader" is code after `Proceed()` in the compute shader:

- Read `CommittedInstanceCustomIndex()`, `CommittedPrimitiveIndex()` and the barycentrics, fetch the three vertices
  through the buffer addresses (a buffer of per-instance vertex/index addresses indexed by the custom index), and
  interpolate normal, tangent and UV.
- Sample material textures through the bindless table with explicit LODs (`SampleLevel`, derived from ray cones or a
  fixed bias), since there are no screen space derivatives in a ray.
- Alpha tested geometry: use `RAY_FLAG_NONE` without the opaque geometry flag and loop on `Proceed()`, calling
  `CommitNonOpaqueTriangleHit()` when the alpha test passes.
- Typical first features, in order of cost: ray traced shadows (the PoC's sun ray, with the scene's real light), ambient
  occlusion, reflections, then a path tracer.

### 6. When (and whether) to leave ray query

Stay on ray query while the work is "a few rays per pixel from one shader". Move to the ray tracing pipeline
(`VK_KHR_ray_tracing_pipeline`) when:

- the scene has many materials and one uber shader with a switch on the material gets slow or unmanageable;
- a path tracer's divergence matters (the pipeline lets the driver regroup invocations, and
  `VK_EXT_ray_tracing_invocation_reorder` reorders explicitly);
- you want any-hit / intersection shaders as separate stages, or recursion and callable shaders.

That is a different pipeline model and **it conflicts with the project rule that every shader is a shader object**:
ray tracing shaders only exist as `VkPipeline` groups. Agree to that exception first, scope it to the ray tracing pass
alone, and keep everything else on shader objects. The work:

1. Enable `VK_KHR_ray_tracing_pipeline` and `rayTracingPipeline` in `vulkan_bootstrap.cxx`; query
   `VkPhysicalDeviceRayTracingPipelinePropertiesKHR` (`shaderGroupHandleSize`, `shaderGroupHandleAlignment`,
   `shaderGroupBaseAlignment`, `maxRayRecursionDepth`).
2. Split `rt_poc.slang` into stages: the per-pixel setup and `TraceRay` become `[shader("raygeneration")]`, the code after
   a hit becomes `[shader("closesthit")]` (with a payload struct), and the sky / "no hit" path
   `[shader("miss")]`. The shadow ray is a second `TraceRay` with its own miss shader (or a payload flag).
3. Create a `VkPipeline` with `vkCreateRayTracingPipelinesKHR` and one shader group per stage set, fetch the group
   handles with `vkGetRayTracingShaderGroupHandlesKHR`, and write them into a shader binding table buffer
   (`SHADER_BINDING_TABLE_KHR` usage). One record per material for the hit group, indexed by instance and geometry.
4. Dispatch with `vkCmdTraceRaysKHR` and four `VkStridedDeviceAddressRegionKHR` (raygen, miss, hit, callable).
5. The pipeline is created through the renderer's pipeline registry like the others; the registry has no ray tracing
   kind today, so that is new code, and the shader hot reload path has to rebuild the pipeline and its SBT.
6. With the pipeline extension on, `SPV_KHR_ray_tracing` is legal, so a TLAS can again be passed as a device address and
   the binding 6 descriptor becomes optional.

The acceleration structure code (steps 1 to 4 above) does not change between ray query and the pipeline; only how
traversal is started and where the hit code lives.
