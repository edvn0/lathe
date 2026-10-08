# GPU skinning

Status: implemented end to end in the renderer. The GPU path was run on lavapipe (llvmpipe, Vulkan validation on, no
errors) through `--game=moving --player --screen-type=headless` with `LATHE_MOVING_SKINNED=1`: deformed, animated
characters with shadows. Not run on a hardware GPU; jump pose and the lying pose were not inspected closely.

## Pipeline

```
glTF skin + JOINTS_0/WEIGHTS_0 --> ModelCpuPrimitive::skin (u16 joints, unorm16 weights)
                                   ModelCpuData::animation (skeleton + clips)
upload (step_model_gpu_upload)  --> GeometryArena: GpuSkinVertex stream (8 B/vertex, MeshGeometry::skin)
                                   rest vertices stay CompressedModelVertex (20 B)
per frame:
  gameplay palette  -> Renderer::set_skin_palette / append_skin_palette
  submit_model_instances(..., palette_offsets)
  prepare_frame: per skinned instance x submesh -> scratch slice + GpuSkinJob, GpuDraw::vertex_address = slice
  graph: skin_upload (transfer) -> skin (compute) -> shadow / depth prepass (early, late) / forward read the scratch
```

Skinning is a **pre-skin pass**: `assets/shaders/skin.slang` deforms the rest vertices into a per-frame scratch buffer
of `CompressedModelVertex` (20 B, model space, f16 positions, octahedral normal/tangent). Draw shaders already fetch
`draw.vertex_address[id]`, and meshlet/index data is shared with the rest geometry, so no depth/shadow/forward shader
knows about skinning and all of them agree on the pose. The model matrix is applied by the existing shaders.

## Data

- `SkinVertex` (importer, `assets/model_skin.hxx`) -> `GpuSkinVertex` (`gpu/skinning.hxx`): 4 x u8 joints, 4 x unorm8
  weights (renormalised in the shader). Joint indices are the importer's skeleton order and must fit 8 bits: a
  primitive with more than 256 joints (or a skin stream that does not match its vertex count) uploads unskinned and
  logs a warning.
- `MeshGeometry::skin` is a `GeometrySlice` shared by all LODs (like `vertices`); `Renderer::destroy_mesh` retires
  it. `Renderer::create_mesh`/`update_submesh_geometry` validate its size (`vertex_count * 8`).
- `Model::animation` / `ModelSlotData::animation` keep the `ModelAnimationData` (skeleton + clips) reachable:
  `Renderer::model_animation(ModelHandle) -> shared_ptr<ModelAnimationData const>` (null for unskinned models).
- All LODs share one vertex stream, so a skin job deforms every vertex of the submesh whatever LOD is drawn; reduced
  LODs reduce index/meshlet work, not skinning work.

## Bounds

Deformed vertices leave the rest-pose volume, so for skinned primitives (at upload):
- meshlet `cone_cutoff = 1` (cone culling off) and meshlet radius `+= skin_inflate`;
- submesh and model bounds are grown by `skin_inflate` on every side (feeds GpuCullBounds, Hi-Z tests, shadow fit).

`compute_skin_inflate()` (`assets/load_model.hxx`) = 1.25 x the largest |skinned - rest| over the bind pose plus 16
phases of every imported clip, evaluated on up to 1024 sampled vertices per skinned primitive. Models with a skin but
no clips fall back to the model's bounding radius. Palettes that go outside the clips (IK, procedural motion, extreme
scale) can exceed the bound and clip at the screen/shadow edge; keep them within it or use a clip that spans the
range. The sampling can also under-estimate when the extreme vertex was not sampled (the 1.25 factor is the margin).

## Submission API

```cpp
renderer.set_skin_palette(palette);                 // replaces this frame's palette, offset 0
auto offset = renderer.append_skin_palette(chars);  // returns the offset of what it added
renderer.submit_model_instances(model, transforms, material_override, /*resident_revision*/ 0, palette_offsets);
```

- The palette is `joint_world * inverse_bind` per joint, column-major (what `Animation::compute_skinning_palette` /
  `AnimationBatch::palette()` write). `palette_offsets[i]` is the index of instance i's first matrix; the instance
  needs `skeleton.joint_count()` matrices there. `Components::InstancedModel::palette_offsets` carries them and
  `main.cxx` forwards them.
- Call `set/append_skin_palette` before `submit_model_instances` each frame; the palette is consumed and cleared by
  `prepare_frame`.
- Skinned models never use the resident (GPU-picked LOD) path; they take the per-instance path (CPU LOD by instance
  position, `batch.palette_offsets` parallel to `batch.transforms`).
- Individual `submit_model` calls of a skinned model have no palette: they draw the rest pose.

## Limits and fallback

`RendererCreateInfo`, per frame in flight:

| field | default | meaning |
| --- | --- | --- |
| `maximum_skin_palette_matrices` | 65536 (4 MiB) | `set/append_skin_palette` fail with `capacity_exceeded` past it, adding nothing |
| `maximum_skin_jobs` | 16384 | skinned instance x submesh pairs per frame |
| `skin_scratch_bytes` | 32 MiB | deformed vertices, 20 B each (~1.6M vertices), 16 B aligned slices |

An instance beyond the job or scratch budget, without a palette offset, or whose offset + joint count exceeds the
palette, draws in its rest pose (it is not skipped) and is counted in `FrameStats::skin_fallback_instance_count`.
`FrameStats` also reports `skin_job_count`, `skinned_vertex_count` and `skin_scratch_bytes_used`. 5000 characters of
20k vertices need 2 GB, so crowds need aggressive vertex budgets or a larger scratch; there is no CPU frustum or
distance cull before skinning in this version (every submitted instance is skinned, even off-screen ones).

## Frame graph

Only when a frame has skin jobs (an unskinned scene's graph is unchanged):
- `skin_upload` (transfer): copies the used ranges of the host-written `skin_upload` buffer (palette | jobs | chunk
  table; palette padded by 256 matrices so a stray joint index stays inside the buffer) to `skin_input`.
- `skin` (compute): reads `skin_input`, writes `skin_scratch` (`write_discard`). One thread per vertex, 64-thread
  chunks, binary search over `skin_chunk_table` for the job; dispatch `(min(total, 65535), ceil(total / 65535), 1)`.
- `shadow_pass`, `depth_prepass` (early), `depth_prepass_late` and `forward` declare `read(skin_scratch,
  shader_read, <their stages>)`; this is what yields the compute -> raster barrier (the passes dereference the
  scratch by device address, which the graph cannot see otherwise). `test/frame_graph_skin_test.cxx` asserts it.
- Shadows: skinned batches count as animated casters, so far cascades are re-rendered when their update period is
  due and the cached atlas never shows a stale pose (same mechanism as wind).

## CPU reference and tests

`skin_vertex()` / `skin_compressed_vertex()` in `include/gpu/skinning.hxx` mirror `skin.slang` (blended matrix,
normal via cofactor/inverse-transpose, tangent via the linear part, unweighted vertices stay at rest).
Tests: `test/skinning_test.cxx`, `test/frame_graph_skin_test.cxx`, `test/skin_inflate_test.cxx`.

## Not done

- Motion vectors from the previous frame's skinned positions (needs a second scratch buffer).
- GPU-driven skin-after-cull, or a CPU frustum/distance cull before skinning; skipping re-skinning for characters
  whose palette did not change.
- More than 256 joints per skin; 8 bit joint indices are baked into `GpuSkinVertex`.
