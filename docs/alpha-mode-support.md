# Alpha Mode Support (opaque / mask / blend) — Implementation Spec

## Context

`Material::alpha_mode` (`scene_types.slang`) already has three states — `opaque`, `mask`, `blend` — but nothing in the renderer currently reads it. All draws go through one indirect draw stream, one PSO per pass, no cull-mode distinction, no CPU-side sorting. This spec adds real handling for `mask` and `blend`, split CPU-side into three separate draw ranges per frame so each alpha mode gets its own cull mode, PSO, and (for blend) depth/sort behavior.

Files known to be affected: `renderer.hxx`/`renderer.cxx`, `scene_types.slang`, `forward_geom.slang`, `depth_prepass.slang`, `shadow_depth.slang`. Exact paths for `PipelineRegisterInfo` and `RendererFrame` struct definitions were not confirmed in this conversation — locate and adjust accordingly.

## Why CPU-side splitting, not a single shared PSO with branching

- **Cull mode is dynamic state, set once per `vkCmdDrawIndexedIndirect` call.** Mask geometry (foliage, cutout planes) generally needs `VK_CULL_MODE_NONE`; opaque needs `VK_CULL_MODE_BACK_BIT`. They cannot share a draw call.
- **Prepass and forward must agree on cull mode per draw**, or depth written in the prepass won't match what forward rasterizes, causing z-fighting or missing fragments. This means mask needs a dedicated draw call in every pass that touches depth: depth prepass, shadow pass, and forward pass.
- **`discard` disables early-Z for the whole PSO it's used in**, even for invocations that don't hit the branch. Keeping opaque's prepass/shadow pipelines discard-free (as they are today, vertex-only) preserves early-Z / cheap depth-only rendering. Mask needs its own fragment+discard variant instead of folding into the opaque PSO.
- **Blend needs different pipeline state entirely** (blending enabled, depth write off, no prepass contribution), and needs draws sorted back-to-front, so it cannot share a draw call or PSO with opaque/mask either.

Net result: **three CPU-side buckets — opaque, mask, blend — each with its own contiguous range in the draw/indirect buffers.**

## 1. `prepare_frame` — three-way batch split

Currently `prepare_frame` builds one `unordered_map<batch_key, batch_entry>` and flattens it into `frame.draws` / `frame.transforms` / `frame.indirect_commands` / `frame.batch_bounds` in a single pass.

Replace the final flattening loop with a three-way partition keyed by `material->alpha_mode` (looked up via `material_storage_.get(batch.material)`):

```cpp
std::vector<batch_entry const *> opaque_batches;
std::vector<batch_entry const *> mask_batches;

struct pending_blend_batch {
    batch_entry const *entry;
    float camera_distance_sq;
};
std::vector<pending_blend_batch> blend_batches;

for (auto const &[key, batch] : batches) {
    auto const *material = material_storage_.get(batch.material);
    auto const alpha_mode = material != nullptr ? material->alpha_mode : AlphaMode::opaque;

    switch (alpha_mode) {
        case AlphaMode::opaque:
            opaque_batches.push_back(&batch);
            break;
        case AlphaMode::mask:
            mask_batches.push_back(&batch);
            break;
        case AlphaMode::blend: {
            auto const centroid = /* world-space centroid of this batch's bounds */;
            blend_batches.push_back({&batch, glm::length2(centroid - camera_position)});
            break;
        }
    }
}

std::sort(blend_batches.begin(), blend_batches.end(),
          [](auto const &a, auto const &b) { return a.camera_distance_sq > b.camera_distance_sq; });
```

Emit into `frame.draws` / `frame.transforms` / `frame.indirect_commands` / `frame.batch_bounds` in this exact order: **opaque_batches, then mask_batches, then blend_batches**, using the same per-batch emission logic the current single loop uses (unchanged — vertex address, material index, transform index, indirect command, batch bounds with wind padding).

**Known sorting limitation**: sorting is per-*batch* (by `(mesh, submesh, material)` key), not per-instance. Two instances of the same blended prop at different depths will still be emitted back-to-back regardless of camera distance, since they share a batch. Acceptable for foliage/glass instances; may show artifacts for a small number of large blended surfaces (water planes, UI quads) at overlapping depths. If that happens, revisit — options are disabling instancing for blend draws or adding a secondary per-instance sort/index buffer.

### New `RendererFrame` fields

Add alongside the existing `indirect_command_count`:

```cpp
std::uint32_t opaque_indirect_count = 0;
std::uint32_t mask_indirect_count = 0;
std::uint32_t blend_indirect_count = 0;
```

Set these after each bucket's batches are emitted. The three ranges in `frame.indirect_commands` (and therefore in `indirect_buffer` / `culled_indirect_buffer` after GPU culling, since the compute pass preserves batch order 1:1) are:

```
opaque: [0, opaque_indirect_count)
mask:   [opaque_indirect_count, opaque_indirect_count + mask_indirect_count)
blend:  [opaque_indirect_count + mask_indirect_count, total)
```

No changes needed to the frustum-cull compute dispatch (`frustum_cull.slang` / the `mainCs` dispatch in `prepare_frame`) — it operates per-workgroup on batch index and is order-preserving, so `culled_indirect_buffer` inherits the same three contiguous ranges automatically.

## 2. New pipelines

Add two new pipeline registrations in `Renderer::initialize`, both with vertex **and** fragment stages (fragment does discard-on-cutoff only, no lighting):

```cpp
PipelineHandle depth_prepass_mask_pipeline_;
PipelineHandle shadow_mask_pipeline_;
PipelineHandle forward_blend_pipeline_;
```

- `depth_prepass_mask_pipeline_`: `depth_prepass.slang` `mainVs` + new `mainFs` (discard only). Same `depth_format`, `samples`, colour formats (none) as `depth_prepass_pipeline_`.
- `shadow_mask_pipeline_`: `shadow_depth.slang` `mainVs` + new `mainFs` (discard only). Push constant range for `shadow_pc` needs `VK_SHADER_STAGE_FRAGMENT_BIT` added to `stageFlags` (material buffer address is read in the fragment stage now).
- `forward_blend_pipeline_`: same shaders as `forward_pipeline_` (`forward_geom.slang` `mainVs`/`mainFs`), but registered with blend enabled (standard alpha-over: `srcColor = SRC_ALPHA`, `dstColor = ONE_MINUS_SRC_ALPHA`) via whichever field `PipelineRegisterInfo` exposes for `VkPipelineColorBlendAttachmentState` — add one if it doesn't exist yet.

`forward_pipeline_` itself is unchanged and reused for both opaque and mask draws in the forward pass — the difference between them there is only cull mode, not shader or PSO.

## 3. Shader changes

### `scene_types.slang`
No changes needed — `AlphaMode` and `alpha_cutoff` already exist on `Material`.

### `forward_geom.slang` — `mainFs`
Add near the top, before shadow/lighting math:

```slang
const var base_sample = texture.Sample(sampler, uv);

if (material.alpha_mode == AlphaMode.mask && base_sample.a < material.alpha_cutoff) {
    discard;
}

var base_colour = base_sample.xyz * float3(material.colour[0], material.colour[1], material.colour[2]);
```

Replace the existing `base_colour` computation (currently samples `.xyz` only, no alpha) with the above. Also change the final output to carry alpha through for the blend PSO to consume (ignored by the opaque/mask PSO since blending is disabled there):

```slang
output.colour = float4(fogged_colour, base_sample.a);
```

### New: `depth_prepass.slang` `mainFs`
Needs `material_index` and UV carried from `mainVs` (add a minimal `VertexOutput` if not already present — just `SV_Position`, `nointerpolation uint material_index`, `float2 texture_coordinate`). Body:

```slang
[shader("fragment")]
void mainFs(VertexOutput input, uniform PC pc)
{
    const var material = pc.materials[input.material_index];

    if (material.alpha_mode == AlphaMode.mask) {
        const var texture = sampled_2d[NonUniformResourceIndex(material.base_colour_texture)];
        const var sampler = samplers[NonUniformResourceIndex(material.sampler_index)];
        float alpha = texture.Sample(sampler, input.texture_coordinate).a;

        if (alpha < material.alpha_cutoff) {
            discard;
        }
    }
}
```

Only this pipeline variant needs this fragment shader — `depth_prepass_pipeline_` (opaque) stays vertex-only, unchanged, to preserve early-Z.

### New: `shadow_depth.slang` `mainFs`
Same pattern as above, using `ShadowPC`'s embedded `base` (`PC`) for material lookup. Only used by `shadow_mask_pipeline_`; `shadow_pipeline_` (opaque) stays vertex-only.

## 4. `record_frame` changes

### Depth prepass region
Replace the single draw with two draws, opaque first (existing `depth_prepass_pipeline_`, `VK_CULL_MODE_BACK_BIT`, count = `opaque_indirect_count`, offset 0), then mask (`depth_prepass_mask_pipeline_`, `VK_CULL_MODE_NONE`, count = `mask_indirect_count`, offset = `opaque_indirect_count * sizeof(VkDrawIndexedIndirectCommand)`), both against `main_view_indirect_buffer`. `set_forward_dynamic_state` needs a cull-mode override or an explicit `vkCmdSetCullMode` call between the two draws — currently it hardcodes `VK_CULL_MODE_BACK_BIT` internally, so either add a parameter or call `vkCmdSetCullMode` directly after it for the mask draw.

### Shadow pass region
Same two-draw split, but repeated inside the existing per-cascade loop (currently one `vkCmdDrawIndexedIndirect` per cascade against `frame.indirect_buffer`; becomes two — opaque then mask, same cull-mode handling as above). This makes 8 draw calls total for 4 cascades instead of 4 — negligible CPU cost.

**Both prepass and shadow draws should use `frame.opaque_indirect_count` and `frame.mask_indirect_count` only — never include the blend range.** Blend draws should never appear in the shadow pass or depth prepass at all under this plan (no shadow casting from blend geometry, no blend contribution to prepass depth).

### Forward pass region
Three draws instead of one, all within the existing single `vkCmdBeginRendering`/`vkCmdEndRendering` block:

1. **Opaque**: `forward_pipeline_`, `VK_CULL_MODE_BACK_BIT`, `VK_COMPARE_OP_EQUAL` depth test, depth write enabled — as today. Count = `opaque_indirect_count`, offset 0.
2. **Mask**: same `forward_pipeline_` (no rebind), just `vkCmdSetCullMode(VK_CULL_MODE_NONE)` before the draw, same `EQUAL` depth compare (valid now that the mask prepass wrote matching depth with matching cull mode). Count = `mask_indirect_count`, offset = `opaque_indirect_count * sizeof(VkDrawIndexedIndirectCommand)`.
3. **Blend**: `forward_blend_pipeline_`, `VK_CULL_MODE_NONE`, `VK_COMPARE_OP_GREATER_OR_EQUAL` depth test (reverse-Z; blend has no prepass contribution so must test against existing depth without expecting an exact match), depth write **disabled**. Count = `blend_indirect_count`, offset = `(opaque_indirect_count + mask_indirect_count) * sizeof(VkDrawIndexedIndirectCommand)`.

`set_forward_dynamic_state` needs a new parameter (e.g. `is_blend_pass`) to toggle depth write and compare op for case 3, alongside the existing `is_prepass` parameter.

## 5. Explicitly out of scope for this pass

- Per-instance sorting for blend (only per-batch sorting specified above).
- Alpha-to-coverage as an alternative to `discard` for mask under MSAA — worth considering later given `samples_` is already a first-class renderer setting, but not included here.
- Two-sided-only-for-mask enforcement at material-authoring time (this spec always forces `VK_CULL_MODE_NONE` for mask draws at the pipeline level, which is a reasonable default but not configurable per-material).
