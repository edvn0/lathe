# Ray traced sun shadows

Where the device has `VK_KHR_ray_query`, the sun's shadow is traced against the scene instead of read from the cascaded shadow map.
It is on by default and there is nothing to start: a project turns it off with

```toml
[render]
ray_traced_shadows = false
```

(Edit > Project Settings > Rendering, applied the next time the project opens), and the frame graph panel's "Ray traced
shadows" stage switches it while the editor runs and sets the sun's angular radius.

## How it works

`rt_shadows` is a compute pass between the depth prepass and the forward pass (`src/rendering/ray_traced_shadows.cxx`,
`assets/shaders/rt_shadows.slang`, declared in `renderer_frame_graph.cxx`). For every pixel it rebuilds the surface point from
the depth buffer, takes the face normal from the neighbours' positions (the side whose depth is closer, so a silhouette does
not tilt it), and traces rays toward the sun against the scene TLAS from a point offset along the normal (the offset grows
with distance: the position is only as exact as the depth buffer, and the TLAS has half precision vertices). The result is
an R8 mask, 1 lit and 0 shadowed. The forward pass multiplies the sun by it where it used to look the point up in the
cascades (`ray_traced_sun_shadow` in `forward_geom.slang`), averaging the pixel and its four neighbours.

- **Soft edges.** Rays go through the sun's disc (a cone of `sun_spread` radians, 0.006 by default, the real sun is about
  0.0047), stratified and turned per pixel by interleaved gradient noise. Two rays go first (the middle and a random point);
  only pixels where they disagree, the penumbra, trace two more, so most pixels cost two rays. 0.8 ms at 1280x720 in the
  town stress scene.
- **Alpha tested geometry.** Instances from the mask batches are `flag_alpha_tested`: not opaque in the TLAS, so the shadow ray
  sees their triangles as candidates, evaluates the material's alpha test (base colour alpha times the texture, against the
  cutoff) and commits only the ones that pass. Leaves cast the shadow of their leaves, not of their cards. The probe GI traces
  with `RAY_FLAG_FORCE_OPAQUE` and still sees them as solid.
- **What is not traced.** The blended pass keeps the cascades (its surfaces are not in the depth buffer the mask comes from),
  as does the editor's second viewport (the TLAS is built for the main view only). Skinned meshes are in the TLAS only near the
  camera and within the frame's vertex budget (`docs/probe-gi.md`); beyond that they neither cast nor receive.
- **Cascades are skipped** on frames where the mask covers every surface (no blended draws, primary view, TLAS ready): the
  shadow pass is not recorded and the cascade cache is dropped, so the first frame that needs them redraws all four
  (`cascades_unused` in `Renderer::prepare_frame`'s shadow section).

## Known limits

## Local lights

The same pass also traces the lights that have a shadow tile (the shadow atlas still decides which lights cast). For each
pixel it walks the cluster's light list (the whole list without clustered lighting) and traces one ray to each shadow
casting light, writing the first three into the mask's G, B and A. The forward pass counts the shadow casting lights of
its own list the same way and reads that slot; a fourth and later light, the blended pass and the second viewport keep
the atlas. One ray per light per pixel gives hard edges. The frame graph panel's "Local lights" checkbox switches it
(`Settings::local_lights`). The pass now runs when the sun is off but a shadowed light exists.

## Known limits

The sun and up to three local lights per pixel are traced. The normal comes from the depth buffer, so very thin or
very distant geometry can show a little acne or leaking near its silhouette; the normal offset trades one for the other.
