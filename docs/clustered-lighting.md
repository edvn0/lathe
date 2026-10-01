# Clustered lighting

Punctual lights (point and spot) are culled and binned into view-space clusters on the GPU every frame. The forward
fragment shader shades only the lights in its own cluster, so the frame cost follows how many lights touch each pixel,
not how many exist. The renderer accepts up to `maximum_light_count` = 65,536 of them.

## Grid

16 x 9 screen tiles by 24 depth slices (`cluster_grid_*`, mirrored in
`scene_types.slang` and `renderer.hxx`). The cluster count is fixed, so
resizing never touches the cluster buffers.

- **Tiles** are cut in NDC, not pixels. The fragment shader projects its
  world position with `view_projection` to find its tile, and the build
  passes unproject the same NDC rectangle through `inverse_projection`.
  Neither side has to know about the forward pass's flipped-Y, reverse-Z
  viewport.
- **Slices** split [near, far] exponentially:
  `slice = floor(log(view_z) * cluster_z_scale + cluster_z_bias)`, with both
  terms in the UBO. Slice 0 reaches back to the eye and the last slice to the
  far plane, so fragments the shader clamps into them are still covered.

## Bounds

A point light is bounded by the sphere of its range. A spot light gets the
tightest sphere around its spherical sector: through the apex and the rim for
cones up to 45 degrees, around the cap's circle for wider ones
(`light_bounding_sphere` in `light_clusters.slang`).

## Build ("Light Clustering" stage)

Two compute dispatches, then a barrier to the fragment shader. Both run
whenever clustering is on, even with no lights, because the forward pass reads
every cluster's count.

1. **`light_cull.slang`**: one 256-wide workgroup walks every light and tests
   its sphere against the camera frustum. It compacts the visible ones into
   `visible_lights_buffer` as view-space spheres plus light indices. A
   shared-memory prefix sum keeps them in light-index order. At 65k lights this
   is 256 iterations.
2. **`light_cluster.slang`**: one workgroup per screen tile (144 in all).
   - The group streams the visible list and keeps the lights that touch the
     tile's four side planes, in shared-memory batches of 1024.
   - Each batch is then assigned to the tile's 24 depth slices. Every wave
     owns some slices and walks the batch one wave at a time. It tests each
     sphere against the cluster's view-space AABB and appends hits with a
     ballot prefix count.

   No atomics are involved, so each cluster's list comes out sorted by light
   index and the output is deterministic.

`cluster_lights_buffer` holds `cluster_count` counts, followed by
`cluster_count * cluster_light_capacity` (256) indices: 3.4 MiB per frame in
flight. When a cluster has more than 256 lights, it keeps the lowest light
indices, and its count still reports the true total.

## Shading (`forward_geom.slang`)

Each lane reads its cluster's sorted list. The wave then merges the lists:
every iteration takes `WaveActiveMin` of each lane's next index, shades that
one light on every lane, and advances the lanes that held it.

- The loop runs once per light in the union of the wave's clusters.
- It stays uniform, so the light fetch is a scalar load and an NVIDIA warp
  never diverges inside the light loop.
- A lane whose cluster lacks the current light is outside that light's
  range. Falloff is exactly zero beyond the range and outside the cone, so
  shading it adds nothing.

The wave ops need subgroup arithmetic in the fragment stage. Every GPU with the
`VK_EXT_mesh_shader` support the renderer already requires provides it.

## Limits

- **Total lights:** 65,536 point and spot lights together (`maximum_light_count`).
  Past that, `submit_point_light` and `submit_spot_light` return
  `capacity_exceeded`. The directional light is separate, and it is the only
  light that casts shadows.
- **Lights per cluster:** 256 (`cluster_light_capacity`). A cluster that
  touches more keeps the lowest light indices and silently drops the rest, so
  those lights go missing from that part of the screen. The heatmap shows such
  clusters in magenta.

  Normal scenes stay far below 256. Overflow takes very dense light fields
  seen from a distance, because a distant cluster spans a large area. In a
  stress test of 35,000 point lights spread evenly over 240 x 240 m (about
  0.6 per square metre, each with a range of 2-6 m), clusters covering up to
  11% of the screen overflowed, and 0-5% in most views.

  If a scene needs more, raise `cluster_light_capacity` in both
  `renderer.hxx` and `scene_types.slang`. Each step of 256 costs another
  3.4 MiB per frame in flight, and distant pixels pay to shade the extra
  lights. A finer `cluster_grid_*` also helps, because smaller clusters hold
  fewer lights.

## Debugging

Lighting > Debug has two toggles:

- **Clustered lighting (GPU)** (`Renderer::set_clustered_lighting`). Turning it
  off falls back to looping over every light per fragment, for comparison.
  With thousands of lights that loop is extremely slow.
- **Cluster light-count heatmap** (`set_cluster_debug_heatmap`). Tints each
  fragment by its cluster's light count:
  - blue through red for 1 up to 64 or more lights,
  - black for none,
  - magenta past `cluster_light_capacity`, where lights are being dropped.

The two dispatches share one GPU timestamp, so they show in the frame-timing
plot and as `light_clustering` in benchmark output.
