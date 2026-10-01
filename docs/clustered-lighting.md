# Clustered lighting

Punctual lights (point and spot) are culled and binned into view-space clusters on the GPU every frame. The forward
fragment shader shades only the lights in its own cluster, so the frame cost follows how many lights touch each pixel,
not how many exist. The renderer accepts up to `maximum_light_count` = 65,536 of them.

## Grid

Screen tiles by depth slices, 16 x 9 x 24 by default. The grid and the
per-cluster capacity are `ClusterGridSettings` (`rendering/cluster_grid.hxx`),
set at runtime with `Renderer::set_cluster_grid` and passed to the shaders in
`UBO::cluster_grid_*` and `UBO::cluster_light_capacity`. The grid doesn't
follow the window size, so resizing never touches the cluster buffers. See
[Tuning the grid](#tuning-the-grid).

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
2. **`light_cluster.slang`**: one workgroup per screen tile (144 for the
   default grid).
   - The group streams the visible list and keeps the lights that touch the
     tile's four side planes, in shared-memory batches of 1024.
   - Each batch is then assigned to the tile's depth slices. Every wave
     owns some slices and walks the batch one wave at a time. It tests each
     sphere against the cluster's view-space AABB and appends hits with a
     ballot prefix count.

   The lists use no atomics, so each comes out sorted by light index and the
   output is deterministic. Each group then adds its clusters to four
   statistics with one atomic per wave (see [Debugging](#debugging)).

`cluster_lights_buffer` holds the 16 bytes of statistics, then one count per
cluster, then `light_capacity` indices per cluster (`cluster_buffer_bytes`):
3.4 MiB per frame in flight for the default grid. When a cluster has more
lights than its capacity, it keeps the lowest light indices, and its count
still reports the true total.

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
- **Lights per cluster:** `light_capacity`, 256 by default and at most 1,024. A
  cluster that touches more keeps the lowest light indices and drops the rest,
  so those lights go missing from that part of the screen. The heatmap shows
  such clusters in magenta, and the statistics count them.

  Normal scenes stay far below 256. Overflow takes very dense light fields
  seen from a distance, because a distant cluster spans a large area. In a
  stress test of 35,000 point lights spread evenly over 240 x 240 m (about
  0.6 per square metre, each with a range of 2-6 m), clusters covering up to
  11% of the screen overflowed, and 0-5% in most views.

  If a scene needs more, raise the capacity or refine the grid (see below).
- **Grid size:** at most 64 x 64 tiles and 64 depth slices, with the lists at
  most 128 MiB per frame in flight. `validate_cluster_grid` refuses anything
  past those.

## Tuning the grid

Lighting > Debug has a **Cluster grid** preset menu and a slider for each
dimension. Changes apply from the next frame, and each frame in flight
reallocates its lists the next time it is prepared, so the heatmap follows
the sliders.

| Preset | Grid | Capacity | Clusters | Lists per frame |
| --- | --- | --- | --- | --- |
| Coarse | 8 x 5 x 16 | 256 | 640 | 0.6 MiB |
| Default | 16 x 9 x 24 | 256 | 3,456 | 3.4 MiB |
| Fine | 32 x 18 x 32 | 128 | 18,432 | 9.1 MiB |
| Very fine | 48 x 27 x 48 | 128 | 62,208 | 30.6 MiB |

The trade-off:

- **Finer** grids give each fragment fewer lights that miss it, so the forward
  pass shades less. Clusters overflow less, so a smaller capacity is enough.
  The build does more work: one workgroup per tile, each walking every
  visible light, so the Light Clustering stage grows with the tile count.
- **Coarser** grids are cheaper to build, but every pixel in a cluster shades
  all its lights. That suits scenes with few lights, or lights that are large
  next to the cluster size.
- **Depth slices** matter when lights stack up along the view direction. Lights
  spread over a wide area seen at a grazing angle (a street, a field) need
  more slices.
- **Capacity** costs memory, not shading time. Only clusters that actually
  hold that many lights pay for them.

The fragment shader shades the union of the lists in its wave, so a finer
grid saves less in practice than the per-cluster averages suggest.

Under the sliders, the panel shows the latest frame's statistics. They are
read back a frames-in-flight cycle late:

- **Occupied clusters:** clusters touching at least one light.
- **Lights per occupied cluster:** the average and the most. The most counts
  lights the cluster dropped.
- **Overflowing clusters:** clusters past the capacity, which drop lights.
  They are shown in magenta when there are any.

To compare grids, benchmark one build twice with `--cluster-grid=XxYxZ` or
`--cluster-grid=XxYxZ:capacity` (`docs/perf-benchmark.md`). Look at the
Light Clustering and Forward Pass stages. The flag also sets the grid for a
normal run.

## Debugging

Lighting > Debug has two toggles:

- **Clustered lighting (GPU)** (`Renderer::set_clustered_lighting`). Turning it
  off falls back to looping over every light per fragment, for comparison.
  With thousands of lights that loop is extremely slow.
- **Cluster light-count heatmap** (`set_cluster_debug_heatmap`). Tints each
  fragment by its cluster's light count:
  - blue through red for 1 up to 64 or more lights,
  - black for none,
  - magenta past the capacity, where lights are being dropped.

`Renderer::last_cluster_stats` returns the statistics behind the panel's
numbers.

The two dispatches share one GPU timestamp, so they show in the frame-timing
plot and as `light_clustering` in benchmark output.
