# Clustered lighting

Punctual lights (point and spot, up to `maximum_light_count` = 256) are
binned into view-space clusters by a compute pass every frame. The forward
fragment shader then shades only the lights in its own cluster, where it used
to loop over every light in the scene.

## Grid

16 x 9 screen tiles by 24 depth slices (`cluster_grid_*`, mirrored in
`scene_types.slang` and `renderer.hxx`). The cluster count is fixed, so
resizing never touches the cluster buffers.

- **Tiles** are cut in NDC, not pixels. The fragment shader projects its
  world position with `view_projection` to find its tile, and the compute
  pass unprojects the same NDC rectangle through `inverse_projection`. Neither
  side has to know about the forward pass's flipped-Y, reverse-Z viewport.
- **Slices** split [near, far] exponentially:
  `slice = floor(log(view_z) * cluster_z_scale + cluster_z_bias)`, with both
  terms in the UBO. Slice 0 reaches back to the eye and the last slice to the
  far plane, so fragments the shader clamps into them are still covered.

## Build (`light_cluster.slang`, "Light Clustering" stage)

One thread per cluster in 128-wide workgroups. Each workgroup first writes
every light's view-space bounding sphere to shared memory. A point light's
sphere is its range. A spot light gets the tightest sphere around its
spherical sector: through the apex and the rim for cones up to 45 degrees,
around the cap's circle for wider ones. Each thread then builds its
cluster's view-space AABB from the four corner rays at the slice's near and
far depths, and tests it against every sphere.

The result is a **bitmask per cluster**, one bit per light. That is
`cluster_mask_words` = 8 uints, so 3456 clusters take 108 KiB per frame in
flight. Masks need no atomics or global counters, and the output is
deterministic. The word loop is unrolled so each word stays in a register.

The pass is skipped when clustering is off or there are no lights. A buffer
barrier hands the masks from compute to fragment.

## Shading (`forward_geom.slang`)

The fragment shader reads its cluster's mask words and ORs each one across
the wave (`WaveActiveBitOr`) before walking the set bits. Every lane then
runs the same light loop. On NVIDIA that keeps the warp convergent and turns
the light fetches into uniform loads, rather than each lane chasing its own
list. Lanes that pick up a neighbour's light lose nothing in correctness:
the falloff is exactly zero beyond a light's range and outside its cone, and
cluster membership is conservative.

The wave ops need subgroup arithmetic in the fragment stage. Every GPU that
has the `VK_EXT_mesh_shader` support the renderer already requires provides
it.

## Debugging

Lighting > Debug has two toggles:

- **Clustered lighting (GPU)** (`Renderer::set_clustered_lighting`). Turn
  it off to fall back to the old all-lights loop and compare.
- **Cluster light-count heatmap** (`set_cluster_debug_heatmap`). Tints each
  fragment by how many lights its cluster holds: blue for few, red at 16 or
  more, and black for none.

The pass has its own GPU timestamp, so it shows in the frame-timing plot and
as `light_clustering` in benchmark output.
