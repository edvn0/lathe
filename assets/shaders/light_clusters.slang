module light_clusters;

import scene_types;

// Clustered lighting shared by light_cluster.slang (which builds the masks) and the forward fragment shader (which
// reads them). Tiles are laid out in NDC rather than pixels, so both sides agree without knowing the viewport flip.

public float cluster_slice_depth(Ptr<UBO> ubo, float slice)
{
    return exp((slice - ubo.cluster_z_bias) / ubo.cluster_z_scale);
}

// `clip` is the fragment's clip-space position, `view_z` its view-space depth.
public uint cluster_index(Ptr<UBO> ubo, float4 clip, float view_z)
{
    const float2 ndc = clip.xy / clip.w;
    const float2 tile_uv = saturate(ndc * 0.5 + 0.5);

    const uint tile_x = min(uint(tile_uv.x * float(cluster_grid_x)), cluster_grid_x - 1);
    const uint tile_y = min(uint(tile_uv.y * float(cluster_grid_y)), cluster_grid_y - 1);

    const float slice = log(max(view_z, 1e-4)) * ubo.cluster_z_scale + ubo.cluster_z_bias;
    const uint slice_index = uint(clamp(slice, 0.0, float(cluster_grid_z - 1)));

    return (slice_index * cluster_grid_y + tile_y) * cluster_grid_x + tile_x;
}
