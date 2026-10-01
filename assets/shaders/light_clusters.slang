module light_clusters;

import scene_types;

// Clustered lighting shared by light_cull.slang, light_cluster.slang and the forward fragment shader. Tiles are laid
// out in NDC rather than pixels, so the build and the lookup agree without knowing the viewport flip.

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

// World-space bounding sphere (xyz = centre, w = radius) of everything a light can reach. A spot light's lit region
// is a spherical sector (range, outer cone angle), bounded by the smallest sphere: wide cones by the cap's circle,
// narrow ones by the sphere through the apex and the rim.
public float4 light_bounding_sphere(GpuLight light)
{
    if (light.type != LightType.spot) {
        return float4(light.position, light.range);
    }

    const float cos_outer = -light.spot_offset / light.spot_scale;

    if (cos_outer <= 0.0) {
        return float4(light.position, light.range);
    }

    const float3 direction = normalize(light.direction);

    if (cos_outer >= 0.70710678) {
        const float radius = light.range / (2.0 * cos_outer);
        return float4(light.position + direction * radius, radius);
    }

    const float sin_outer = sqrt(saturate(1.0 - cos_outer * cos_outer));
    return float4(light.position + direction * (light.range * cos_outer), light.range * sin_outer);
}

// View-space direction through an NDC point, scaled to unit view depth.
public float3 cluster_view_ray(float4x4 inverse_projection, float2 ndc)
{
    const float4 near_point = mul(inverse_projection, float4(ndc, 0.0, 1.0));
    const float3 point = near_point.xyz / near_point.w;
    return point / point.z;
}

// A tile's NDC rectangle.
public void cluster_tile_ndc(uint tile_x, uint tile_y, out float2 ndc_min, out float2 ndc_max)
{
    const float2 grid = float2(cluster_grid_x, cluster_grid_y);
    ndc_min = float2(tile_x, tile_y) / grid * 2.0 - 1.0;
    ndc_max = float2(tile_x + 1, tile_y + 1) / grid * 2.0 - 1.0;
}

// View-space AABB of one cluster. The first slice reaches back to the eye and the last to the far plane, so fragments
// cluster_index() clamps into them stay covered.
public void cluster_aabb(Ptr<UBO> ubo, float3 rays[4], uint slice, out float3 aabb_min, out float3 aabb_max)
{
    const float near_depth = slice == 0 ? 0.0 : cluster_slice_depth(ubo, float(slice));
    const float far_depth = cluster_slice_depth(ubo, float(slice + 1));

    aabb_min = rays[0] * near_depth;
    aabb_max = aabb_min;

    [unroll]
    for (uint corner = 0; corner < 4; ++corner) {
        const float3 near_corner = rays[corner] * near_depth;
        const float3 far_corner = rays[corner] * far_depth;
        aabb_min = min(aabb_min, min(near_corner, far_corner));
        aabb_max = max(aabb_max, max(near_corner, far_corner));
    }
}

public bool sphere_intersects_aabb(float4 sphere, float3 aabb_min, float3 aabb_max)
{
    const float3 offset = clamp(sphere.xyz, aabb_min, aabb_max) - sphere.xyz;
    return dot(offset, offset) <= sphere.w * sphere.w;
}
