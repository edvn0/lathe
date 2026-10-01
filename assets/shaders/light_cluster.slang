import scene_types;
import light_clusters;

// Builds the per-cluster light bitmasks the forward pass shades from. One thread per cluster: the view-space
// bounding sphere of every light goes into shared memory once per workgroup, then each thread tests its cluster's
// view-space AABB against all of them and writes cluster_mask_words words.
//
// Masks rather than index lists keep this free of atomics and the output deterministic, and with at most
// maximum_light_count lights a cluster's mask is only cluster_mask_words uints.

static const uint group_size = 128;

struct LightClusterPushConstants
{
    Ptr<UBO> ubo;
    Ptr<GpuLight> lights;

    // cluster_count * cluster_mask_words, cluster-major.
    Ptr<uint> cluster_light_masks;

    uint light_count;
    uint _padding;
};

[vk::push_constant] LightClusterPushConstants pc;

// xyz = view-space centre, w = radius.
groupshared float4 s_light_spheres[maximum_light_count];

// A spot light's lit region is a spherical sector (range, outer cone angle); bound it with the smallest sphere.
// Wide cones are bounded by the cap's circle, narrow ones by the sphere through the apex and the rim.
float4 light_bounding_sphere(GpuLight light, float4x4 view)
{
    const float3 position = mul(view, float4(light.position, 1.0)).xyz;

    if (light.type != LightType.spot) {
        return float4(position, light.range);
    }

    const float cos_outer = -light.spot_offset / light.spot_scale;

    if (cos_outer <= 0.0) {
        return float4(position, light.range);
    }

    const float3 direction = normalize(mul((float3x3)view, light.direction));

    if (cos_outer >= 0.70710678) {
        const float radius = light.range / (2.0 * cos_outer);
        return float4(position + direction * radius, radius);
    }

    const float sin_outer = sqrt(saturate(1.0 - cos_outer * cos_outer));
    return float4(position + direction * (light.range * cos_outer), light.range * sin_outer);
}

// View-space direction through an NDC point, scaled to unit view depth.
float3 view_ray(float4x4 inverse_projection, float2 ndc)
{
    const float4 near_point = mul(inverse_projection, float4(ndc, 0.0, 1.0));
    const float3 point = near_point.xyz / near_point.w;
    return point / point.z;
}

bool sphere_intersects_aabb(float4 sphere, float3 aabb_min, float3 aabb_max)
{
    const float3 offset = clamp(sphere.xyz, aabb_min, aabb_max) - sphere.xyz;
    return dot(offset, offset) <= sphere.w * sphere.w;
}

[shader("compute")]
[numthreads(group_size, 1, 1)]
void main_cs(uint3 dispatch_id: SV_DispatchThreadID, uint lane: SV_GroupIndex)
{
    const var ubo = pc.ubo;
    const uint light_count = min(pc.light_count, maximum_light_count);

    for (uint light_index = lane; light_index < light_count; light_index += group_size) {
        s_light_spheres[light_index] = light_bounding_sphere(pc.lights[light_index], ubo.view);
    }

    GroupMemoryBarrierWithGroupSync();

    const uint cluster = dispatch_id.x;

    if (cluster >= cluster_count) {
        return;
    }

    const uint tile_x = cluster % cluster_grid_x;
    const uint tile_y = (cluster / cluster_grid_x) % cluster_grid_y;
    const uint slice = cluster / (cluster_grid_x * cluster_grid_y);

    // The first slice reaches back to the eye and the last to the far plane, so fragments the forward pass clamps
    // into them stay covered.
    const float near_depth = slice == 0 ? 0.0 : cluster_slice_depth(ubo, float(slice));
    const float far_depth = cluster_slice_depth(ubo, float(slice + 1));

    const float2 grid = float2(cluster_grid_x, cluster_grid_y);
    const float2 ndc_min = float2(tile_x, tile_y) / grid * 2.0 - 1.0;
    const float2 ndc_max = float2(tile_x + 1, tile_y + 1) / grid * 2.0 - 1.0;

    const float4x4 inverse_projection = ubo.inverse_projection;
    const float3 rays[4] = {
        view_ray(inverse_projection, ndc_min),
        view_ray(inverse_projection, float2(ndc_max.x, ndc_min.y)),
        view_ray(inverse_projection, float2(ndc_min.x, ndc_max.y)),
        view_ray(inverse_projection, ndc_max),
    };

    float3 aabb_min = rays[0] * near_depth;
    float3 aabb_max = aabb_min;

    [unroll]
    for (uint corner = 0; corner < 4; ++corner) {
        const float3 near_corner = rays[corner] * near_depth;
        const float3 far_corner = rays[corner] * far_depth;
        aabb_min = min(aabb_min, min(near_corner, far_corner));
        aabb_max = max(aabb_max, max(near_corner, far_corner));
    }

    const uint mask_base = cluster * cluster_mask_words;

    // Unrolled so each word stays in a register rather than a dynamically indexed array.
    [unroll]
    for (uint word_index = 0; word_index < cluster_mask_words; ++word_index) {
        const uint first_light = word_index * 32;
        uint word = 0;

        if (first_light < light_count) {
            const uint word_light_count = min(light_count - first_light, 32u);

            for (uint bit = 0; bit < word_light_count; ++bit) {
                if (sphere_intersects_aabb(s_light_spheres[first_light + bit], aabb_min, aabb_max)) {
                    word |= 1u << bit;
                }
            }
        }

        pc.cluster_light_masks[mask_base + word_index] = word;
    }
}
