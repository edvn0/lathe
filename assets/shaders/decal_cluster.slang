import scene_types;
import light_clusters;

static const uint group_size = 128;
static const uint maximum_tile_decals = 1024;

struct DecalClusterPushConstants
{
    Ptr<UBO> ubo;
    Ptr<GpuDecal> decals;
    Ptr<uint> decal_clusters;
    uint decal_count;
    uint _padding;
};

[vk::push_constant] DecalClusterPushConstants pc;

groupshared float4 s_tile_spheres[maximum_tile_decals];
groupshared uint s_tile_decal_indices[maximum_tile_decals];
groupshared uint s_tile_decal_count;

groupshared uint s_wave_totals[group_size];
groupshared uint s_wave_offsets[group_size];

groupshared float3 s_slice_min[cluster_grid_maximum_depth_slices];
groupshared float3 s_slice_max[cluster_grid_maximum_depth_slices];
groupshared uint s_slice_counts[cluster_grid_maximum_depth_slices];

float3 tile_side_plane(float3 ray_a, float3 ray_b, float3 centre_ray)
{
    const float3 normal = normalize(cross(ray_a, ray_b));
    return dot(normal, centre_ray) < 0.0 ? -normal : normal;
}

// Clusters share the light grid. Layout: one count per cluster, then decal_cluster_capacity indices per cluster.
// Every cluster's count is written, so the buffer needs no clearing.
[shader("compute")]
[numthreads(group_size, 1, 1)]
void main_cs(uint3 group_id: SV_GroupID, uint lane: SV_GroupIndex)
{
    const var ubo = pc.ubo;

    const uint tile_x = group_id.x % ubo.cluster_grid_x;
    const uint tile_y = group_id.x / ubo.cluster_grid_x;
    const uint slice_count = ubo.cluster_grid_z;

    const uint lane_in_wave = WaveGetLaneIndex();
    const uint wave_id = lane / WaveGetLaneCount();
    const uint wave_count = (group_size + WaveGetLaneCount() - 1) / WaveGetLaneCount();

    float2 ndc_min;
    float2 ndc_max;
    cluster_tile_ndc(ubo, tile_x, tile_y, ndc_min, ndc_max);

    const float4x4 inverse_projection = ubo.inverse_projection;
    float3 rays[4] = {
        cluster_view_ray(inverse_projection, ndc_min),
        cluster_view_ray(inverse_projection, float2(ndc_max.x, ndc_min.y)),
        cluster_view_ray(inverse_projection, float2(ndc_min.x, ndc_max.y)),
        cluster_view_ray(inverse_projection, ndc_max),
    };

    const float3 centre_ray = rays[0] + rays[1] + rays[2] + rays[3];
    const float3 side_planes[4] = {
        tile_side_plane(rays[0], rays[2], centre_ray),
        tile_side_plane(rays[1], rays[3], centre_ray),
        tile_side_plane(rays[0], rays[1], centre_ray),
        tile_side_plane(rays[2], rays[3], centre_ray),
    };

    if (lane < slice_count) {
        float3 aabb_min;
        float3 aabb_max;
        cluster_aabb(ubo, rays, lane, aabb_min, aabb_max);
        s_slice_min[lane] = aabb_min;
        s_slice_max[lane] = aabb_max;
        s_slice_counts[lane] = 0;
    }

    if (lane == 0) {
        s_tile_decal_count = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    const float4x4 view = ubo.view;

    for (uint base = 0; base < pc.decal_count; base += group_size) {
        const uint decal_index = base + lane;

        bool in_tile = false;
        float4 sphere = float4(0.0);

        if (decal_index < pc.decal_count) {
            const float4 world_sphere = pc.decals[decal_index].bounding_sphere;
            sphere = float4(mul(view, float4(world_sphere.xyz, 1.0)).xyz, world_sphere.w);
            in_tile = true;

            [unroll]
            for (uint plane = 0; plane < 4; ++plane) {
                in_tile = in_tile && dot(side_planes[plane], sphere.xyz) >= -sphere.w;
            }
        }

        const uint wave_exclusive = WavePrefixCountBits(in_tile);
        const uint wave_total = WaveActiveCountBits(in_tile);

        if (lane_in_wave == 0) {
            s_wave_totals[wave_id] = wave_total;
        }

        GroupMemoryBarrierWithGroupSync();

        if (lane == 0) {
            uint running = s_tile_decal_count;

            for (uint wave = 0; wave < wave_count; ++wave) {
                s_wave_offsets[wave] = running;
                running += s_wave_totals[wave];
            }

            s_tile_decal_count = running;
        }

        GroupMemoryBarrierWithGroupSync();

        if (in_tile) {
            const uint slot = s_wave_offsets[wave_id] + wave_exclusive;
            s_tile_spheres[slot] = sphere;
            s_tile_decal_indices[slot] = decal_index;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    const uint batch_count = s_tile_decal_count;
    const uint lane_count = WaveGetLaneCount();
    const uint list_start = cluster_count(ubo);

    for (uint slice = wave_id; slice < slice_count; slice += wave_count) {
        const uint cluster = cluster_flat_index(ubo, tile_x, tile_y, slice);
        const uint list_base = list_start + cluster * decal_cluster_capacity;
        const float3 aabb_min = s_slice_min[slice];
        const float3 aabb_max = s_slice_max[slice];

        uint running = 0;

        for (uint wave_base = 0; wave_base < batch_count; wave_base += lane_count) {
            const uint batch_index = wave_base + lane_in_wave;
            const bool hit = batch_index < batch_count &&
                             sphere_intersects_aabb(s_tile_spheres[batch_index], aabb_min, aabb_max);

            const uint slot = running + WavePrefixCountBits(hit);

            if (hit && slot < decal_cluster_capacity) {
                pc.decal_clusters[list_base + slot] = s_tile_decal_indices[batch_index];
            }

            running += WaveActiveCountBits(hit);
        }

        if (lane_in_wave == 0) {
            pc.decal_clusters[cluster] = min(running, decal_cluster_capacity);
        }
    }
}
