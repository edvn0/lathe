import scene_types;
import light_clusters;

// Second half of the light clustering: builds the per-cluster light lists the forward pass shades from.
//
// One workgroup per screen tile (cluster_tile_count(ubo) of them). The group streams light_cull.slang's visible lights,
// keeps the ones touching the tile's side planes in shared memory, and every tile_light_batch of those assigns them
// to the tile's depth slices: each wave owns some slices and walks the batch a wave at a time, appending hits with a
// ballot prefix count.
//
// Lists stay in light-index order, with no atomics, so the output is deterministic and the forward pass can merge
// lists across a wave. A cluster past UBO::cluster_light_capacity keeps its lowest indices; its count still reports
// the total.
//
// Each group also adds its clusters to cluster_stats for the editor. Those are the only atomics, so the lists stay
// deterministic.

static const uint group_size = 128;
static const uint tile_light_batch = 1024;

struct LightClusterPushConstants
{
    Ptr<UBO> ubo;

    // light_cull.slang's output.
    Ptr<float4> visible_spheres;
    Ptr<uint> visible_light_indices;
    Ptr<uint> visible_light_count;

    // cluster_count(ubo) counts, then cluster_count(ubo) * UBO::cluster_light_capacity light indices.
    Ptr<uint> cluster_lights;

    // Zeroed before the dispatch: occupied clusters, overflowing clusters, the most lights in one cluster, and the
    // light indices stored. Mirrors ClusterStats.
    Ptr<uint> cluster_stats;
};

[vk::push_constant] LightClusterPushConstants pc;

groupshared float4 s_tile_spheres[tile_light_batch];
groupshared uint s_tile_light_indices[tile_light_batch];
groupshared uint s_tile_light_count;

groupshared uint s_wave_totals[group_size];
groupshared uint s_wave_offsets[group_size];

groupshared float3 s_slice_min[cluster_grid_maximum_depth_slices];
groupshared float3 s_slice_max[cluster_grid_maximum_depth_slices];
groupshared uint s_slice_counts[cluster_grid_maximum_depth_slices];

// The plane through the eye and two corner rays, facing the tile's centre ray.
float3 tile_side_plane(float3 ray_a, float3 ray_b, float3 centre_ray)
{
    const float3 normal = normalize(cross(ray_a, ray_b));
    return dot(normal, centre_ray) < 0.0 ? -normal : normal;
}

// Appends the batched tile lights to the slices this wave owns.
void assign_batch_to_slices(Ptr<UBO> ubo, uint tile_x, uint tile_y, uint wave_id, uint wave_count, uint lane_in_wave)
{
    const uint batch_count = s_tile_light_count;
    const uint lane_count = WaveGetLaneCount();
    const uint capacity = ubo.cluster_light_capacity;
    const uint list_start = cluster_count(ubo);

    for (uint slice = wave_id; slice < ubo.cluster_grid_z; slice += wave_count) {
        const uint list_base = list_start + cluster_flat_index(ubo, tile_x, tile_y, slice) * capacity;
        const float3 aabb_min = s_slice_min[slice];
        const float3 aabb_max = s_slice_max[slice];

        uint running = s_slice_counts[slice];

        for (uint wave_base = 0; wave_base < batch_count; wave_base += lane_count) {
            const uint batch_index = wave_base + lane_in_wave;
            const bool hit = batch_index < batch_count &&
                             sphere_intersects_aabb(s_tile_spheres[batch_index], aabb_min, aabb_max);

            const uint slot = running + WavePrefixCountBits(hit);

            if (hit && slot < capacity) {
                pc.cluster_lights[list_base + slot] = s_tile_light_indices[batch_index];
            }

            running += WaveActiveCountBits(hit);
        }

        if (lane_in_wave == 0) {
            s_slice_counts[slice] = running;
        }
    }
}

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
        tile_side_plane(rays[0], rays[2], centre_ray), // ndc_min.x
        tile_side_plane(rays[1], rays[3], centre_ray), // ndc_max.x
        tile_side_plane(rays[0], rays[1], centre_ray), // ndc_min.y
        tile_side_plane(rays[2], rays[3], centre_ray), // ndc_max.y
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
        s_tile_light_count = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    const uint visible_count = pc.visible_light_count[0];

    for (uint base = 0; base < visible_count; base += group_size) {
        const uint visible_index = base + lane;

        bool in_tile = false;
        float4 sphere = float4(0.0);

        if (visible_index < visible_count) {
            sphere = pc.visible_spheres[visible_index];
            in_tile = true;

            [unroll]
            for (uint plane = 0; plane < 4; ++plane) {
                in_tile = in_tile && dot(side_planes[plane], sphere.xyz) >= -sphere.w;
            }
        }

        // Wave ops stay outside the branch so every lane contributes.
        const uint wave_exclusive = WavePrefixCountBits(in_tile);
        const uint wave_total = WaveActiveCountBits(in_tile);

        if (lane_in_wave == 0) {
            s_wave_totals[wave_id] = wave_total;
        }

        GroupMemoryBarrierWithGroupSync();

        if (lane == 0) {
            uint running = s_tile_light_count;

            for (uint wave = 0; wave < wave_count; ++wave) {
                s_wave_offsets[wave] = running;
                running += s_wave_totals[wave];
            }

            s_tile_light_count = running;
        }

        GroupMemoryBarrierWithGroupSync();

        if (in_tile) {
            const uint slot = s_wave_offsets[wave_id] + wave_exclusive;
            s_tile_spheres[slot] = sphere;
            s_tile_light_indices[slot] = pc.visible_light_indices[visible_index];
        }

        GroupMemoryBarrierWithGroupSync();

        // Uniform: every lane reads the same shared count after the barrier.
        const bool last_block = base + group_size >= visible_count;

        if (s_tile_light_count != 0 && (last_block || s_tile_light_count + group_size > tile_light_batch)) {
            assign_batch_to_slices(ubo, tile_x, tile_y, wave_id, wave_count, lane_in_wave);

            GroupMemoryBarrierWithGroupSync();

            if (lane == 0) {
                s_tile_light_count = 0;
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }

    uint light_count = 0;

    if (lane < slice_count) {
        light_count = s_slice_counts[lane];
        pc.cluster_lights[cluster_flat_index(ubo, tile_x, tile_y, lane)] = light_count;
    }

    // Wave ops outside the branch, so every lane contributes.
    const uint capacity = ubo.cluster_light_capacity;
    const uint occupied = WaveActiveCountBits(light_count != 0);
    const uint overflowing = WaveActiveCountBits(light_count > capacity);
    const uint maximum = WaveActiveMax(light_count);
    const uint stored = WaveActiveSum(min(light_count, capacity));

    if (lane_in_wave == 0 && occupied != 0) {
        InterlockedAdd(pc.cluster_stats[0], occupied);
        InterlockedAdd(pc.cluster_stats[1], overflowing);
        InterlockedMax(pc.cluster_stats[2], maximum);
        InterlockedAdd(pc.cluster_stats[3], stored);
    }
}
