import scene_types;

// GPU frustum culling, one workgroup per batch. Each workgroup tests the batch's instances against the camera
// frustum and compacts the survivors, with a shared-memory prefix sum, into
// [first_instance, first_instance + survivors). The compaction is deterministic and the dispatch size is the
// batch count. It then rewrites the batch's instance count and task group counts; per-meshlet culling
// happens later in the task shader.
//
// Only the main view reads the result. The shadow pass draws the un-culled buffers, since casters outside the
// camera frustum still cast into the cascades.

static const uint WG_SIZE = 256;

// Local-space AABB for one batch. Mirrors GpuCullBounds in renderer.hxx. wind_offset() moves a vertex's XZ by
// at most wind_strength, so growing the world AABB by wind_padding keeps swaying foliage from popping at the
// frustum edge.
public struct GpuCullBounds
{
    public float3 bounds_min;
    public float wind_padding;
    public float3 bounds_max;
    public float pad1;
};

// Guaranteed minimum of maxTaskWorkGroupCount. Mirrors meshlet.hxx.
static const uint max_task_group_count_x = 65535;

void set_task_group_counts(inout DrawCommand command)
{
    const uint chunk_count = (command.meshlet_count + meshlets_per_task - 1) / meshlets_per_task;
    const uint total = command.instance_count * chunk_count;

    command.group_count_x = min(total, max_task_group_count_x);
    command.group_count_y = total == 0 ? 1 : (total + command.group_count_x - 1) / command.group_count_x;
    command.group_count_z = 1;
}

public struct CullPC
{
    public Ptr<GpuDraw> src_draws;
    public Ptr<float4x4> src_transforms;
    public Ptr<GpuCullBounds> batch_bounds;

    // The batch's instance range and meshlet_count.
    public Ptr<DrawCommand> src_indirect;

    // Written by thread 0 only; one workgroup per batch means one writer.
    public Ptr<DrawCommand> dst_indirect;

    public Ptr<GpuDraw> dst_draws;
    public Ptr<float4x4> dst_transforms;

    // Inward-facing planes (xyz = normal, w = distance): left, right, bottom, top, near, far.
    public Ptr<float4> frustum_planes;

    public uint batch_count;
    public uint padding;
};

// World-space centre/extent of a local AABB: summing the absolute transformed axes gives the tightest
// axis-aligned box.
void transform_aabb(float4x4 model, float3 local_centre, float3 local_extent, float wind_padding,
                    out float3 world_centre, out float3 world_extent)
{
    world_centre = mul(model, float4(local_centre, 1.0)).xyz;

    float3 axis_x = mul(model, float4(local_extent.x, 0.0, 0.0, 0.0)).xyz;
    float3 axis_y = mul(model, float4(0.0, local_extent.y, 0.0, 0.0)).xyz;
    float3 axis_z = mul(model, float4(0.0, 0.0, local_extent.z, 0.0)).xyz;

    world_extent = abs(axis_x) + abs(axis_y) + abs(axis_z);

    // Wind only moves X/Z.
    world_extent.xz += wind_padding;
}

bool instance_visible(CullPC pc, GpuCullBounds bounds, uint instance_index)
{
    float3 local_centre = (bounds.bounds_min + bounds.bounds_max) * 0.5;
    float3 local_extent = (bounds.bounds_max - bounds.bounds_min) * 0.5;

    float4x4 model = pc.src_transforms[instance_index];

    float3 world_centre;
    float3 world_extent;
    transform_aabb(model, local_centre, local_extent, bounds.wind_padding, world_centre, world_extent);

    [unroll]
    for (uint plane_index = 0; plane_index < 6; ++plane_index) {
        float4 plane = pc.frustum_planes[plane_index];

        float distance = dot(plane.xyz, world_centre) + plane.w;
        float radius = dot(abs(plane.xyz), world_extent);

        if (distance + radius < 0.0) {
            return false;
        }
    }

    return true;
}


// Separate arrays for the read and write phases.
groupshared uint s_wave_totals[WG_SIZE];
groupshared uint s_wave_offsets[WG_SIZE];
groupshared uint s_chunk_total;

[shader("compute")]
[numthreads(256, 1, 1)]
void main_cs(uint3 group_id: SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform CullPC pc)
{
    uint batch = group_id.x;

    if (batch >= pc.batch_count) {
        return;
    }

    DrawCommand source_command = pc.src_indirect[batch];
    GpuCullBounds bounds = pc.batch_bounds[batch];
    uint lane = group_thread_id.x;

    uint running_survivors = 0;

    for (uint base = 0; base < source_command.instance_count; base += WG_SIZE) {
        uint local_index = base + lane;
        bool visible = false;

        if (local_index < source_command.instance_count) {
            visible = instance_visible(pc, bounds, source_command.first_instance + local_index);
        }

        uint wave_exclusive = WavePrefixCountBits(visible);
        uint wave_total = WaveActiveCountBits(visible);

        uint lane_in_wave = WaveGetLaneIndex();
        uint wave_id = lane / WaveGetLaneCount();

        if (lane_in_wave == 0) {
            s_wave_totals[wave_id] = wave_total;
        }

        GroupMemoryBarrierWithGroupSync();

        if (lane == 0) {
            uint num_waves = (WG_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();
            uint running_wave_total = 0;
            for (uint i = 0; i < num_waves; ++i) {
                s_wave_offsets[i] = running_wave_total;
                running_wave_total += s_wave_totals[i];
            }
            s_chunk_total = running_wave_total;
        }

        GroupMemoryBarrierWithGroupSync();

        uint exclusive_scan = s_wave_offsets[wave_id] + wave_exclusive;
        uint chunk_survivor_count = s_chunk_total;

        if (visible) {
            uint source_index = source_command.first_instance + local_index;
            uint destination_index = source_command.first_instance + running_survivors + exclusive_scan;

            pc.dst_draws[destination_index] = pc.src_draws[source_index];
            pc.dst_transforms[destination_index] = pc.src_transforms[source_index];
        }

        running_survivors += chunk_survivor_count;

        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0) {
        DrawCommand out_command = source_command;
        out_command.instance_count = running_survivors;
        set_task_group_counts(out_command);
        pc.dst_indirect[batch] = out_command;
    }
}