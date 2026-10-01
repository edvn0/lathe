import scene_types;
import hiz_occlusion;

// GPU instance culling, one workgroup per batch, in two entry points sharing CullPC (docs/occlusion-culling.md).
//
// main_cs (phase 1, prepare_frame): tests the batch's instances against the camera frustum and, for opaque and mask
// batches, against last frame's Hi-Z (*occlusion, view [0]). Instances that pass are compacted, with a shared-memory
// prefix sum, into [first_instance, first_instance + n1) and dst_indirect gets that range. Frustum-visible instances
// the history says are hidden are not dropped: their source indices go to occlusion_candidates[first_instance, + c)
// and candidate_counts[batch] = c.
//
// late_cs (phase 2, record_frame after the Hi-Z build): re-tests only the candidates against this frame's Hi-Z
// (*occlusion, view [1]) and appends the survivors at [first_instance + n1, + n2). late_indirect gets that range for
// the late prepass and merged_indirect gets [first_instance, + n1 + n2) for the forward pass. dst_indirect is read
// (n1), not written.
//
// With occlusion inactive both views are disabled, c = 0 and main_cs's output is exactly plain frustum culling. The
// compaction is deterministic and per-meshlet culling happens later in the task shader. The shadow pass draws the
// un-culled buffers, since casters outside the camera frustum still cast into the cascades.

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

    // main_cs: view [0] (last frame's pyramid); late_cs: view [1] (this frame's).
    public Ptr<OcclusionView> occlusion;

    // Source instance indices deferred by main_cs, at [first_instance, + candidate_counts[batch]) per batch.
    public Ptr<uint> occlusion_candidates;
    public Ptr<uint> candidate_counts;

    // Written by late_cs only.
    public Ptr<DrawCommand> merged_indirect;
    public Ptr<DrawCommand> late_indirect;

    // occlusion_stat_* counters, accumulated with atomics.
    public Ptr<uint> occlusion_stats;

    public uint batch_count;

    // Opaque and mask batches come first; only they are occluders and occludees. Blend batches are never deferred.
    public uint occludable_batch_count;

    // Meshlet-level occlusion hook (docs/occlusion-culling-m2.md): non-zero makes a meshlet batch's late command cover
    // [first_instance, + n1 + n2) instead of the phase-2 survivors alone. Always 0 until then.
    public uint late_union_meshlet_batches;
    public uint _padding;
};

// OcclusionView::stats / CullPC::occlusion_stats slots. Mirrors occlusion_stat_* in renderer.hxx.
static const uint occlusion_stat_frustum_visible = 0;
static const uint occlusion_stat_early = 1;
static const uint occlusion_stat_candidates = 2;
static const uint occlusion_stat_late = 3;

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

// The instance's world-space AABB, grown for wind sway.
void instance_world_aabb(CullPC pc, GpuCullBounds bounds, uint instance_index, out float3 world_centre,
                         out float3 world_extent)
{
    float3 local_centre = (bounds.bounds_min + bounds.bounds_max) * 0.5;
    float3 local_extent = (bounds.bounds_max - bounds.bounds_min) * 0.5;

    transform_aabb(pc.src_transforms[instance_index], local_centre, local_extent, bounds.wind_padding, world_centre,
                   world_extent);
}

bool frustum_visible(CullPC pc, float3 world_centre, float3 world_extent)
{
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

// Workgroup-wide compaction of one flag per lane: `exclusive` is the number of flagged lanes before this one and
// `total` the number flagged in the workgroup. Every lane must call it, in uniform control flow; it ends with a
// barrier so the shared arrays can be reused straight away.
void workgroup_compact(bool flag, uint lane, out uint exclusive, out uint total)
{
    uint wave_exclusive = WavePrefixCountBits(flag);
    uint wave_total = WaveActiveCountBits(flag);

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

    exclusive = s_wave_offsets[wave_id] + wave_exclusive;
    total = s_chunk_total;

    GroupMemoryBarrierWithGroupSync();
}

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

    OcclusionView occlusion = pc.occlusion[0];
    bool occludable = batch < pc.occludable_batch_count;

    uint running_survivors = 0;
    uint running_candidates = 0;

    for (uint base = 0; base < source_command.instance_count; base += WG_SIZE) {
        uint local_index = base + lane;
        uint source_index = source_command.first_instance + local_index;
        bool in_frustum = false;
        bool occluded = false;

        if (local_index < source_command.instance_count) {
            float3 world_centre;
            float3 world_extent;
            instance_world_aabb(pc, bounds, source_index, world_centre, world_extent);

            in_frustum = frustum_visible(pc, world_centre, world_extent);
            occluded = in_frustum && occludable &&
                       aabb_occluded(occlusion, world_centre - world_extent, world_centre + world_extent);
        }

        bool draw_now = in_frustum && !occluded;

        uint survivor_offset;
        uint chunk_survivors;
        workgroup_compact(draw_now, lane, survivor_offset, chunk_survivors);

        uint candidate_offset;
        uint chunk_candidates;
        workgroup_compact(occluded, lane, candidate_offset, chunk_candidates);

        if (draw_now) {
            uint destination_index = source_command.first_instance + running_survivors + survivor_offset;

            pc.dst_draws[destination_index] = pc.src_draws[source_index];
            pc.dst_transforms[destination_index] = pc.src_transforms[source_index];
        }

        if (occluded) {
            pc.occlusion_candidates[source_command.first_instance + running_candidates + candidate_offset] =
                    source_index;
        }

        running_survivors += chunk_survivors;
        running_candidates += chunk_candidates;
    }

    if (lane == 0) {
        DrawCommand out_command = source_command;
        out_command.instance_count = running_survivors;
        set_task_group_counts(out_command);
        pc.dst_indirect[batch] = out_command;

        pc.candidate_counts[batch] = running_candidates;

        InterlockedAdd(pc.occlusion_stats[occlusion_stat_frustum_visible], running_survivors + running_candidates);
        InterlockedAdd(pc.occlusion_stats[occlusion_stat_early], running_survivors);
        InterlockedAdd(pc.occlusion_stats[occlusion_stat_candidates], running_candidates);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void late_cs(uint3 group_id: SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform CullPC pc)
{
    uint batch = group_id.x;

    if (batch >= pc.batch_count) {
        return;
    }

    DrawCommand source_command = pc.src_indirect[batch];
    GpuCullBounds bounds = pc.batch_bounds[batch];
    uint lane = group_thread_id.x;

    // main_cs's phase-1 command; its survivors fill [first_instance, + early_count).
    uint early_count = pc.dst_indirect[batch].instance_count;
    uint candidate_count = pc.candidate_counts[batch];

    OcclusionView occlusion = pc.occlusion[0];

    uint running_late = 0;

    for (uint base = 0; base < candidate_count; base += WG_SIZE) {
        uint local_index = base + lane;
        uint source_index = 0;
        bool visible = false;

        if (local_index < candidate_count) {
            source_index = pc.occlusion_candidates[source_command.first_instance + local_index];

            float3 world_centre;
            float3 world_extent;
            instance_world_aabb(pc, bounds, source_index, world_centre, world_extent);

            visible = !aabb_occluded(occlusion, world_centre - world_extent, world_centre + world_extent);
        }

        uint late_offset;
        uint chunk_late;
        workgroup_compact(visible, lane, late_offset, chunk_late);

        if (visible) {
            uint destination_index = source_command.first_instance + early_count + running_late + late_offset;

            pc.dst_draws[destination_index] = pc.src_draws[source_index];
            pc.dst_transforms[destination_index] = pc.src_transforms[source_index];
        }

        running_late += chunk_late;
    }

    if (lane == 0) {
        DrawCommand late_command = source_command;
        late_command.first_instance = source_command.first_instance + early_count;
        late_command.instance_count = running_late;

        if (pc.late_union_meshlet_batches != 0 && source_command.meshlet_count != 0) {
            late_command.first_instance = source_command.first_instance;
            late_command.instance_count = early_count + running_late;
        }

        set_task_group_counts(late_command);
        pc.late_indirect[batch] = late_command;

        DrawCommand merged_command = source_command;
        merged_command.instance_count = early_count + running_late;
        set_task_group_counts(merged_command);
        pc.merged_indirect[batch] = merged_command;

        InterlockedAdd(pc.occlusion_stats[occlusion_stat_late], running_late);
    }
}
