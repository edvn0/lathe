import scene_types;
import hiz_occlusion;

// GPU instance culling, in two entry points sharing CullPC (docs/occlusion-culling.md).
//
// main_cs (phase 1, prepare_frame): tests every instance against the camera frustum and, for opaque and mask batches,
// against last frame's Hi-Z (*occlusion, view [0]). Instances that pass are compacted into
// [first_instance, first_instance + n1) and dst_indirect gets that range. Frustum-visible instances the history says
// are hidden are not dropped: their source indices go to occlusion_candidates[first_instance, + c), and c is kept in
// the batch's first chunk.
//
// late_cs (phase 2, record_frame after the Hi-Z build): re-tests only the candidates against this frame's Hi-Z
// (*occlusion, view [1]) and appends the survivors at [first_instance + n1, + n2). late_indirect gets that range for
// the late prepass and merged_indirect gets [first_instance, + n1 + n2) for the forward pass. dst_indirect is read
// (n1), not written.
//
// Each runs as three dispatches (CullPC::flags' stage) so a batch of any size spreads over the whole GPU, rather than
// one workgroup looping over it:
//   test    one workgroup per chunk of 256 instances: test, keep the results as per-chunk bit masks and counts;
//   scan    one workgroup per batch: prefix-sum its chunks' counts into their output offsets, write its commands;
//   scatter one workgroup per chunk: copy each kept instance to its offset plus its rank within the chunk mask.
// The output order is the source order, as when one workgroup did it all, so the result is deterministic.
//
// With occlusion inactive both views are disabled, c = 0 and main_cs's output is exactly plain frustum culling.
// Per-meshlet culling happens later in the task shader. The shadow pass draws the un-culled buffers, since casters
// outside the camera frustum still cast into the cascades.

static const uint WG_SIZE = 256;

// Local-space AABB for one batch. Mirrors GpuCullBounds in renderer.hxx. wind_offset() moves a vertex's XZ by
// at most wind_strength, so growing the world AABB by wind_padding keeps swaying foliage from popping at the
// frustum edge. first_chunk is the batch's first CullChunk; batches' chunks are consecutive, in batch order.
public struct GpuCullBounds
{
    public float3 bounds_min;
    public float wind_padding;
    public float3 bounds_max;
    public uint first_chunk;
};

static const uint chunk_mask_words = WG_SIZE / 32;

// Per-chunk scratch, GPU-only. A chunk is WG_SIZE consecutive instances of one batch (candidates, for late_cs).
public struct CullChunk
{
    // main_cs: counts from test, offsets within the batch's output from scan.
    public uint survivor_count;
    public uint candidate_count;
    public uint survivor_offset;
    public uint candidate_offset;

    // late_cs, likewise.
    public uint late_count;
    public uint late_offset;

    // Only in a batch's first chunk: the batch's candidate count, for late_cs.
    public uint batch_candidates;
    public uint _padding;

    // Bit `lane` of word lane / 32: whether that instance of the chunk is kept.
    public uint survivor_mask[chunk_mask_words];
    public uint candidate_mask[chunk_mask_words];
    public uint late_mask[chunk_mask_words];
};

// CullPC::flags.
static const uint cull_flag_late_union_meshlet_batches = 1u;
static const uint cull_stage_shift = 8u;
static const uint cull_stage_test = 0u;
static const uint cull_stage_scan = 1u;
static const uint cull_stage_scatter = 2u;

// Dispatches wider than this go 2D (maxComputeWorkGroupCount is at least 65535 per dimension). Mirrors
// cull_dispatch_width in renderer.cxx.
static const uint cull_dispatch_width = 65535;

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

    // Source instance indices deferred by main_cs, at [first_instance, + the batch's candidate count) per batch.
    public Ptr<uint> occlusion_candidates;

    // chunk_count of them; see CullChunk.
    public Ptr<CullChunk> chunks;

    // Written by late_cs only.
    public Ptr<DrawCommand> merged_indirect;
    public Ptr<DrawCommand> late_indirect;

    // occlusion_stat_* counters, accumulated with atomics.
    public Ptr<uint> occlusion_stats;

    public uint batch_count;

    // Opaque and mask batches come first; only they are occluders and occludees. Blend batches are never deferred.
    public uint occludable_batch_count;

    // cull_flag_late_union_meshlet_batches, and the stage (cull_stage_*) << cull_stage_shift.
    //
    // The late-union flag is meshlet-level occlusion (docs/occlusion-culling.md, "Meshlet level"): it makes a meshlet
    // batch's late command cover [first_instance, + n1 + n2) instead of the phase-2 survivors alone, so the late
    // prepass retests the meshlets phase 1 deferred. Off without meshlet occlusion.
    public uint flags;

    public uint chunk_count;
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

groupshared uint s_wave_totals[WG_SIZE];
groupshared uint s_wave_offsets[WG_SIZE];
groupshared uint s_total;
groupshared uint s_masks[3][chunk_mask_words];

// Workgroup-wide exclusive prefix sum of one value per lane: `exclusive` sums the lanes before this one, `total` the
// whole workgroup. Every lane must call it, in uniform control flow; it ends with a barrier so the shared arrays can
// be reused straight away.
void workgroup_scan(uint value, uint lane, out uint exclusive, out uint total)
{
    uint wave_exclusive = WavePrefixSum(value);
    uint wave_total = WaveActiveSum(value);

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
        s_total = running_wave_total;
    }

    GroupMemoryBarrierWithGroupSync();

    exclusive = s_wave_offsets[wave_id] + wave_exclusive;
    total = s_total;

    GroupMemoryBarrierWithGroupSync();
}

uint cull_stage(CullPC pc)
{
    return pc.flags >> cull_stage_shift;
}

uint linear_group(uint3 group_id)
{
    return group_id.y * cull_dispatch_width + group_id.x;
}

uint batch_chunk_count(DrawCommand command)
{
    return (command.instance_count + WG_SIZE - 1) / WG_SIZE;
}

// The batch `chunk` belongs to: the last one whose first_chunk is at or before it.
uint chunk_batch(CullPC pc, uint chunk)
{
    uint low = 0;
    uint high = pc.batch_count - 1;

    while (low < high) {
        uint middle = (low + high + 1) / 2;
        if (pc.batch_bounds[middle].first_chunk <= chunk) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }

    return low;
}

// Rank of `lane` among the set bits of a chunk mask: how many kept lanes come before it.
uint mask_rank(uint mask[chunk_mask_words], uint lane)
{
    uint word = lane / 32;
    uint rank = countbits(mask[word] & ((1u << (lane % 32)) - 1u));

    for (uint i = 0; i < word; ++i) {
        rank += countbits(mask[i]);
    }

    return rank;
}

bool mask_bit(uint mask[chunk_mask_words], uint lane)
{
    return (mask[lane / 32] & (1u << (lane % 32))) != 0;
}

void clear_shared_masks(uint lane)
{
    if (lane < 3 * chunk_mask_words) {
        s_masks[lane / chunk_mask_words][lane % chunk_mask_words] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
}

void set_shared_mask_bit(uint mask_index, uint lane)
{
    InterlockedOr(s_masks[mask_index][lane / 32], 1u << (lane % 32));
}

// ---- main_cs

void main_test(CullPC pc, uint chunk, uint lane)
{
    uint batch = chunk_batch(pc, chunk);
    DrawCommand source_command = pc.src_indirect[batch];
    GpuCullBounds bounds = pc.batch_bounds[batch];

    uint local_index = (chunk - bounds.first_chunk) * WG_SIZE + lane;
    uint source_index = source_command.first_instance + local_index;

    bool in_frustum = false;
    bool occluded = false;

    if (local_index < source_command.instance_count) {
        float3 world_centre;
        float3 world_extent;
        instance_world_aabb(pc, bounds, source_index, world_centre, world_extent);

        in_frustum = frustum_visible(pc, world_centre, world_extent);
        occluded = in_frustum && batch < pc.occludable_batch_count &&
                   aabb_occluded(pc.occlusion[0], world_centre - world_extent, world_centre + world_extent);
    }

    clear_shared_masks(lane);

    if (in_frustum && !occluded) {
        set_shared_mask_bit(0, lane);
    }
    if (occluded) {
        set_shared_mask_bit(1, lane);
    }

    GroupMemoryBarrierWithGroupSync();

    if (lane < chunk_mask_words) {
        pc.chunks[chunk].survivor_mask[lane] = s_masks[0][lane];
        pc.chunks[chunk].candidate_mask[lane] = s_masks[1][lane];
    }

    if (lane == 0) {
        uint survivors = 0;
        uint candidates = 0;
        for (uint i = 0; i < chunk_mask_words; ++i) {
            survivors += countbits(s_masks[0][i]);
            candidates += countbits(s_masks[1][i]);
        }
        pc.chunks[chunk].survivor_count = survivors;
        pc.chunks[chunk].candidate_count = candidates;
    }
}

void main_scan(CullPC pc, uint batch, uint lane)
{
    DrawCommand source_command = pc.src_indirect[batch];
    uint first_chunk = pc.batch_bounds[batch].first_chunk;
    uint chunk_count = batch_chunk_count(source_command);

    uint running_survivors = 0;
    uint running_candidates = 0;

    for (uint base = 0; base < chunk_count; base += WG_SIZE) {
        uint chunk = first_chunk + base + lane;
        bool live = base + lane < chunk_count;

        uint survivors = live ? pc.chunks[chunk].survivor_count : 0;
        uint candidates = live ? pc.chunks[chunk].candidate_count : 0;

        uint survivor_offset;
        uint survivor_total;
        workgroup_scan(survivors, lane, survivor_offset, survivor_total);

        uint candidate_offset;
        uint candidate_total;
        workgroup_scan(candidates, lane, candidate_offset, candidate_total);

        if (live) {
            pc.chunks[chunk].survivor_offset = running_survivors + survivor_offset;
            pc.chunks[chunk].candidate_offset = running_candidates + candidate_offset;
        }

        running_survivors += survivor_total;
        running_candidates += candidate_total;
    }

    if (lane == 0) {
        DrawCommand out_command = source_command;
        out_command.instance_count = running_survivors;
        set_task_group_counts(out_command);
        pc.dst_indirect[batch] = out_command;

        // An empty batch owns no chunk; its first_chunk is the next batch's.
        if (chunk_count != 0) {
            pc.chunks[first_chunk].batch_candidates = running_candidates;
        }

        InterlockedAdd(pc.occlusion_stats[occlusion_stat_frustum_visible], running_survivors + running_candidates);
        InterlockedAdd(pc.occlusion_stats[occlusion_stat_early], running_survivors);
        InterlockedAdd(pc.occlusion_stats[occlusion_stat_candidates], running_candidates);
    }
}

void main_scatter(CullPC pc, uint chunk, uint lane)
{
    uint batch = chunk_batch(pc, chunk);
    uint first_instance = pc.src_indirect[batch].first_instance;
    uint source_index = first_instance + (chunk - pc.batch_bounds[batch].first_chunk) * WG_SIZE + lane;

    CullChunk state = pc.chunks[chunk];

    if (mask_bit(state.survivor_mask, lane)) {
        uint destination_index = first_instance + state.survivor_offset + mask_rank(state.survivor_mask, lane);

        pc.dst_draws[destination_index] = pc.src_draws[source_index];
        pc.dst_transforms[destination_index] = pc.src_transforms[source_index];
    }

    if (mask_bit(state.candidate_mask, lane)) {
        pc.occlusion_candidates[first_instance + state.candidate_offset + mask_rank(state.candidate_mask, lane)] =
                source_index;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_cs(uint3 group_id: SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform CullPC pc)
{
    uint group = linear_group(group_id);
    uint lane = group_thread_id.x;
    uint stage = cull_stage(pc);

    if (stage == cull_stage_scan) {
        if (group < pc.batch_count) {
            main_scan(pc, group, lane);
        }
        return;
    }

    if (group >= pc.chunk_count) {
        return;
    }

    if (stage == cull_stage_test) {
        main_test(pc, group, lane);
    } else {
        main_scatter(pc, group, lane);
    }
}

// ---- late_cs

void late_test(CullPC pc, uint chunk, uint lane)
{
    uint batch = chunk_batch(pc, chunk);
    DrawCommand source_command = pc.src_indirect[batch];
    GpuCullBounds bounds = pc.batch_bounds[batch];

    uint candidate_count = pc.chunks[bounds.first_chunk].batch_candidates;
    uint local_index = (chunk - bounds.first_chunk) * WG_SIZE + lane;

    bool visible = false;

    if (local_index < candidate_count) {
        uint source_index = pc.occlusion_candidates[source_command.first_instance + local_index];

        float3 world_centre;
        float3 world_extent;
        instance_world_aabb(pc, bounds, source_index, world_centre, world_extent);

        visible = !aabb_occluded(pc.occlusion[0], world_centre - world_extent, world_centre + world_extent);
    }

    clear_shared_masks(lane);

    if (visible) {
        set_shared_mask_bit(2, lane);
    }

    GroupMemoryBarrierWithGroupSync();

    if (lane < chunk_mask_words) {
        pc.chunks[chunk].late_mask[lane] = s_masks[2][lane];
    }

    if (lane == 0) {
        uint late = 0;
        for (uint i = 0; i < chunk_mask_words; ++i) {
            late += countbits(s_masks[2][i]);
        }
        pc.chunks[chunk].late_count = late;
    }
}

void late_scan(CullPC pc, uint batch, uint lane)
{
    DrawCommand source_command = pc.src_indirect[batch];
    uint first_chunk = pc.batch_bounds[batch].first_chunk;
    uint chunk_count = batch_chunk_count(source_command);

    // main_cs's phase-1 command; its survivors fill [first_instance, + early_count).
    uint early_count = pc.dst_indirect[batch].instance_count;

    uint running_late = 0;

    for (uint base = 0; base < chunk_count; base += WG_SIZE) {
        uint chunk = first_chunk + base + lane;
        bool live = base + lane < chunk_count;

        uint late = live ? pc.chunks[chunk].late_count : 0;

        uint late_offset;
        uint late_total;
        workgroup_scan(late, lane, late_offset, late_total);

        if (live) {
            pc.chunks[chunk].late_offset = running_late + late_offset;
        }

        running_late += late_total;
    }

    if (lane == 0) {
        DrawCommand late_command = source_command;
        late_command.first_instance = source_command.first_instance + early_count;
        late_command.instance_count = running_late;

        if ((pc.flags & cull_flag_late_union_meshlet_batches) != 0 && source_command.meshlet_count != 0) {
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

void late_scatter(CullPC pc, uint chunk, uint lane)
{
    uint batch = chunk_batch(pc, chunk);
    uint first_instance = pc.src_indirect[batch].first_instance;
    uint local_index = (chunk - pc.batch_bounds[batch].first_chunk) * WG_SIZE + lane;

    CullChunk state = pc.chunks[chunk];

    if (!mask_bit(state.late_mask, lane)) {
        return;
    }

    uint source_index = pc.occlusion_candidates[first_instance + local_index];
    uint early_count = pc.dst_indirect[batch].instance_count;
    uint destination_index = first_instance + early_count + state.late_offset + mask_rank(state.late_mask, lane);

    pc.dst_draws[destination_index] = pc.src_draws[source_index];
    pc.dst_transforms[destination_index] = pc.src_transforms[source_index];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void late_cs(uint3 group_id: SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform CullPC pc)
{
    uint group = linear_group(group_id);
    uint lane = group_thread_id.x;
    uint stage = cull_stage(pc);

    if (stage == cull_stage_scan) {
        if (group < pc.batch_count) {
            late_scan(pc, group, lane);
        }
        return;
    }

    if (group >= pc.chunk_count) {
        return;
    }

    if (stage == cull_stage_test) {
        late_test(pc, group, lane);
    } else {
        late_scatter(pc, group, lane);
    }
}
