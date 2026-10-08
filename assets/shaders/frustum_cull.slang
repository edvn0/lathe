import scene_types;
import hiz_occlusion;

static const uint WG_SIZE = 256;

public struct GpuCullBounds
{
    public float3 bounds_min;
    public float wind_padding;
    public float3 bounds_max;
    public uint first_chunk;
};

static const uint chunk_mask_words = WG_SIZE / 32;

public struct CullChunk
{
    public uint survivor_count;
    public uint candidate_count;
    public uint survivor_offset;
    public uint candidate_offset;

    public uint late_count;
    public uint late_offset;

    public uint batch_candidates;
    public uint _padding;

    public uint survivor_mask[chunk_mask_words];
    public uint candidate_mask[chunk_mask_words];
    public uint late_mask[chunk_mask_words];
};

static const uint cull_flag_late_union_meshlet_batches = 1u;
static const uint cull_stage_shift = 8u;
static const uint cull_stage_test = 0u;
static const uint cull_stage_scan = 1u;
static const uint cull_stage_scatter = 2u;

static const uint cull_dispatch_width = 65535;

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

    public Ptr<DrawCommand> src_indirect;

    public Ptr<DrawCommand> dst_indirect;

    public Ptr<GpuDraw> dst_draws;
    public Ptr<float4x4> dst_transforms;

    public Ptr<float4> frustum_planes;

    public Ptr<OcclusionView> occlusion;

    public Ptr<uint> occlusion_candidates;

    public Ptr<CullChunk> chunks;

    public Ptr<DrawCommand> merged_indirect;
    public Ptr<DrawCommand> late_indirect;

    public Ptr<uint> occlusion_stats;

    public uint batch_count;

    public uint occludable_batch_count;

    public uint flags;

    public uint chunk_count;
};

static const uint occlusion_stat_frustum_visible = 0;
static const uint occlusion_stat_early = 1;
static const uint occlusion_stat_candidates = 2;
static const uint occlusion_stat_late = 3;

void transform_aabb(float4x4 model, float3 local_centre, float3 local_extent, float wind_padding,
                    out float3 world_centre, out float3 world_extent)
{
    world_centre = mul(model, float4(local_centre, 1.0)).xyz;

    float3 axis_x = mul(model, float4(local_extent.x, 0.0, 0.0, 0.0)).xyz;
    float3 axis_y = mul(model, float4(0.0, local_extent.y, 0.0, 0.0)).xyz;
    float3 axis_z = mul(model, float4(0.0, 0.0, local_extent.z, 0.0)).xyz;

    world_extent = abs(axis_x) + abs(axis_y) + abs(axis_z);

    world_extent.xz += wind_padding;
}

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

bool chunk_past_end(uint chunk, uint first_chunk, uint count)
{
    return (chunk - first_chunk) * WG_SIZE >= count;
}

void main_test(CullPC pc, uint chunk, uint lane)
{
    uint batch = chunk_batch(pc, chunk);
    DrawCommand source_command = pc.src_indirect[batch];
    GpuCullBounds bounds = pc.batch_bounds[batch];

    if (chunk_past_end(chunk, bounds.first_chunk, source_command.instance_count)) {
        return;
    }

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
    uint first_chunk = pc.batch_bounds[batch].first_chunk;

    if (chunk_past_end(chunk, first_chunk, pc.src_indirect[batch].instance_count)) {
        return;
    }

    uint source_index = first_instance + (chunk - first_chunk) * WG_SIZE + lane;

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

void late_test(CullPC pc, uint chunk, uint lane)
{
    uint batch = chunk_batch(pc, chunk);
    DrawCommand source_command = pc.src_indirect[batch];
    GpuCullBounds bounds = pc.batch_bounds[batch];

    uint candidate_count = pc.chunks[bounds.first_chunk].batch_candidates;

    if (chunk_past_end(chunk, bounds.first_chunk, source_command.instance_count)) {
        return;
    }

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
    uint first_chunk = pc.batch_bounds[batch].first_chunk;

    if (chunk_past_end(chunk, first_chunk, pc.src_indirect[batch].instance_count)) {
        return;
    }

    uint local_index = (chunk - first_chunk) * WG_SIZE + lane;

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
