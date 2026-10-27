import scene_types;

static const uint WG_SIZE = 256;

static const uint max_lod_groups = 4;

public struct LodGroup
{
    public uint batch;
    public uint first_instance;
    public uint meshlet_count;
    public uint first_meshlet_bit;

    public GpuDraw draw;
};

public struct LodJob
{
    public Ptr<float4x4> transforms;
    public uint instance_count;

    public uint first_chunk;
    public uint group_count;

    public uint lod_groups;

    // The mesh's bounding sphere as four half floats: centre x and y, then centre z and radius.
    public uint sphere_xy;
    public uint sphere_zr;

    public LodGroup groups[max_lod_groups];
};

public struct LodChunk
{
    public uint counts[max_lod_groups];
    public uint offsets[max_lod_groups];
};

static const uint lod_stage_test = 0u;
static const uint lod_stage_scan = 1u;
static const uint lod_stage_scatter = 2u;

static const uint lod_dispatch_width = 65535;

static const uint64_t maximum_meshlet_visibility_bits = 1ull << 28;

public struct LodPC
{
    public Ptr<LodJob> jobs;
    public Ptr<LodChunk> chunks;

    public Ptr<GpuDraw> draws;
    public Ptr<float4x4> transforms;
    public Ptr<DrawCommand> indirect;

    public float camera_x;
    public float camera_y;
    public float camera_z;
    public uint job_count;

    public float lod_pixel_scale;
    public float lod_radius_pixels_sq0;
    public float lod_radius_pixels_sq1;
    public float lod_radius_pixels_sq2;

    public uint chunk_count;
    public uint stage;
    public uint _padding;
    public uint _padding1;
};

static const uint max_task_group_count_x = 65535;

void set_task_group_counts(inout DrawCommand command)
{
    const uint chunk_count = (command.meshlet_count + meshlets_per_task - 1) / meshlets_per_task;
    const uint total = command.instance_count * chunk_count;

    command.group_count_x = min(total, max_task_group_count_x);
    command.group_count_y = total == 0 ? 1 : (total + command.group_count_x - 1) / command.group_count_x;
    command.group_count_z = 1;
}

uint linear_group(uint3 group_id)
{
    return group_id.y * lod_dispatch_width + group_id.x;
}

uint chunk_job(LodPC pc, uint chunk)
{
    uint low = 0;
    uint high = pc.job_count - 1;

    while (low < high) {
        uint middle = (low + high + 1) / 2;
        if (pc.jobs[middle].first_chunk <= chunk) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }

    return low;
}

uint instance_group(LodPC pc, LodJob job, uint local_index)
{
    const float4x4 transform = job.transforms[local_index];
    const float3 local_centre = float3(f16tof32(job.sphere_xy & 0xFFFFu), f16tof32(job.sphere_xy >> 16),
                                       f16tof32(job.sphere_zr & 0xFFFFu));
    const float local_radius = f16tof32(job.sphere_zr >> 16);
    const float3 centre = mul(transform, float4(local_centre, 1.0)).xyz;
    const float3 delta = centre - float3(pc.camera_x, pc.camera_y, pc.camera_z);
    const float distance_sq = dot(delta, delta);

    // The same test as lod_index_for on the CPU: how many pixels the instance's bounding sphere spans. The columns of
    // the transform are the basis vectors, and the largest of them is the instance's scale.
    const float3 column0 = float3(transform[0][0], transform[1][0], transform[2][0]);
    const float3 column1 = float3(transform[0][1], transform[1][1], transform[2][1]);
    const float3 column2 = float3(transform[0][2], transform[1][2], transform[2][2]);
    const float scale_sq = max(dot(column0, column0), max(dot(column1, column1), dot(column2, column2)));
    const float radius_pixels = local_radius * pc.lod_pixel_scale;
    const float radius_pixels_sq = radius_pixels * radius_pixels * scale_sq;

    uint lod = 3;
    if (radius_pixels_sq >= pc.lod_radius_pixels_sq0 * distance_sq) {
        lod = 0;
    } else if (radius_pixels_sq >= pc.lod_radius_pixels_sq1 * distance_sq) {
        lod = 1;
    } else if (radius_pixels_sq >= pc.lod_radius_pixels_sq2 * distance_sq) {
        lod = 2;
    }

    return (job.lod_groups >> (lod * 8)) & 0xFFu;
}

groupshared uint s_counts[max_lod_groups];
groupshared uint s_wave_totals[WG_SIZE];
groupshared uint s_wave_offsets[WG_SIZE];
groupshared uint s_total;

void workgroup_scan(uint value, uint lane, out uint exclusive, out uint total)
{
    uint wave_exclusive = WavePrefixSum(value);
    uint wave_total = WaveActiveSum(value);

    uint wave_id = lane / WaveGetLaneCount();

    if (WaveGetLaneIndex() == 0) {
        s_wave_totals[wave_id] = wave_total;
    }

    GroupMemoryBarrierWithGroupSync();

    if (lane == 0) {
        uint num_waves = (WG_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();
        uint running = 0;
        for (uint i = 0; i < num_waves; ++i) {
            s_wave_offsets[i] = running;
            running += s_wave_totals[i];
        }
        s_total = running;
    }

    GroupMemoryBarrierWithGroupSync();

    exclusive = s_wave_offsets[wave_id] + wave_exclusive;
    total = s_total;

    GroupMemoryBarrierWithGroupSync();
}

void lod_test(LodPC pc, uint chunk, uint lane)
{
    const uint job_index = chunk_job(pc, chunk);
    const LodJob job = pc.jobs[job_index];
    const uint local_index = (chunk - job.first_chunk) * WG_SIZE + lane;

    if (lane < max_lod_groups) {
        s_counts[lane] = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    const uint my_group = local_index < job.instance_count ? instance_group(pc, job, local_index) : max_lod_groups;

    for (uint group = 0; group < job.group_count; ++group) {
        const uint count = WaveActiveCountBits(my_group == group);
        if (WaveIsFirstLane() && count != 0) {
            InterlockedAdd(s_counts[group], count);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (lane < max_lod_groups) {
        pc.chunks[chunk].counts[lane] = s_counts[lane];
    }
}

void lod_scan(LodPC pc, uint job_index, uint lane)
{
    const LodJob job = pc.jobs[job_index];
    const uint chunk_count = (job.instance_count + WG_SIZE - 1) / WG_SIZE;

    for (uint group = 0; group < job.group_count; ++group) {
        uint running = 0;

        for (uint base = 0; base < chunk_count; base += WG_SIZE) {
            const uint chunk = job.first_chunk + base + lane;
            const bool live = base + lane < chunk_count;

            const uint count = live ? pc.chunks[chunk].counts[group] : 0;

            uint offset;
            uint total;
            workgroup_scan(count, lane, offset, total);

            if (live) {
                pc.chunks[chunk].offsets[group] = running + offset;
            }

            running += total;
        }

        if (lane == 0) {
            const uint batch = job.groups[group].batch;
            DrawCommand command = pc.indirect[batch];
            command.instance_count = running;
            set_task_group_counts(command);
            pc.indirect[batch] = command;
        }
    }
}

void lod_scatter(LodPC pc, uint chunk, uint lane)
{
    const uint job_index = chunk_job(pc, chunk);
    const LodJob job = pc.jobs[job_index];
    const uint local_index = (chunk - job.first_chunk) * WG_SIZE + lane;
    const bool live = local_index < job.instance_count;

    const uint my_group = live ? instance_group(pc, job, local_index) : max_lod_groups;
    const LodChunk state = pc.chunks[chunk];

    for (uint group = 0; group < job.group_count; ++group) {
        uint rank;
        uint total;
        workgroup_scan(my_group == group ? 1u : 0u, lane, rank, total);

        if (my_group == group) {
            const LodGroup target = job.groups[group];
            const uint slot = state.offsets[group] + rank;
            const uint destination = target.first_instance + slot;

            GpuDraw draw = target.draw;
            draw.meshlet_visibility_offset = 0;
            if (target.meshlet_count != 0) {
                const uint64_t first_bit = uint64_t(target.first_meshlet_bit) + uint64_t(slot) * target.meshlet_count;
                draw.meshlet_visibility_offset = first_bit < maximum_meshlet_visibility_bits ? uint(first_bit) : 0u;
            }

            pc.draws[destination] = draw;
            pc.transforms[destination] = job.transforms[local_index];
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_cs(uint3 group_id: SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform LodPC pc)
{
    const uint group = linear_group(group_id);
    const uint lane = group_thread_id.x;

    if (pc.stage == lod_stage_scan) {
        if (group < pc.job_count) {
            lod_scan(pc, group, lane);
        }
        return;
    }

    if (group >= pc.chunk_count) {
        return;
    }

    if (pc.stage == lod_stage_test) {
        lod_test(pc, group, lane);
    } else {
        lod_scatter(pc, group, lane);
    }
}
