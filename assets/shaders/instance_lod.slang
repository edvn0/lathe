import scene_types;

// GPU level-of-detail selection for resident instanced models (docs/occlusion-culling.md, "Resident instances").
//
// A resident instanced model keeps its transforms in a GPU buffer from frame to frame, so the CPU neither copies them
// nor picks their LODs. Each frame it emits, per submesh, one batch per distinct (geometry, material) its LODs use
// (an "LOD group"), each reserving a slot for every instance, and one LodJob tying them together. This shader then
// gives every instance the LOD Renderer::prepare_frame() would have (the first lod_distances entry its origin is
// closer than, as select_lod_index() does) and writes its GpuDraw and transform into its group's batch, in the frame's
// source draw and transform arrays, and the group's instance count into its indirect command. Everything after it
// (shadows, main_cs, late_cs, the draws) reads those arrays as if the CPU had written them.
//
// Three dispatches, like frustum_cull.slang, so the result is deterministic (source order within each group):
//   test    one workgroup per chunk of 256 instances: count the chunk's instances per group;
//   scan    one workgroup per job: prefix-sum its chunks' counts per group, write the groups' commands;
//   scatter one workgroup per chunk: write each instance at its group offset plus its rank in the chunk.

static const uint WG_SIZE = 256;

// lod_count in config.hxx.
static const uint max_lod_groups = 4;

// One output batch of a job. Mirrors GpuLodGroup in renderer.hxx.
public struct LodGroup
{
    // Its indirect command (source array index), where its instances start in the source draw and transform arrays,
    // and, for meshlet batches, its meshlet count and first meshlet visibility bit (0 / 0 otherwise).
    public uint batch;
    public uint first_instance;
    public uint meshlet_count;
    public uint first_meshlet_bit;

    // Every instance's GpuDraw but meshlet_visibility_offset.
    public GpuDraw draw;
};

// One submesh of one resident instanced model. Mirrors GpuLodJob in renderer.hxx.
public struct LodJob
{
    public Ptr<float4x4> transforms;
    public uint instance_count;

    // The job's chunks are consecutive from here, ceil(instance_count / WG_SIZE) of them, in job order.
    public uint first_chunk;
    public uint group_count;

    // Byte `lod` is the group LOD `lod` draws with.
    public uint lod_groups;
    public uint _pad0;
    public uint _pad1;

    public LodGroup groups[max_lod_groups];
};

// Per-chunk scratch: test's counts and scan's offsets, per group.
public struct LodChunk
{
    public uint counts[max_lod_groups];
    public uint offsets[max_lod_groups];
};

static const uint lod_stage_test = 0u;
static const uint lod_stage_scan = 1u;
static const uint lod_stage_scatter = 2u;

// Mirrors cull_dispatch_width in renderer.cxx.
static const uint lod_dispatch_width = 65535;

// Mirrors maximum_meshlet_visibility_bits in meshlet_visibility.hxx.
static const uint64_t maximum_meshlet_visibility_bits = 1ull << 28;

public struct LodPC
{
    public Ptr<LodJob> jobs;
    public Ptr<LodChunk> chunks;

    // The frame's source arrays (what main_cs culls and the shadow pass draws).
    public Ptr<GpuDraw> draws;
    public Ptr<float4x4> transforms;
    public Ptr<DrawCommand> indirect;

    public float camera_x;
    public float camera_y;
    public float camera_z;
    public uint job_count;

    // lod_distances, squared.
    public float lod_distance_sq0;
    public float lod_distance_sq1;
    public float lod_distance_sq2;
    public uint chunk_count;

    public uint stage;
    public uint _padding;
};

// Same as frustum_cull.slang's: the task-shader grid for a command's instances.
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

// The job `chunk` belongs to: the last one whose first_chunk is at or before it.
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

// The group instance `local_index` of `job` draws with. Matches select_lod_index() in renderer.cxx: distance from the
// instance's origin to the camera against lod_distances.
uint instance_group(LodPC pc, LodJob job, uint local_index)
{
    const float3 origin = mul(job.transforms[local_index], float4(0.0, 0.0, 0.0, 1.0)).xyz;
    const float3 delta = origin - float3(pc.camera_x, pc.camera_y, pc.camera_z);
    const float distance_sq = dot(delta, delta);

    uint lod = 3;
    if (distance_sq < pc.lod_distance_sq0) {
        lod = 0;
    } else if (distance_sq < pc.lod_distance_sq1) {
        lod = 1;
    } else if (distance_sq < pc.lod_distance_sq2) {
        lod = 2;
    }

    return (job.lod_groups >> (lod * 8)) & 0xFFu;
}

groupshared uint s_counts[max_lod_groups];
groupshared uint s_wave_totals[WG_SIZE];
groupshared uint s_wave_offsets[WG_SIZE];
groupshared uint s_total;

// Workgroup-wide exclusive prefix sum; see frustum_cull.slang's workgroup_scan. Uniform control flow only.
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

    // One shared atomic per wave and group: a lane each would serialise the whole workgroup on one address.
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

    // The group loop runs the same number of times on every lane, so the scans stay in uniform control flow.
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
