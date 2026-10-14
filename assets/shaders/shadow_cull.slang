import scene_types;

static const uint WG_SIZE = 256;
static const uint cull_dispatch_width = 65535;
static const uint max_task_group_count_x = 65535;

static const uint stage_clear = 0u;
static const uint stage_cull = 1u;
static const uint stage_finalize = 2u;

// Mirrors GpuCullBounds in frustum_cull.slang.
struct GpuCullBounds
{
    float3 bounds_min;
    float wind_padding;
    float3 bounds_max;
    uint first_chunk;
};

struct ShadowCullPushConstants
{
    Ptr<float4x4> src_transforms;
    Ptr<GpuCullBounds> batch_bounds;
    Ptr<DrawCommand> src_indirect;

    // Six planes per tile.
    Ptr<float4> tile_planes;

    // survivors[tile * instance_capacity + first_instance + slot] = source instance. Order within a batch is
    // arbitrary, which depth rendering does not mind.
    Ptr<uint> survivors;
    Ptr<uint> counts;
    Ptr<DrawCommand> dst_indirect;

    // Batches in the source indirect buffer (the chunk search range) and how many of them the tiles draw.
    uint source_batch_count;
    uint batch_count;
    uint chunk_count;
    uint instance_capacity;

    uint stage;
    uint _padding;
};

[vk::push_constant] ShadowCullPushConstants pc;

void set_task_group_counts(inout DrawCommand command)
{
    const uint chunk_count = (command.meshlet_count + meshlets_per_task - 1) / meshlets_per_task;
    const uint total = command.instance_count * chunk_count;

    command.group_count_x = min(total, max_task_group_count_x);
    command.group_count_y = total == 0 ? 1 : (total + command.group_count_x - 1) / command.group_count_x;
    command.group_count_z = 1;
}

uint chunk_batch(uint chunk)
{
    uint low = 0;
    uint high = pc.source_batch_count - 1;

    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        if (pc.batch_bounds[middle].first_chunk <= chunk) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }

    return low;
}

bool box_in_planes(uint tile, float3 centre, float3 extent)
{
    [unroll]
    for (uint plane_index = 0; plane_index < 6; ++plane_index) {
        const float4 plane = pc.tile_planes[tile * 6 + plane_index];

        if (dot(plane.xyz, centre) + plane.w + dot(abs(plane.xyz), extent) < 0.0) {
            return false;
        }
    }

    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_cs(uint3 group_id: SV_GroupID, uint3 group_thread_id: SV_GroupThreadID)
{
    const uint tile = group_id.z;
    const uint lane = group_thread_id.x;

    if (pc.stage != stage_cull) {
        const uint batch = group_id.x * WG_SIZE + lane;

        if (batch >= pc.batch_count) {
            return;
        }

        const uint slot = tile * pc.batch_count + batch;

        if (pc.stage == stage_clear) {
            pc.counts[slot] = 0;
        } else {
            DrawCommand command = pc.src_indirect[batch];
            command.instance_count = pc.counts[slot];
            set_task_group_counts(command);
            pc.dst_indirect[slot] = command;
        }

        return;
    }

    const uint chunk = group_id.y * cull_dispatch_width + group_id.x;

    if (chunk >= pc.chunk_count) {
        return;
    }

    const uint batch = chunk_batch(chunk);

    if (batch >= pc.batch_count) {
        return;
    }

    const DrawCommand command = pc.src_indirect[batch];
    const GpuCullBounds bounds = pc.batch_bounds[batch];

    const uint local_index = (chunk - bounds.first_chunk) * WG_SIZE + lane;

    // Uniform across the group: the chunk lies wholly past the batch's last instance.
    if ((chunk - bounds.first_chunk) * WG_SIZE >= command.instance_count) {
        return;
    }

    const uint source_index = command.first_instance + local_index;
    bool visible = false;

    if (local_index < command.instance_count) {
        const float3 local_centre = (bounds.bounds_min + bounds.bounds_max) * 0.5;
        const float3 local_extent = (bounds.bounds_max - bounds.bounds_min) * 0.5;
        const float4x4 model = pc.src_transforms[source_index];

        const float3 centre = mul(model, float4(local_centre, 1.0)).xyz;
        float3 extent = abs(mul(model, float4(local_extent.x, 0.0, 0.0, 0.0)).xyz) +
                        abs(mul(model, float4(0.0, local_extent.y, 0.0, 0.0)).xyz) +
                        abs(mul(model, float4(0.0, 0.0, local_extent.z, 0.0)).xyz);
        extent.xz += bounds.wind_padding;

        visible = box_in_planes(tile, centre, extent);
    }

    const uint wave_visible = WaveActiveCountBits(visible);
    uint wave_base = 0;

    if (WaveIsFirstLane() && wave_visible != 0) {
        InterlockedAdd(pc.counts[tile * pc.batch_count + batch], wave_visible, wave_base);
    }

    wave_base = WaveReadLaneFirst(wave_base);

    if (visible) {
        pc.survivors[tile * pc.instance_capacity + command.first_instance + wave_base + WavePrefixCountBits(visible)] =
                source_index;
    }
}
