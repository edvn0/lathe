import scene_types;

// GPU frustum culling: one workgroup per batch (mainCs, one thread per
// SV_VulkanInstanceID is deliberately NOT how this works -- see below). Each
// workgroup reads its own batch's un-culled indirect command to learn
// [firstInstance, firstInstance + instanceCount), tests every instance in
// that range against the camera frustum (CullPC.frustum_planes), and
// compacts survivors back into [firstInstance, firstInstance + survivor
// count) using an in-workgroup shared-memory prefix sum -- so the surviving
// instances of a batch stay contiguous starting at that batch's original
// firstInstance, which is exactly what SV_VulkanInstanceID indexing in
// forward_geom.slang / depth_prepass.slang already expects. No changes are
// needed in those shaders.
//
// This replaces an earlier two-dispatch design (one thread per instance,
// global InterlockedAdd into a per-batch atomic counter, then a second
// dispatch to copy that counter into the indirect command). That design's
// survivor order depended on which thread's atomic add landed first --
// harmless in principle (compaction doesn't need to preserve order), but it
// meant every batch's output order could vary from dispatch to dispatch.
// This version's shared-memory scan is deterministic: given the same
// visibility results, the same instances land in the same slots every time.
// One dispatch, one workgroup per batch, and dispatch size is therefore
// exactly the batch count -- it never depends on total instance count,
// which removes an entire class of workgroup-count-vs-instance-count bugs.
//
// Only the main view (depth prepass + forward) reads the output of this
// pass; the shadow pass keeps drawing from the un-culled buffers, since a
// shadow caster can be outside the camera frustum while still being visible
// in a shadow cascade.

static const uint WG_SIZE = 256;

// Local-space AABB for one batch (mesh+submesh+material), plus wind_padding:
// see wind_offset() in wind.slang. wind_offset displaces a vertex's world
// XZ position by at most wind_strength (its sway amplitude is a sum of
// sines each clamped to +-1 and weighted 0.7/0.3, and its height weight is
// clamped to [0, 1]) -- so growing this batch's world-space AABB by
// wind_padding along X and Z (see transform_aabb below) keeps swaying
// foliage from popping at the frustum edge before it's actually offscreen.
// Mirrors GpuCullBounds in include/renderer.hxx.
public struct GpuCullBounds
{
    public float3 bounds_min;
    public float wind_padding;
    public float3 bounds_max;
    public float pad1;
};

// Mirrors VkDrawIndexedIndirectCommand.
public struct IndirectCommand
{
    public uint indexCount;
    public uint instanceCount;
    public uint firstIndex;
    public int vertexOffset;
    public uint firstInstance;
};

public struct CullPC
{
    public Ptr<GpuDraw> src_draws;
    public Ptr<float4x4> src_transforms;
    public Ptr<GpuCullBounds> batch_bounds;

    // Read-only: each workgroup's own [firstIndex, firstInstance,
    // instanceCount) range to cull.
    public Ptr<IndirectCommand> src_indirect;

    // Written once per batch, by thread 0 only -- no atomics needed, since
    // each batch has exactly one workgroup and thus exactly one writer.
    public Ptr<IndirectCommand> dst_indirect;

    public Ptr<GpuDraw> dst_draws;
    public Ptr<float4x4> dst_transforms;

    // 6 world-space frustum planes (xyz = inward normal, w = distance),
    // order: left, right, bottom, top, near, far. Deliberately its own flat
    // Ptr<float4> array rather than a field on the shared UBO struct -- see
    // the comment on Renderer::RendererFrame::frustum_planes_buffer.
    public Ptr<float4> frustum_planes;

    public uint batch_count;
    public uint padding;
};

// Transforms a local-space AABB (given as center/extent) into a world-space
// center/extent pair. Each local axis, scaled by its extent, is pushed
// through the same mul(model, vec) used everywhere else in this codebase
// for vertex positions (see forward_geom.slang) -- summing the absolute
// values of the transformed axes gives the tightest axis-aligned box
// around the transformed original box, without assuming anything about
// how float4x4 rows/columns map to mul()'s convention.
void transform_aabb(float4x4 model, float3 local_centre, float3 local_extent, float wind_padding,
                    out float3 world_centre, out float3 world_extent)
{
    world_centre = mul(model, float4(local_centre, 1.0)).xyz;

    float3 axis_x = mul(model, float4(local_extent.x, 0.0, 0.0, 0.0)).xyz;
    float3 axis_y = mul(model, float4(0.0, local_extent.y, 0.0, 0.0)).xyz;
    float3 axis_z = mul(model, float4(0.0, 0.0, local_extent.z, 0.0)).xyz;

    world_extent = abs(axis_x) + abs(axis_y) + abs(axis_z);

    // wind_offset() (wind.slang) displaces world-space X/Z only.
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


// Separate the read phase and write phase into distinct arrays
groupshared uint s_wave_totals[WG_SIZE];
groupshared uint s_wave_offsets[WG_SIZE];
groupshared uint s_chunk_total;

[shader("compute")]
[numthreads(256, 1, 1)]
void mainCs(uint3 group_id: SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform CullPC pc)
{
    uint batch = group_id.x;

    if (batch >= pc.batch_count) {
        return;
    }

    IndirectCommand source_command = pc.src_indirect[batch];
    GpuCullBounds bounds = pc.batch_bounds[batch];
    uint lane = group_thread_id.x;

    uint running_survivors = 0;

    for (uint base = 0; base < source_command.instanceCount; base += WG_SIZE) {
        uint local_index = base + lane;
        bool visible = false;

        if (local_index < source_command.instanceCount) {
            visible = instance_visible(pc, bounds, source_command.firstInstance + local_index);
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
            uint source_index = source_command.firstInstance + local_index;
            uint destination_index = source_command.firstInstance + running_survivors + exclusive_scan;

            pc.dst_draws[destination_index] = pc.src_draws[source_index];
            pc.dst_transforms[destination_index] = pc.src_transforms[source_index];
        }

        running_survivors += chunk_survivor_count;

        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0) {
        IndirectCommand out_command = source_command;
        out_command.instanceCount = running_survivors;
        pc.dst_indirect[batch] = out_command;
    }
}