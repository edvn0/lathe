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

groupshared uint s_scan[WG_SIZE];
groupshared uint s_running;

[shader("compute")]
[numthreads(256, 1, 1)]
void mainCs(uint3 group_id: SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform CullPC pc)
{
    uint batch = group_id.x;

    // Workgroup-uniform early-out (every thread in the workgroup takes the
    // same branch, since it depends only on group_id, not lane index) --
    // safe to return here, before any GroupMemoryBarrierWithGroupSync below.
    // A per-thread early-out on a per-lane condition would instead leave
    // some lanes never reaching the barrier, deadlocking the workgroup.
    if (batch >= pc.batch_count) {
        return;
    }

    IndirectCommand source_command = pc.src_indirect[batch];
    GpuCullBounds bounds = pc.batch_bounds[batch];
    uint lane = group_thread_id.x;

    if (lane == 0) {
        s_running = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    for (uint base = 0; base < source_command.instanceCount; base += WG_SIZE) {
        uint local_index = base + lane;
        bool visible = false;

        if (local_index < source_command.instanceCount) {
            visible = instance_visible(pc, bounds, source_command.firstInstance + local_index);
        }

        // Hillis-Steele inclusive scan over this chunk's visibility flags.
        // Each step reads a neighbour lane's current running total into a
        // local before writing -- reading s_scan[lane - offset] while other
        // lanes are simultaneously writing s_scan[lane] needs the read and
        // the write on opposite sides of a barrier, hence two syncs per step
        // rather than one.
        s_scan[lane] = visible ? 1u : 0u;

        GroupMemoryBarrierWithGroupSync();

        for (uint offset = 1; offset < WG_SIZE; offset <<= 1) {
            uint addend = (lane >= offset) ? s_scan[lane - offset] : 0u;

            GroupMemoryBarrierWithGroupSync();

            s_scan[lane] += addend;

            GroupMemoryBarrierWithGroupSync();
        }

        uint exclusive_scan = s_scan[lane] - (visible ? 1u : 0u);
        uint chunk_survivor_count = s_scan[WG_SIZE - 1];

        if (visible) {
            uint source_index = source_command.firstInstance + local_index;
            uint destination_index = source_command.firstInstance + s_running + exclusive_scan;

            // transform_index is stale in the compacted copy (it still
            // points at this instance's original slot in src_transforms,
            // not destination_index) -- harmless, since every consumer of
            // dst_draws (forward_geom.slang, depth_prepass.slang) indexes
            // transforms by SV_VulkanInstanceID, never by
            // GpuDraw::transform_index.
            pc.dst_draws[destination_index] = pc.src_draws[source_index];
            pc.dst_transforms[destination_index] = pc.src_transforms[source_index];
        }

        GroupMemoryBarrierWithGroupSync();

        if (lane == 0) {
            s_running += chunk_survivor_count;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0) {
        IndirectCommand out_command = source_command;

        out_command.instanceCount = s_running;

        pc.dst_indirect[batch] = out_command;
    }
}
