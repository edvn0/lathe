import scene_types;

// GPU frustum culling: one thread per instance. Tests each instance's
// world-space AABB against the camera frustum (CullPC.frustum_planes) and, for
// survivors, atomically appends into the culled indirect command's own
// [firstInstance, firstInstance + instanceCount) run -- so the surviving
// instances of a batch stay contiguous starting at that batch's original
// firstInstance, which is exactly what SV_VulkanInstanceID indexing in
// forward_geom.slang / depth_prepass.slang already expects. No changes are
// needed in those shaders.
//
// Only the main view (depth prepass + forward) reads the output of this
// pass; the shadow pass keeps drawing from the un-culled buffers, since a
// shadow caster can be outside the camera frustum while still being visible
// in a shadow cascade.

public struct GpuAabb
{
    public float3 bounds_min;
    public float pad0;
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
    public Ptr<uint> cull_batch;
    public Ptr<GpuAabb> batch_bounds;

    // Read-only here (firstInstance only) -- instanceCount is filled in by
    // mainFinalizeCs below, not by this pass. See batch_visible_count for
    // why the atomic slot counter lives in its own flat buffer instead of
    // this struct.
    public Ptr<IndirectCommand> culled_indirect;

    public Ptr<GpuDraw> dst_draws;
    public Ptr<float4x4> dst_transforms;

    // 6 world-space frustum planes (xyz = inward normal, w = distance),
    // order: left, right, bottom, top, near, far. Deliberately its own flat
    // Ptr<float4> array rather than a field on the shared UBO struct -- see
    // the comment on Renderer::RendererFrame::frustum_planes_buffer.
    public Ptr<float4> frustum_planes;

    // Per-batch atomic survivor counter, one uint per batch. Deliberately a
    // flat Ptr<uint> array -- rather than atomically incrementing
    // culled_indirect[batch_index].instanceCount directly -- so the atomic
    // target is the simplest possible case (a scalar through a pointer),
    // not a struct member reached through an indexed pointer. mainCs only
    // ever increments this; mainFinalizeCs (one thread per batch, no
    // atomics needed since each batch has exactly one writer) copies each
    // batch's final count into its indirect command afterwards.
    public Ptr<uint> batch_visible_count;

    public uint instance_count;
    public uint padding;
};

public struct FinalizePC
{
    public Ptr<uint> batch_visible_count;
    public Ptr<IndirectCommand> culled_indirect;
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
void transform_aabb(float4x4 model, float3 local_centre, float3 local_extent, out float3 world_centre,
                    out float3 world_extent)
{
    world_centre = mul(model, float4(local_centre, 1.0)).xyz;

    float3 axis_x = mul(model, float4(local_extent.x, 0.0, 0.0, 0.0)).xyz;
    float3 axis_y = mul(model, float4(0.0, local_extent.y, 0.0, 0.0)).xyz;
    float3 axis_z = mul(model, float4(0.0, 0.0, local_extent.z, 0.0)).xyz;

    world_extent = abs(axis_x) + abs(axis_y) + abs(axis_z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mainCs(uint3 dispatch_thread_id: SV_DispatchThreadID, uniform CullPC pc)
{
    uint i = dispatch_thread_id.x;

    if (i >= pc.instance_count) {
        return;
    }

    uint batch_index = pc.cull_batch[i];
    GpuAabb aabb = pc.batch_bounds[batch_index];

    float3 local_centre = (aabb.bounds_min + aabb.bounds_max) * 0.5;
    float3 local_extent = (aabb.bounds_max - aabb.bounds_min) * 0.5;

    float4x4 model = pc.src_transforms[i];

    float3 world_centre;
    float3 world_extent;
    transform_aabb(model, local_centre, local_extent, world_centre, world_extent);

    bool visible = true;

    [unroll]
    for (uint plane_index = 0; plane_index < 6; ++plane_index) {
        float4 plane = pc.frustum_planes[plane_index];

        float distance = dot(plane.xyz, world_centre) + plane.w;
        float radius = dot(abs(plane.xyz), world_extent);

        if (distance + radius < 0.0) {
            visible = false;
            break;
        }
    }

    if (!visible) {
        return;
    }

    uint slot;
    InterlockedAdd(pc.batch_visible_count[batch_index], 1, slot);

    uint destination = pc.culled_indirect[batch_index].firstInstance + slot;

    pc.dst_draws[destination] = pc.src_draws[i];
    pc.dst_transforms[destination] = model;
}

// One thread per batch: copies each batch's final survivor count (written
// by mainCs above, now complete thanks to the barrier between the two
// dispatches) into its indirect command's instanceCount. Exactly one
// thread ever writes each culled_indirect[b] here, so this needs no
// atomics at all.
[shader("compute")]
[numthreads(64, 1, 1)]
void mainFinalizeCs(uint3 dispatch_thread_id: SV_DispatchThreadID, uniform FinalizePC pc)
{
    uint b = dispatch_thread_id.x;

    if (b >= pc.batch_count) {
        return;
    }

    pc.culled_indirect[b].instanceCount = pc.batch_visible_count[b];
}
