module meshlet_task;

import scene_types;

// Shared task (amplification) stage for every meshlet scene pass --
// forward_geom, depth_prepass and shadow_depth all call run_meshlet_task()
// from their own main_task entry point, so the three passes cull meshlets
// identically. That matters beyond tidiness: the forward pass depth-tests
// EQUAL against the prepass, so a meshlet the prepass skipped but forward
// drew (or vice versa) would show up as holes or z-fighting.
//
// One task workgroup covers up to meshlets_per_task consecutive meshlets of
// ONE instance: SV_GroupID (flattened over X/Y, see DrawCommand) picks the
// (instance, meshlet chunk), each lane tests one meshlet, and the survivors
// are compacted into the payload in lane order (deterministic, unlike an
// atomic append) before a single DispatchMesh.

public struct MeshletPayload
{
    public uint instance_index;
    public uint meshlet_base;
    public uint meshlet_offsets[meshlets_per_task];
};

groupshared MeshletPayload s_payload;
groupshared uint s_visible[meshlets_per_task];

// wind_offset() (wind.slang) moves a vertex by at most wind_strength along
// each of X and Z (sum of sines weighted 0.7/0.3, height weight <= 1), so
// sqrt(2) * wind_strength bounds the displacement length.
static const float wind_radius_scale = 1.41421356;

// Relative spread between the largest and smallest axis scale below which
// a transform counts as uniformly scaled -- cone culling rotates the cone
// axis with the model matrix directly (no inverse-transpose), which is only
// exact for rotation + uniform scale, same assumption forward_geom makes
// for normals.
static const float uniform_scale_tolerance = 1.01;

bool meshlet_visible(PC pc, uint plane_offset, uint instance_index, uint meshlet_index)
{
    const GpuDraw draw = pc.draws[instance_index];
    const Meshlet meshlet = draw.meshlets[meshlet_index];
    const float4x4 model = pc.transforms[instance_index];

    const float3 axis_x = mul(model, float4(1.0, 0.0, 0.0, 0.0)).xyz;
    const float3 axis_y = mul(model, float4(0.0, 1.0, 0.0, 0.0)).xyz;
    const float3 axis_z = mul(model, float4(0.0, 0.0, 1.0, 0.0)).xyz;

    const float3 scale_sq = float3(dot(axis_x, axis_x), dot(axis_y, axis_y), dot(axis_z, axis_z));
    const float max_scale = sqrt(max(scale_sq.x, max(scale_sq.y, scale_sq.z)));
    const float min_scale = sqrt(min(scale_sq.x, min(scale_sq.y, scale_sq.z)));

    const float wind_strength = abs(pc.materials[draw.material_index].wind_strength);

    const float3 centre = mul(model, float4(meshlet.centre, 1.0)).xyz;
    const float radius = meshlet.radius * max_scale + wind_strength * wind_radius_scale;

    if ((pc.cull_flags & cull_frustum_bit) != 0) {
        [unroll]
        for (uint plane_index = 0; plane_index < 6; ++plane_index) {
            const float4 plane = pc.cull_planes[plane_offset + plane_index];

            if (dot(plane.xyz, centre) + plane.w < -radius) {
                return false;
            }
        }
    }

    // Wind-swayed meshlets are skipped: sway bends individual triangles, so
    // the cone baked from the rest pose no longer bounds their normals.
    // Mirrored transforms (negative determinant) flip which side is front.
    if ((pc.cull_flags & cull_backface_bit) != 0 && meshlet.cone_cutoff < 1.0 && wind_strength <= 0.0 &&
        max_scale <= min_scale * uniform_scale_tolerance && dot(cross(axis_x, axis_y), axis_z) > 0.0) {
        const float3 cone_axis = normalize(mul(model, float4(meshlet.cone_axis, 0.0)).xyz);
        const float3 to_centre = centre - pc.ubo.camera_position;

        if (dot(to_centre, cone_axis) >= meshlet.cone_cutoff * length(to_centre) + radius) {
            return false;
        }
    }

    return true;
}

// `plane_offset`: index of this view's first plane in pc.cull_planes (0 for
// the main camera, cascade_index * 6 for a shadow cascade).
public void run_meshlet_task(PC pc, uint plane_offset, uint draw_index, uint3 group_id, uint lane)
{
    const DrawCommand command = pc.task_commands[draw_index];

    const uint chunk_count = (command.meshlet_count + meshlets_per_task - 1) / meshlets_per_task;
    const uint linear_group = group_id.x + group_id.y * command.group_count_x;
    const bool group_valid = chunk_count != 0 && linear_group < command.instance_count * chunk_count;

    const uint instance_index = command.first_instance + (group_valid ? linear_group / chunk_count : 0);
    const uint meshlet_base = group_valid ? (linear_group % chunk_count) * meshlets_per_task : 0;
    const uint meshlet_index = meshlet_base + lane;

    bool visible = group_valid && meshlet_index < command.meshlet_count;

    if (visible) {
        visible = meshlet_visible(pc, plane_offset, instance_index, meshlet_index);
    }

    s_visible[lane] = visible ? 1u : 0u;

    if (lane == 0) {
        s_payload.instance_index = instance_index;
        s_payload.meshlet_base = meshlet_base;
    }

    GroupMemoryBarrierWithGroupSync();

    uint slot = 0;
    uint visible_count = 0;

    for (uint i = 0; i < meshlets_per_task; ++i) {
        if (i < lane) {
            slot += s_visible[i];
        }
        visible_count += s_visible[i];
    }

    if (visible) {
        s_payload.meshlet_offsets[slot] = lane;
    }

    GroupMemoryBarrierWithGroupSync();

    DispatchMesh(visible_count, 1, 1, s_payload);
}

public uint meshlet_vertex_index(GpuDraw draw, Meshlet meshlet, uint local_vertex)
{
    return draw.meshlet_data[meshlet.vertex_offset + local_vertex];
}

public uint3 meshlet_triangle(GpuDraw draw, Meshlet meshlet, uint triangle)
{
    const uint packed = draw.meshlet_data[meshlet.triangle_offset + triangle];
    return uint3(packed & 0xFFu, (packed >> 8u) & 0xFFu, (packed >> 16u) & 0xFFu);
}

public uint payload_meshlet_index(MeshletPayload payload, uint mesh_group)
{
    return payload.meshlet_base + payload.meshlet_offsets[mesh_group];
}
