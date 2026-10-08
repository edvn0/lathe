module meshlet_task;

import scene_types;
import hiz_occlusion;

public struct MeshletPayload
{
    public uint instance_index;
    public uint meshlet_base;
    public uint meshlet_offsets[meshlets_per_task];
};

groupshared MeshletPayload s_payload;
groupshared uint s_visible[meshlets_per_task];
groupshared uint s_occluded[meshlets_per_task];

static const float wind_radius_scale = 1.41421356;

static const float uniform_scale_tolerance = 1.01;

bool meshlet_visible(PC pc, uint plane_offset, uint instance_index, uint meshlet_index, out float3 centre,
                     out float radius)
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

    centre = mul(model, float4(meshlet.centre, 1.0)).xyz;
    radius = meshlet.radius * max_scale + wind_strength * wind_radius_scale;

    if ((pc.cull_flags & cull_frustum_bit) != 0) {
        [unroll]
        for (uint plane_index = 0; plane_index < 6; ++plane_index) {
            const float4 plane = pc.cull_planes[plane_offset + plane_index];

            if (dot(plane.xyz, centre) + plane.w < -radius) {
                return false;
            }
        }
    }

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
    bool occluded = false;

    const uint flags = pc.cull_flags;
    const uint bitset_flags = cull_replay_bit | cull_skip_recorded_bit | cull_record_bit;

    Ptr<uint> bits = Ptr<uint>(nullptr);
    uint bit_word = 0;
    uint bit_mask = 0;

    if (visible && (flags & bitset_flags) != 0) {
        const uint bit = pc.draws[instance_index].meshlet_visibility_offset + meshlet_index;
        bits = pc.occlusion[0].meshlet_visibility;
        bit_word = bit >> 5u;
        bit_mask = 1u << (bit & 31u);
    }

    if (visible && (flags & cull_replay_bit) != 0) {
        visible = (bits[bit_word] & bit_mask) != 0;
    } else if (visible) {
        if ((flags & cull_skip_recorded_bit) != 0 && (bits[bit_word] & bit_mask) != 0) {
            visible = false;
        }

        float3 centre = float3(0.0);
        float radius = 0.0;

        if (visible) {
            visible = meshlet_visible(pc, plane_offset, instance_index, meshlet_index, centre, radius);
        }

        if (visible && (flags & cull_occlusion_bit) != 0 && sphere_occluded(pc.occlusion[0], centre, radius)) {
            visible = false;
            occluded = true;
        }

        if (visible && (flags & cull_record_bit) != 0) {
            InterlockedOr(bits[bit_word], bit_mask);
        }
    }

    s_visible[lane] = visible ? 1u : 0u;
    s_occluded[lane] = occluded ? 1u : 0u;

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

    if ((flags & cull_stats_bit) != 0 && lane == 0) {
        uint occluded_count = 0;

        for (uint i = 0; i < meshlets_per_task; ++i) {
            occluded_count += s_occluded[i];
        }

        if (occluded_count != 0) {
            InterlockedAdd(pc.occlusion[0].stats[0], occluded_count);
        }
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
