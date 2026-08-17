import scene_types;
import bindless;

struct PushConstants {
    Ptr<GpuLight> lights;
    Ptr<UBO> ubo;
    uint light_count;
    uint icon_texture_index;
    uint sampler_index;
    float icon_world_size;
};

[[vk::push_constant]]
PushConstants pc;

struct Payload {
    uint light_index[32];
};

groupshared Payload s_payload;
groupshared uint s_visible_count;

[shader("amplification")]
[numthreads(32, 1, 1)]
void main_task(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
    const uint light_index = group_id.x * 32 + group_thread_id.x;

    if (group_thread_id.x == 0) {
        s_visible_count = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    if (light_index < pc.light_count) {
        uint slot;
        InterlockedAdd(s_visible_count, 1, slot);
        s_payload.light_index[slot] = light_index;
    }

    GroupMemoryBarrierWithGroupSync();

    DispatchMesh(s_visible_count, 1, 1, s_payload);
}

struct QuadVertex {
    float4 clip_position : SV_Position;
    float2 uv : TEXCOORD0;
    float3 tint : COLOR0;
};

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void main_mesh(in payload Payload payload, uint3 group_id : SV_GroupID,
               out vertices QuadVertex verts[4], out indices uint3 tris[2]) {
    SetMeshOutputCounts(4, 2);

    const GpuLight light = pc.lights[payload.light_index[group_id.x]];
    const UBO ubo = pc.ubo[0];

    // Spherical billboard: face directly at the camera regardless of the
    // view matrix's element layout/convention (row- vs column-major
    // indexing was ambiguous and untested -- this sidesteps it entirely).
    const float3 forward = normalize(ubo.camera_position - light.position);

    // Guard against the light sitting on the world-up axis relative to the
    // camera (forward nearly parallel to world_up), where cross() below
    // would degenerate towards zero and the quad would collapse/flip.
    float3 world_up = float3(0.0, 1.0, 0.0);
    if (abs(dot(forward, world_up)) > 0.999) {
        world_up = float3(1.0, 0.0, 0.0);
    }

    const float3 cam_right = normalize(cross(world_up, forward));
    const float3 cam_up = cross(forward, cam_right);

    const float half_size = pc.icon_world_size * 0.5;
    const float3 right_offset = cam_right * half_size;
    const float3 up_offset = cam_up * half_size;

    const float3 corners[4] = {
        light.position - right_offset - up_offset,
        light.position + right_offset - up_offset,
        light.position - right_offset + up_offset,
        light.position + right_offset + up_offset,
    };

    const float2 uvs[4] = { float2(0, 1), float2(1, 1), float2(0, 0), float2(1, 0) };

    const float3 tint = saturate(light.colour);

    [unroll]
    for (uint i = 0; i < 4; ++i) {
        verts[i].clip_position = mul(ubo.view_projection, float4(corners[i], 1.0));
        verts[i].uv = uvs[i];
        verts[i].tint = tint;
    }

    tris[0] = uint3(0, 1, 2);
    tris[1] = uint3(2, 1, 3);
}

[shader("fragment")]
float4 main_fs(QuadVertex input) : SV_Target {
    const float4 sampled = sampled_2d[NonUniformResourceIndex(pc.icon_texture_index)]
                                .Sample(samplers[NonUniformResourceIndex(pc.sampler_index)], input.uv);

    return float4(sampled.rgb * input.tint, sampled.a);
}