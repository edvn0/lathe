import bindless;
import scene_types;
import wind;
import meshlet_task;

struct VertexOutput
{
    float4 position : SV_Position;

    nointerpolation uint material_index;
    float2 texture_coordinate;
};

// Culls against the cascade's own frustum (cull_planes is the cascade
// plane block, see ShadowPassInfo) -- casters outside the camera frustum
// still get drawn when they're inside the cascade.
[shader("amplification")]
[numthreads(meshlets_per_task, 1, 1)]
void main_task(uint3 group_id: SV_GroupID, uint lane: SV_GroupIndex, uint draw_index: SV_DrawIndex,
               uniform ShadowPC pc)
{
    run_meshlet_task(pc.base, pc.cascade_index * 6, draw_index, group_id, lane);
}

VertexOutput shadow_vertex(ShadowPC pc, GpuDraw draw, float4x4 model, float wind_strength, uint vertex_id)
{
    const var ubo = pc.base.ubo;
    const var vertex = decode_vertex(draw.vertex_address[vertex_id]);
    var world_position = mul(model, float4(vertex.position, 1.0));

    world_position.xz += wind_offset(vertex.position, world_position.xyz, wind_strength, ubo.time);

    VertexOutput output;
    output.position = mul(ubo.cascade_view_projection[pc.cascade_index], world_position);
    output.material_index = draw.material_index;
    output.texture_coordinate = vertex.uvs;
    return output;
}

[shader("mesh")]
[numthreads(mesh_group_size, 1, 1)]
[outputtopology("triangle")]
void main_mesh(in payload MeshletPayload payload, uint3 group_id: SV_GroupID, uint lane: SV_GroupIndex,
               uniform ShadowPC pc, out vertices VertexOutput verts[meshlet_max_vertices],
               out indices uint3 tris[meshlet_max_triangles])
{
    const GpuDraw draw = pc.base.draws[payload.instance_index];
    const Meshlet meshlet = draw.meshlets[payload_meshlet_index(payload, group_id.x)];

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);

    const float4x4 model = pc.base.transforms[payload.instance_index];
    const float wind_strength = pc.base.materials[draw.material_index].wind_strength;

    for (uint v = lane; v < meshlet.vertex_count; v += mesh_group_size) {
        verts[v] = shadow_vertex(pc, draw, model, wind_strength, meshlet_vertex_index(draw, meshlet, v));
    }

    for (uint t = lane; t < meshlet.triangle_count; t += mesh_group_size) {
        tris[t] = meshlet_triangle(draw, meshlet, t);
    }
}

// Only used by shadow_mask_pipeline_ -- the opaque shadow PSO stays
// vertex-only (no mainFs) to preserve early-Z / cheap depth-only rendering.
[shader("fragment")]
void mainFs(VertexOutput input, uniform ShadowPC pc)
{
    const var material = pc.base.materials[input.material_index];

    if (material.alpha_mode == AlphaMode.mask) {
        const var texture = sampled_2d[NonUniformResourceIndex(material.base_colour_texture)];
        const var sampler = samplers[NonUniformResourceIndex(material.sampler_index)];
        float alpha = texture.Sample(sampler, input.texture_coordinate).a;

        if (alpha < material.alpha_cutoff) {
            discard;
        }
    }
}
