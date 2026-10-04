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

[shader("amplification")]
[numthreads(meshlets_per_task, 1, 1)]
void main_task(uint3 group_id: SV_GroupID, uint lane: SV_GroupIndex, uint draw_index: SV_DrawIndex, uniform PC pc)
{
    run_meshlet_task(pc, 0, draw_index, group_id, lane);
}

VertexOutput prepass_vertex(PC pc, GpuDraw draw, float4x4 model, float wind_strength, uint vertex_id)
{
    const var ubo = pc.ubo;
    const var vertex = decode_vertex(draw.vertex_address[vertex_id]);
    var world_position = mul(model, float4(vertex.position, 1.0));

    world_position.xz += wind_offset(vertex.position, world_position.xyz, wind_strength, ubo.time);

    VertexOutput output;
    output.position = mul(ubo.view_projection, world_position);
    output.material_index = draw.material_index;
    output.texture_coordinate = vertex.uvs;
    return output;
}

// Instanced path for batches too small for meshlets. Same vertex function as main_mesh.
[shader("vertex")]
VertexOutput main_vs(uint vertex_id: SV_VulkanVertexID, uint instance_index: SV_VulkanInstanceID, uniform PC pc)
{
    const GpuDraw draw = pc.draws[instance_index];
    const float wind_strength = pc.materials[draw.material_index].wind_strength;
    return prepass_vertex(pc, draw, pc.transforms[instance_index], wind_strength, vertex_id);
}

[shader("mesh")]
[numthreads(mesh_group_size, 1, 1)]
[outputtopology("triangle")]
void main_mesh(in payload MeshletPayload payload, uint3 group_id: SV_GroupID, uint lane: SV_GroupIndex,
               uniform PC pc, out vertices VertexOutput verts[meshlet_max_vertices],
               out indices uint3 tris[meshlet_max_triangles])
{
    const GpuDraw draw = pc.draws[payload.instance_index];
    const Meshlet meshlet = draw.meshlets[payload_meshlet_index(payload, group_id.x)];

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);

    const float4x4 model = pc.transforms[payload.instance_index];
    const float wind_strength = pc.materials[draw.material_index].wind_strength;

    for (uint v = lane; v < meshlet.vertex_count; v += mesh_group_size) {
        verts[v] = prepass_vertex(pc, draw, model, wind_strength, meshlet_vertex_index(draw, meshlet, v));
    }

    for (uint t = lane; t < meshlet.triangle_count; t += mesh_group_size) {
        tris[t] = meshlet_triangle(draw, meshlet, t);
    }
}

// Only for the masked prepass pipelines; the opaque prepass has no fragment shader. Returns the samples the fragment
// covers: all of them, or with alpha to coverage under MSAA the share its alpha asks for, so forward (which tests
// depth EQUAL) shades exactly those.
[shader("fragment")]
uint main_fs(VertexOutput input, uniform PC pc) : SV_Coverage
{
    const var material = pc.materials[input.material_index];

    if (material.alpha_mode != AlphaMode.mask) {
        return 0xFFFFFFFFu;
    }

    const var texture = sampled_2d[NonUniformResourceIndex(material.base_colour_texture)];
    const var sampler = samplers[NonUniformResourceIndex(material.sampler_index)];
    const float alpha = texture.Sample(sampler, input.texture_coordinate).a;

    const uint sample_count = pc.sample_count;

    if ((material.flags & material_flag_alpha_to_coverage) == 0 || sample_count <= 1) {
        if (alpha < material.alpha_cutoff) {
            discard;
        }
        return 0xFFFFFFFFu;
    }

    // Sharpened so the cutoff edge is about a pixel wide whatever the mip level, then turned into that share of
    // the samples (Castano's alpha-to-coverage sharpening).
    const float edge_width = max(fwidth(alpha), 1e-4);
    const float coverage = saturate((alpha - material.alpha_cutoff) / edge_width + 0.5);
    const uint covered = min(uint(coverage * float(sample_count) + 0.5), sample_count);

    if (covered == 0) {
        discard;
    }

    if (covered >= sample_count) {
        return 0xFFFFFFFFu;
    }

    // Which samples a partial pixel keeps rotates with the pixel, so neighbouring edge pixels don't all drop the same
    // samples (that reads as a pattern once resolved).
    const uint2 pixel = uint2(input.position.xy);
    const uint rotation = (pixel.x + pixel.y * 3u) % sample_count;
    const uint all_samples = (1u << sample_count) - 1u;
    const uint kept = (1u << covered) - 1u;

    return ((kept << rotation) | (kept >> (sample_count - rotation))) & all_samples;
}
