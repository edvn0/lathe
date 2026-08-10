import bindless;
import scene_types;
import wind;

struct VertexOutput
{
    float4 position : SV_Position;

    nointerpolation uint material_index;
    float2 texture_coordinate;
};

[shader("vertex")]
VertexOutput mainVs(
uint vertex_id: SV_VulkanVertexID,
                 uint draw_id: SV_DrawIndex,
                 uint inst_base: SV_VulkanInstanceID,
    uniform PC pc)
{
    const var draw = pc.draws[inst_base];
    const var ubo = pc.ubo;
    const var vertex = draw.vertex_address[vertex_id];
    const var model = pc.transforms[inst_base];
    var world_position = mul(
        model,
        float4(vertex.position, 1.0)
    );

    const var wind_material = pc.materials[draw.material_index];
    world_position.xz += wind_offset(vertex.position, world_position.xyz, wind_material.wind_strength, ubo.time);

    VertexOutput output;
    output.position = mul(ubo.view_projection, world_position);
    output.material_index = draw.material_index;
    output.texture_coordinate = vertex.uvs;
    return output;
}

// Only used by depth_prepass_mask_pipeline_ -- the opaque prepass PSO stays
// vertex-only (no mainFs) to preserve early-Z / cheap depth-only rendering.
[shader("fragment")]
void mainFs(VertexOutput input, uniform PC pc)
{
    const var material = pc.materials[input.material_index];

    if (material.alpha_mode == AlphaMode.mask) {
        const var texture = sampled_2d[NonUniformResourceIndex(material.base_colour_texture)];
        const var sampler = samplers[NonUniformResourceIndex(material.sampler_index)];
        float alpha = texture.Sample(sampler, input.texture_coordinate).a;

        if (alpha < material.alpha_cutoff) {
            discard;
        }
    }
}
