import bindless;
import scene_types;
import wind;

struct VertexOutput
{
    float4 position : SV_Position;
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
    return output;
}
