import bindless;
import scene_types;

struct VertexOutput
{
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput mainVs(
uint vertex_id: SV_VulkanVertexID,
                 uint draw_id: SV_DrawIndex,
                 uint inst_base: SV_VulkanInstanceID,
    uniform ShadowPC pc)
{
    const var draw = pc.base.draws[inst_base];
    const var ubo = pc.base.ubo;
    const var vertex = draw.vertex_address[vertex_id];
    const var model = pc.base.transforms[inst_base];
    const var world_position = mul(
        model,
        float4(vertex.position, 1.0)
    );

    VertexOutput output;
    output.position = mul(ubo.cascade_view_projection[pc.cascade_index], world_position);
    return output;
}
