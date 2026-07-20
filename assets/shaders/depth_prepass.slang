import bindless;

struct Vertex
{
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uvs;
};

struct GpuDraw
{
    Ptr<Vertex> vertex_address;

    uint material_index;
    uint transform_index;
};

enum class AlphaMode : uint {
    opaque,
    mask,
    blend,
};

struct Material {
    float colour[4];
    float emissive[3];
    float emissive_strength = 0.0F;
    float metallic_factor = 1.0F;
    float roughness_factor = 1.0F;
    float normal_scale = 1.0F;
    float occlusion_strength = 1.0F;

    uint base_colour_texture = 0;
    uint normal_texture = 2;
    uint metallic_roughness_texture = 3;
    uint occlusion_texture = 4;

    uint emissive_texture = 5;
    uint sampler_index = 0;

    AlphaMode alpha_mode = AlphaMode.opaque;
    float alpha_cutoff = 0.5F;
};

struct UBO {
    float4x4 view_projection;
    float4x4 view;
    float4x4 projection;

    float3 camera_position;
    float3 fog_colour;
    float fog_extinction;     // A
    float fog_inscattering;   // B
};

struct PC
{
    Ptr<GpuDraw> draws;
    Ptr<float4x4> transforms;
    Ptr<Material> materials;
    Ptr<UBO> ubo;
};

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
    const var world_position = mul(
        model,
        float4(vertex.position, 1.0)
    );

    VertexOutput output;
    output.position = mul(ubo.view_projection, world_position);
    return output;
}
