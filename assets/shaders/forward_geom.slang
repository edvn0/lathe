// assets/shaders/forward_geom.slang
import bindless;

/**
struct ModelVertex {
    glm::vec3 position{};
    glm::vec3 normal{};
    glm::vec4 tangent{};
    glm::vec2 texcoord{};
};
*/

struct Vertex
{
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uvs;
};

struct GpuDraw
{
    // Matches VkDeviceAddress on the C++ side.
    Ptr<Vertex> vertex_address;

    uint material_index;
    uint transform_index;
};

struct PC
{
    Ptr<GpuDraw> draws;
    Ptr<float4x4> transforms;

    float4x4 view_projection;
};

struct VertexOutput
{
    float4 position : SV_Position;

    float3 world_position;
    float3 world_normal;
    float2 texture_coordinate;

    nointerpolation uint material_index;
};

[shader("vertex")]
VertexOutput mainVs(
    uint vertex_id : SV_VertexID,
    uint draw_id : SV_InstanceID,
    uniform PC pc)
{
    GpuDraw draw = pc.draws[draw_id];

    Vertex vertex = draw.vertex_address[vertex_id];

    float4x4 model = pc.transforms[draw.transform_index];

    float4 world_position = mul(
        model,
        float4(vertex.position, 1.0)
    );

    // This is correct for uniform scale and rotation.
    // Use inverse-transpose when non-uniform scaling is supported.
    float3 world_normal = normalize(
        mul(
            (float3x3)model,
            vertex.normal
        )
    );

    VertexOutput output;
    output.position = mul(pc.view_projection, world_position);
    output.world_position = world_position.xyz;
    output.world_normal = world_normal;
    output.texture_coordinate = vertex.uvs;
    output.material_index = draw.material_index;

    return output;
}

struct OutputFS
{
    float4 colour : SV_Target;
};

[shader("fragment")]
OutputFS mainFs(VertexOutput input)
{
    float3 normal = normalize(input.world_normal);

    // Temporary directional lighting until material/image access is connected.
    float3 light_direction = normalize(
        float3(0.4, 0.8, 0.25)
    );

    float ndotl = saturate(dot(normal, light_direction));

    float3 base_colour = float3(
        0.72,
        0.76,
        0.82
    );

    float debug_tint =
    float(input.material_index & 1) * 0.05;



    float3 ambient = base_colour * 0.08;
    float3 diffuse = base_colour * ndotl;

    /* OutputFS output;
output.colour = float4(
    ambient + diffuse + debug_tint,
    1.0
); */
OutputFS output;
output.colour = float4(1.0, 0.0, 0.0, 1.0);

    return output;
}