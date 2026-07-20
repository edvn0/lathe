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

    float3 world_position;
    float3 world_normal;
    float4 world_tangent; // xyz = tangent, w = handedness sign
    float2 texture_coordinate;

    nointerpolation uint material_index;
};

[shader("vertex")]
VertexOutput mainVs(
uint vertex_id: SV_VulkanVertexID,
                 uint draw_id: SV_DrawIndex,
                 uint inst_base: SV_VulkanInstanceID,
    uniform PC pc)
{
    GpuDraw draw = pc.draws[inst_base];
    const var ubo = pc.ubo;

    Vertex vertex = draw.vertex_address[vertex_id];

    float4x4 model = pc.transforms[inst_base];

    float4 world_position = mul(
        model,
        float4(vertex.position, 1.0)
    );

    // This is correct for uniform scale and rotation.
    // Use inverse-transpose when non-uniform scaling is supported.
    float3x3 normal_matrix = (float3x3)model;

    float3 world_normal = normalize(
        mul(
            normal_matrix,
            vertex.normal
        )
    );

    float3 world_tangent = normalize(
        mul(
            normal_matrix,
            vertex.tangent.xyz
        )
    );

    VertexOutput output;
    output.position = mul(ubo.view_projection, world_position);
    output.world_position = world_position.xyz;
    output.world_normal = world_normal;
    output.world_tangent = float4(world_tangent, vertex.tangent.w);
    output.texture_coordinate = vertex.uvs;
    output.material_index = draw.material_index;

    return output;
}

struct OutputFS
{
    float4 colour : SV_Target;
};

[shader("fragment")]
OutputFS mainFs(VertexOutput input, uniform PC pc)
{
    float3 normal = normalize(input.world_normal);
    float3 tangent = normalize(input.world_tangent.xyz);

    // Re-orthogonalize in case interpolation skewed the basis.
    tangent = normalize(tangent - normal * dot(normal, tangent));
    float3 bitangent = cross(normal, tangent) * input.world_tangent.w;

    float3x3 tbn = float3x3(tangent, bitangent, normal);

    // Temporary directional lighting until material/image access is connected.
    float3 light_direction = normalize(
        float3(0.4, 0.8, 0.25)
    );

    const var material = pc.materials[input.material_index];
    const var texture = sampled_2d[NonUniformResourceIndex(material.base_colour_texture)];
    const var emissive_texture = sampled_2d[NonUniformResourceIndex(material.emissive_texture)];
    const var ambient_occlusion_texture = sampled_2d[NonUniformResourceIndex(material.occlusion_texture)];
    const var normal_texture = sampled_2d[NonUniformResourceIndex(material.normal_texture)];
    const var sampler = samplers[NonUniformResourceIndex(material.sampler_index)];

    float2 uv = float2(input.texture_coordinate.x, input.texture_coordinate.y);

    float3 tangent_normal = normal_texture.Sample(sampler, uv).xyz * 2.0 - 1.0;
    tangent_normal.xy *= material.normal_scale;
    tangent_normal = normalize(tangent_normal);

    float3 shading_normal = normalize(mul(tangent_normal, tbn));

    float ndotl = saturate(dot(shading_normal, light_direction));


    float ambient_occlusion = ambient_occlusion_texture.Sample(sampler, uv).r;
ambient_occlusion = lerp(1.0, ambient_occlusion, material.occlusion_strength);

var base_colour = texture.Sample(sampler, uv).xyz;

float3 ambient = base_colour * ambient_occlusion;
    float3 diffuse = base_colour * ndotl;
    float3 emissive = emissive_texture.Sample(sampler, uv).xyz *
                  float3(material.emissive[0], material.emissive[1], material.emissive[2]) *
                  material.emissive_strength;

    float3 shaded_colour = ambient + diffuse + emissive;

    const var ubo = pc.ubo;
    float distance_to_camera = length(ubo.camera_position - input.world_position);

    float transmittance = exp(-distance_to_camera * ubo.fog_extinction);
    float3 scattered = ubo.fog_colour * ubo.fog_inscattering * (1.0 - transmittance);

    float3 fogged_colour = shaded_colour * transmittance + scattered;

    OutputFS output;
    output.colour = float4(fogged_colour, 1.0);

    return output;
}