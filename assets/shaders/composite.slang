module composite;

import bindless;

struct CompositePC {
    uint hdr_texture_index;
    uint bloom_texture_index;
    uint sampler_index;
    float exposure;
    float bloom_intensity;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput mainVs(uint vertex_id: SV_VertexID)
{
    VertexOutput output;

    float2 position;
    float2 uv;

    switch (vertex_id)
    {
    case 0:
        position = float2(-1.0, -1.0);
        uv = float2(0.0, 0.0);
        break;

    case 1:
        position = float2(-1.0, 3.0);
        uv = float2(0.0, 2.0);
        break;

    default:
        position = float2(3.0, -1.0);
        uv = float2(2.0, 0.0);
        break;
    }

    output.position = float4(position, 0.0, 1.0);
    output.uv = uv;

    return output;
}

float3 aces_fitted(float3 colour)
{
    constexpr float a = 2.51;
    constexpr float b = 0.03;
    constexpr float c = 2.43;
    constexpr float d = 0.59;
    constexpr float e = 0.14;

    return saturate(
        (colour * (a * colour + b)) /
        (colour * (c * colour + d) + e)
    );
}

struct FragmentOutput
{
    float4 colour : SV_Target0;
};

[shader("fragment")]
FragmentOutput mainFs(VertexOutput input, uniform CompositePC pc)
{
    uint hdr_index = NonUniformResourceIndex(pc.hdr_texture_index);
    uint bloom_index = NonUniformResourceIndex(pc.bloom_texture_index);
    uint sampler_index = NonUniformResourceIndex(pc.sampler_index);

    // 1. Sample main HDR texture and Bloom texture via bindless arrays
    float3 hdr_colour = sampled_2d[hdr_index].Sample(samplers[sampler_index], input.uv).rgb;
    float3 bloom_colour = sampled_2d[bloom_index].Sample(samplers[sampler_index], input.uv).rgb;

    // 2. Add weighted bloom to the raw scene color in linear space
    float3 combined_hdr = hdr_colour + (bloom_colour * 30*pc.bloom_intensity);

    // 3. Apply exposure and tonemapping to the combined linear color
    float3 mapped = aces_fitted(combined_hdr * pc.exposure);

    FragmentOutput output;
    output.colour = float4(mapped, 1.0);
    return output;
}