module composite;

import bindless;

struct CompositePC
{
    uint hdr_texture_index;
    uint sampler_index;
    float exposure;
    uint padding;
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
    uint texture_index =
        NonUniformResourceIndex(pc.hdr_texture_index);

    uint sampler_index =
        NonUniformResourceIndex(pc.sampler_index);

    float3 hdr_colour =
        sampled_2d[texture_index]
            .Sample(
                get_sampler(sampler_index),
                input.uv
            )
            .rgb;

    float3 mapped =
        aces_fitted(hdr_colour * pc.exposure);

    FragmentOutput output;
    output.colour = float4(mapped, 1.0);

    return output;
}