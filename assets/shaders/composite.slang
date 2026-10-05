module composite;

import bindless;

struct CompositePC {
    uint hdr_texture_index;
    uint bloom_texture_index;
    uint sampler_index;
    float exposure;
    float bloom_intensity;

    // The outline mask (R8, 1 where an outlined object drew), or 0 for no outline this frame.
    uint outline_texture_index;
    float outline_thickness;
    float outline_texel_x;
    float outline_texel_y;
    float outline_r;
    float outline_g;
    float outline_b;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput main_vs(uint vertex_id: SV_VertexID)
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
FragmentOutput main_fs(VertexOutput input, uniform CompositePC pc)
{
    uint hdr_index = NonUniformResourceIndex(pc.hdr_texture_index);
    uint bloom_index = NonUniformResourceIndex(pc.bloom_texture_index);
    uint sampler_index = NonUniformResourceIndex(pc.sampler_index);

    float3 hdr_colour = sampled_2d[hdr_index].Sample(samplers[sampler_index], input.uv).rgb;
    float3 bloom_colour = sampled_2d[bloom_index].Sample(samplers[sampler_index], input.uv).rgb;

    float3 combined_hdr = hdr_colour + bloom_colour * pc.bloom_intensity;

    float3 mapped = aces_fitted(combined_hdr * pc.exposure);

    if (pc.outline_texture_index != 0) {
        const var mask = sampled_2d[NonUniformResourceIndex(pc.outline_texture_index)];
        const var mask_sampler = samplers[sampler_index];
        const float2 texel = float2(pc.outline_texel_x, pc.outline_texel_y);

        // The outline is the ring just outside the mask: strongest where a pixel off the mask has mask pixels within
        // the thickness around it. Eight directions at the full radius and eight at half of it keep thin gaps closed.
        float nearby = 0.0;
        [unroll]
        for (int i = 0; i < 8; ++i) {
            const float angle = float(i) * 0.78539816;
            const float2 direction = float2(cos(angle), sin(angle)) * texel;
            nearby = max(nearby, mask.Sample(mask_sampler, input.uv + direction * pc.outline_thickness).r);
            nearby = max(nearby, mask.Sample(mask_sampler, input.uv + direction * (pc.outline_thickness * 0.5)).r);
        }

        const float inside = mask.Sample(mask_sampler, input.uv).r;
        const float edge = saturate(nearby - inside);

        mapped = lerp(mapped, float3(pc.outline_r, pc.outline_g, pc.outline_b), edge);
    }

    FragmentOutput output;
    output.colour = float4(mapped, 1.0);
    return output;
}