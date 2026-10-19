module composite;

import bindless;

struct CompositePC {
    uint hdr_texture_index;
    uint bloom_texture_index;
    uint sampler_index;
    float exposure;
    float bloom_intensity;

    uint outline_texture_index;
    float outline_thickness;
    float outline_texel_x;
    float outline_texel_y;
    float outline_r;
    float outline_g;
    float outline_b;

    uint lut_texture_index;
    float lut_size;
    float lut_intensity;
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

// Interleaved gradient noise (Jimenez), remapped to a triangular distribution in [-1, 1].
float dither_noise(float2 pixel)
{
    float a = frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
    float b = frac(52.9829189 * frac(dot(pixel + 17.0, float2(0.06711056, 0.00583715))));
    return a + b - 1.0;
}

float3 linear_to_srgb(float3 c)
{
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

float3 srgb_to_linear(float3 c)
{
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}

// Colour grading LUT stored as a strip: lut_size * lut_size texels wide, lut_size tall, one blue slice after another
// along x with green down y. The hardware filters inside a slice; the two neighbouring blue slices are blended here.
// `encoded` is display-referred (sRGB-encoded) colour.
float3 sample_lut(float3 encoded, uint lut_index, float lut_size, SamplerState lut_sampler)
{
    const var lut = sampled_2d[NonUniformResourceIndex(lut_index)];

    const float3 scaled = saturate(encoded) * (lut_size - 1.0);
    const float slice = scaled.b;
    const float slice_low = floor(slice);
    const float slice_high = min(slice_low + 1.0, lut_size - 1.0);

    // Texel centres: red runs 0.5 .. size - 0.5 inside a slice, so filtering never reaches the neighbouring slice.
    const float strip_width = lut_size * lut_size;
    const float u_in_slice = scaled.r + 0.5;
    const float v = (scaled.g + 0.5) / lut_size;

    const float3 low = lut.Sample(lut_sampler, float2((slice_low * lut_size + u_in_slice) / strip_width, v)).rgb;
    const float3 high = lut.Sample(lut_sampler, float2((slice_high * lut_size + u_in_slice) / strip_width, v)).rgb;

    return lerp(low, high, slice - slice_low);
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

    if (pc.lut_texture_index != 0 && pc.lut_intensity > 0.0) {
        const float3 encoded_in = linear_to_srgb(saturate(mapped));
        const float3 graded = sample_lut(encoded_in, pc.lut_texture_index, pc.lut_size, samplers[sampler_index]);
        mapped = srgb_to_linear(lerp(encoded_in, graded, saturate(pc.lut_intensity)));
    }

    if (pc.outline_texture_index != 0) {
        const var mask = sampled_2d[NonUniformResourceIndex(pc.outline_texture_index)];
        const var mask_sampler = samplers[sampler_index];
        const float2 texel = float2(pc.outline_texel_x, pc.outline_texel_y);

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

    // The swapchain is 8-bit sRGB: dither by one code value in encoded space so smooth gradients don't band.
    float3 encoded = linear_to_srgb(saturate(mapped)) + dither_noise(input.position.xy) / 255.0;
    mapped = srgb_to_linear(saturate(encoded));

    FragmentOutput output;
    output.colour = float4(mapped, 1.0);
    return output;
}