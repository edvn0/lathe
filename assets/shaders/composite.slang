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

    uint hdr_output;
    float hdr_peak_relative;
    float output_scale;
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

// HDR: values up to paper white (1.0) pass through unchanged; above it the brightest channel rolls off exponentially
// towards the display peak, and all channels are scaled together so the hue holds.
float3 hdr_rolloff(float3 colour, float peak_relative)
{
    const float knee = 1.0;
    const float range = max(peak_relative - knee, 1e-3);
    const float brightest = max(colour.r, max(colour.g, colour.b));

    if (brightest <= knee) {
        return colour;
    }

    const float mapped = knee + range * (1.0 - exp(-(brightest - knee) / range));
    return colour * (mapped / brightest);
}

// PCG hash of the pixel coordinate.
uint pcg_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// White noise with a triangular distribution in [-1, 1]. Unlike interleaved gradient noise it has no structure, so
// it doesn't leave faint patterns on very shallow gradients.
float dither_noise(float2 pixel)
{
    const uint2 p = uint2(pixel);
    const uint h = pcg_hash(p.x + pcg_hash(p.y));
    const float a = float(h & 0xFFFFu) * (1.0 / 65536.0);
    const float b = float(h >> 16u) * (1.0 / 65536.0);
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

// Colour grading through a lut_size^3 volume (x = red, y = green, z = blue). `encoded` is display-referred
// (sRGB-encoded) colour. Colour 0 .. 1 maps onto the texel centres, so the ends of the range hit the first and last
// entries exactly and the hardware does the trilinear blend.
float3 sample_lut(float3 encoded, uint lut_index, float lut_size, SamplerState lut_sampler)
{
    const float3 coordinate = saturate(encoded) * ((lut_size - 1.0) / lut_size) + 0.5 / lut_size;

    return sampled_3d[NonUniformResourceIndex(lut_index)].SampleLevel(lut_sampler, coordinate, 0).rgb;
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

    const bool hdr = pc.hdr_output != 0;
    float3 mapped = hdr ? hdr_rolloff(max(combined_hdr * pc.exposure, 0.0), pc.hdr_peak_relative)
                        : aces_fitted(combined_hdr * pc.exposure);

    if (!hdr && pc.lut_texture_index != 0 && pc.lut_intensity > 0.0) {
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

    if (hdr) {
        FragmentOutput hdr_output;
        hdr_output.colour = float4(mapped * pc.output_scale, 1.0);
        return hdr_output;
    }

    // The swapchain is 8-bit sRGB: dither by one code value in encoded space so smooth gradients don't band.
    float3 encoded = linear_to_srgb(saturate(mapped)) + dither_noise(input.position.xy) / 255.0;
    mapped = srgb_to_linear(saturate(encoded));

    FragmentOutput output;
    output.colour = float4(mapped, 1.0);
    return output;
}