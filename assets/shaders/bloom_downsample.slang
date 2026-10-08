module bloom_downsample;

import bindless;

struct DownsamplePushConstants
{
    uint src_texture_index;
    uint dst_storage_index;
    uint linear_sampler_index;

    uint is_first_level;

    float2 src_texel_size;
    int2 dst_size;

    float threshold;
    float knee;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] DownsamplePushConstants pc;

float luma(float3 colour)
{
    return dot(colour, float3(0.2126, 0.7152, 0.0722));
}

float3 sanitize(float3 colour)
{
    if (any(isnan(colour)) || any(isinf(colour)))
    {
        return float3(0.0);
    }

    return min(colour, float3(65504.0));
}

float3 fetch(float2 uv)
{
    Texture2D<float4> src = sampled_2d[pc.src_texture_index];
    SamplerState smp = samplers[pc.linear_sampler_index];
    float3 colour = src.SampleLevel(smp, uv, 0).rgb;
    return pc.is_first_level != 0 ? sanitize(colour) : colour;
}

float3 soft_threshold(float3 colour)
{
    float brightness = luma(colour);
    float soft = clamp(brightness - pc.threshold + pc.knee, 0.0, 2.0 * pc.knee);
    soft = (soft * soft) / (4.0 * pc.knee + 1e-4);
    float contribution = max(soft, brightness - pc.threshold) / max(brightness, 1e-4);
    return colour * contribution;
}

float3 downsample_13_tap(float2 uv, float2 texel)
{
    float3 a = fetch(uv + texel * float2(-2.0,  2.0));
    float3 b = fetch(uv + texel * float2( 0.0,  2.0));
    float3 c = fetch(uv + texel * float2( 2.0,  2.0));
    float3 d = fetch(uv + texel * float2(-2.0,  0.0));
    float3 e = fetch(uv);
    float3 f = fetch(uv + texel * float2( 2.0,  0.0));
    float3 g = fetch(uv + texel * float2(-2.0, -2.0));
    float3 h = fetch(uv + texel * float2( 0.0, -2.0));
    float3 i = fetch(uv + texel * float2( 2.0, -2.0));
    float3 j = fetch(uv + texel * float2(-1.0,  1.0));
    float3 k = fetch(uv + texel * float2( 1.0,  1.0));
    float3 l = fetch(uv + texel * float2(-1.0, -1.0));
    float3 m = fetch(uv + texel * float2( 1.0, -1.0));

    float3 box_inner = (j + k + l + m) * 0.25;
    float3 box_top_left = (a + b + d + e) * 0.25;
    float3 box_top_right = (b + c + e + f) * 0.25;
    float3 box_bottom_left = (d + e + g + h) * 0.25;
    float3 box_bottom_right = (e + f + h + i) * 0.25;

    float weight_inner = 0.5;
    float weight_top_left = 0.125;
    float weight_top_right = 0.125;
    float weight_bottom_left = 0.125;
    float weight_bottom_right = 0.125;

    if (pc.is_first_level != 0)
    {
        weight_inner /= 1.0 + luma(box_inner);
        weight_top_left /= 1.0 + luma(box_top_left);
        weight_top_right /= 1.0 + luma(box_top_right);
        weight_bottom_left /= 1.0 + luma(box_bottom_left);
        weight_bottom_right /= 1.0 + luma(box_bottom_right);
    }

    float3 colour = box_inner * weight_inner + box_top_left * weight_top_left + box_top_right * weight_top_right +
                    box_bottom_left * weight_bottom_left + box_bottom_right * weight_bottom_right;
    float weight_sum = weight_inner + weight_top_left + weight_top_right + weight_bottom_left + weight_bottom_right;

    return colour / weight_sum;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID)
{
    int2 coord = int2(thread_id.xy);

    if (coord.x >= pc.dst_size.x || coord.y >= pc.dst_size.y)
    {
        return;
    }

    float2 uv = (float2(coord) + 0.5) / float2(pc.dst_size);
    float3 colour = downsample_13_tap(uv, pc.src_texel_size);

    if (pc.is_first_level != 0 && pc.threshold >= 0.0)
    {
        colour = soft_threshold(colour);
    }

    storage_2d[pc.dst_storage_index][coord] = float4(colour, 1.0);
}
