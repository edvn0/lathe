module bloom_downsample;

import bindless;

// One level of the bloom mip chain: reads the level above through a
// sampled view and writes this level through a storage view. The two are
// always different subresources, and render_pass::bloom() puts a barrier
// between consecutive levels, so no invocation ever reads a texel another
// invocation of the same dispatch is writing.
//
// The filter is the 13-tap downsample from Jimenez, "Next Generation Post
// Processing in Call of Duty: Advanced Warfare" (SIGGRAPH 2014): five
// overlapping 2x2 box footprints around the destination texel, weighted
// 0.5 (inner) and 4 x 0.125 (corners). The first level additionally uses a
// Karis average over those five boxes -- each weighted by 1 / (1 + luma) --
// so single sub-pixel highlights (specular sparkles on the helmet, say)
// can't dominate a whole bloom texel and pop on and off as the camera
// moves, and applies the soft-knee brightness threshold.
struct DownsamplePushConstants
{
    uint src_texture_index;
    uint dst_storage_index;
    uint linear_sampler_index;

    // Non-zero for the HDR -> mip 0 dispatch only.
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

// The HDR target can hold the odd NaN/inf (degenerate normals, a zero
// roughness specular lobe). Without this one bad texel would be smeared
// across every bloom level and show up as a flashing black or white block.
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
    // a . b . c
    // . j . k .
    // d . e . f
    // . l . m .
    // g . h . i
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

    // The five overlapping 2x2 boxes. Written out rather than as a local
    // array walked by an [unroll]ed loop: that form crashes SPIRV-Tools'
    // merge-blocks pass (vulkan-sdk-1.4.357.0) in SlangCompiler's
    // performance-pass run.
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
void mainCs(uint3 thread_id : SV_DispatchThreadID)
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
