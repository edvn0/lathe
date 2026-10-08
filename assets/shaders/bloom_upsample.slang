module bloom_upsample;

import bindless;

struct UpsamplePushConstants
{
    uint lower_texture_index;
    uint target_storage_index;
    uint linear_sampler_index;

    float2 lower_texel_size;
    int2 target_size;

    float filter_radius;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] UpsamplePushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID)
{
    int2 coord = int2(thread_id.xy);

    if (coord.x >= pc.target_size.x || coord.y >= pc.target_size.y)
    {
        return;
    }

    float2 uv = (float2(coord) + 0.5) / float2(pc.target_size);
    float2 d = pc.lower_texel_size * pc.filter_radius;

    Texture2D<float4> lower = sampled_2d[pc.lower_texture_index];
    SamplerState smp = samplers[pc.linear_sampler_index];

    float3 blur = lower.SampleLevel(smp, uv + float2(-d.x,  d.y), 0).rgb * 1.0;
    blur += lower.SampleLevel(smp, uv + float2( 0.0,  d.y), 0).rgb * 2.0;
    blur += lower.SampleLevel(smp, uv + float2( d.x,  d.y), 0).rgb * 1.0;

    blur += lower.SampleLevel(smp, uv + float2(-d.x,  0.0), 0).rgb * 2.0;
    blur += lower.SampleLevel(smp, uv,                      0).rgb * 4.0;
    blur += lower.SampleLevel(smp, uv + float2( d.x,  0.0), 0).rgb * 2.0;

    blur += lower.SampleLevel(smp, uv + float2(-d.x, -d.y), 0).rgb * 1.0;
    blur += lower.SampleLevel(smp, uv + float2( 0.0, -d.y), 0).rgb * 2.0;
    blur += lower.SampleLevel(smp, uv + float2( d.x, -d.y), 0).rgb * 1.0;

    blur *= 1.0 / 16.0;

    float3 current = storage_2d[pc.target_storage_index][coord].rgb;
    storage_2d[pc.target_storage_index][coord] = float4(current + blur, 1.0);
}
