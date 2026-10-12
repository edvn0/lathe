module tint;

import bindless;

// Mirrors tint.json: its bindings, then its params, in that order.
struct TintPushConstants
{
    uint source;
    uint result;
    float strength;
    float3 tint;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] TintPushConstants pc;

// Blends the scene colour towards a tint and writes the result, which composition then samples.
[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID)
{
    uint width, height;
    sampled_2d[pc.source].GetDimensions(width, height);

    if (thread_id.x >= width || thread_id.y >= height)
    {
        return;
    }

    float4 colour = sampled_2d[pc.source].Load(int3(thread_id.xy, 0));
    float luminance = dot(colour.rgb, float3(0.2126, 0.7152, 0.0722));
    float3 tinted = luminance * pc.tint * 2.0;

    storage_2d[pc.result][thread_id.xy] = float4(lerp(colour.rgb, tinted, pc.strength), colour.a);
}
