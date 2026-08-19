import bindless;

struct UpsamplePushConstants
{
    uint lowerMipTextureIndex;  // Index in sampled_2d[]
    uint targetMipStorageIndex; // Index in storage_2d[]
    uint linearSamplerIndex;    // Index in samplers[]
    float2 lowerTexelSize;
    int2 targetSize;
    float filterRadius;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] UpsamplePushConstants pc;

[numthreads(16, 16, 1)]
void mainCs(uint3 threadID : SV_DispatchThreadID)
{
    if (threadID.x >= (uint)pc.targetSize.x || threadID.y >= (uint)pc.targetSize.y)
        return;

    float2 uv = (float2(threadID.xy) + 0.5) / float2(pc.targetSize);
    float2 d = pc.lowerTexelSize * pc.filterRadius;

    Texture2D<float4> lowerMip = sampled_2d[pc.lowerMipTextureIndex];
    SamplerState smp = samplers[pc.linearSamplerIndex];

    float3 blur = lowerMip.SampleLevel(smp, uv + float2(-d.x,  d.y), 0).rgb * 2.0;
    blur += lowerMip.SampleLevel(smp, uv + float2( 0.0,  d.y), 0).rgb * 1.0;
    blur += lowerMip.SampleLevel(smp, uv + float2( d.x,  d.y), 0).rgb * 2.0;

    blur += lowerMip.SampleLevel(smp, uv + float2(-d.x,  0.0), 0).rgb * 1.0;
    blur += lowerMip.SampleLevel(smp, uv + float2( 0.0,  0.0), 0).rgb * 4.0;
    blur += lowerMip.SampleLevel(smp, uv + float2( d.x,  0.0), 0).rgb * 1.0;

    blur += lowerMip.SampleLevel(smp, uv + float2(-d.x, -d.y), 0).rgb * 2.0;
    blur += lowerMip.SampleLevel(smp, uv + float2( 0.0, -d.y), 0).rgb * 1.0;
    blur += lowerMip.SampleLevel(smp, uv + float2( d.x, -d.y), 0).rgb * 2.0;

    blur *= (1.0 / 16.0);

    // Read-modify-write via bindless storage handle
    float3 current = storage_2d[pc.targetMipStorageIndex][threadID.xy].rgb;
    storage_2d[pc.targetMipStorageIndex][threadID.xy] = float4(current + blur, 1.0);
}
