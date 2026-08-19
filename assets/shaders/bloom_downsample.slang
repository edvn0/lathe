import bindless;

struct DownsamplePushConstants
{
    uint srcTextureIndex;
    uint linearSamplerIndex;
    uint dstMip0StorageIndex;
    uint dstMip1StorageIndex;
    uint dstMip2StorageIndex;
    uint dstMip3StorageIndex;
    float2 srcTexelSize;
    int2 mip0Size;
    float threshold;
    float knee;
};

// Bindless storage targets initialized via descriptor indexing
public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] DownsamplePushConstants pc;

groupshared float3 g_TileCache[8][8];

float3 Downsample13Tap(float2 uv, float2 texelSize)
{
    float2 d = texelSize;
    Texture2D<float4> src = sampled_2d[pc.srcTextureIndex];
    SamplerState smp = samplers[pc.linearSamplerIndex];

    float3 A = src.SampleLevel(smp, uv, 0).rgb;
    float3 B = src.SampleLevel(smp, uv + float2(-2.0 * d.x,  2.0 * d.y), 0).rgb;
    float3 C = src.SampleLevel(smp, uv + float2( 2.0 * d.x,  2.0 * d.y), 0).rgb;
    float3 D = src.SampleLevel(smp, uv + float2(-2.0 * d.x, -2.0 * d.y), 0).rgb;
    float3 E = src.SampleLevel(smp, uv + float2( 2.0 * d.x, -2.0 * d.y), 0).rgb;

    float3 F = src.SampleLevel(smp, uv + float2(-d.x,  d.y), 0).rgb;
    float3 G = src.SampleLevel(smp, uv + float2( d.x,  d.y), 0).rgb;
    float3 H = src.SampleLevel(smp, uv + float2(-d.x, -d.y), 0).rgb;
    float3 I = src.SampleLevel(smp, uv + float2( d.x, -d.y), 0).rgb;

    return (A * 0.125) + ((B + C + D + E) * 0.03125) + ((F + G + H + I) * 0.0625);
}

[numthreads(16, 16, 1)]
void mainCs(uint3 localID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    // PASS 0: Source -> Mip 0
    int2 globalCoord = groupID.xy * 16 + localID.xy;
    float3 color = float3(0.0);

    if (globalCoord.x < pc.mip0Size.x && globalCoord.y < pc.mip0Size.y)
    {
        float2 uv = (float2(globalCoord) + 0.5) / float2(pc.mip0Size);
        color = Downsample13Tap(uv, pc.srcTexelSize);

        if (pc.threshold >= 0.0)
        {
            float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
            color /= (1.0 + luma); // Partial Karis weighting
            
            float soft = clamp(luma - pc.threshold + pc.knee, 0.0, 2.0 * pc.knee);
            soft = (soft * soft) / (4.0 * pc.knee + 1e-4);
            float weight = max(soft, luma - pc.threshold) / max(luma, 1e-4);
            color *= weight;
        }

        storage_2d[pc.dstMip0StorageIndex][globalCoord] = float4(color, 1.0);
    }

    // PASS 1: Mip 0 -> Mip 1 (Subgroup Quad Broadcast In Register)
    float3 q0 = color;
    float3 q1 = WaveReadLaneAt(color, WaveGetLaneIndex() ^ 1);
    float3 q2 = WaveReadLaneAt(color, WaveGetLaneIndex() ^ 2);
    float3 q3 = WaveReadLaneAt(color, WaveGetLaneIndex() ^ 3);

    float3 mip1Color = (q0 + q1 + q2 + q3) * 0.25;

    if ((localID.x % 2 == 0) && (localID.y % 2 == 0))
    {
        int2 mip1Coord = globalCoord / 2;
        storage_2d[pc.dstMip1StorageIndex][mip1Coord] = float4(mip1Color, 1.0);
        g_TileCache[localID.x / 2][localID.y / 2] = mip1Color;
    }

    GroupMemoryBarrierWithGroupSync();

    // PASS 2: Mip 1 -> Mip 2
    if ((localID.x % 4 == 0) && (localID.y % 4 == 0))
    {
        uint x = localID.x / 2;
        uint y = localID.y / 2;
        float3 c0 = g_TileCache[x][y];
        float3 c1 = g_TileCache[x + 1][y];
        float3 c2 = g_TileCache[x][y + 1];
        float3 c3 = g_TileCache[x + 1][y + 1];
        float3 mip2Color = (c0 + c1 + c2 + c3) * 0.25;

        int2 mip2Coord = globalCoord / 4;
        storage_2d[pc.dstMip2StorageIndex][mip2Coord] = float4(mip2Color, 1.0);
        g_TileCache[localID.x / 4][localID.y / 4] = mip2Color;
    }

    GroupMemoryBarrierWithGroupSync();

    // PASS 3: Mip 2 -> Mip 3
    if ((localID.x % 8 == 0) && (localID.y % 8 == 0))
    {
        uint x = localID.x / 4;
        uint y = localID.y / 4;
        float3 c0 = g_TileCache[x][y];
        float3 c1 = g_TileCache[x + 1][y];
        float3 c2 = g_TileCache[x][y + 1];
        float3 c3 = g_TileCache[x + 1][y + 1];
        float3 mip3Color = (c0 + c1 + c2 + c3) * 0.25;

        int2 mip3Coord = globalCoord / 8;
        storage_2d[pc.dstMip3StorageIndex][mip3Coord] = float4(mip3Color, 1.0);
    }
}
