module env_equirect_to_cube;

import bindless;
import environment;

// Projects an equirect panorama onto one face of the radiance cube (mip 0): 2x2 rotated-grid samples per texel through
// a sampler that repeats in U and clamps in V.
struct EnvEquirectToCubePushConstants
{
    uint equirect_texture_index;
    uint sampler_index;
    uint dst_storage_index;
    uint face;
    uint size;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] EnvEquirectToCubePushConstants pc;

float3 sample_equirect(float2 texel, float2 offset)
{
    const Texture2D<float4> equirect = sampled_2d[pc.equirect_texture_index];
    const SamplerState sampler = samplers[pc.sampler_index];

    const float2 uv = (texel + 0.5 + offset) / float(pc.size);
    const float3 direction = cube_texel_direction(pc.face, uv);

    return equirect.SampleLevel(sampler, direction_to_equirect_uv(direction), 0).rgb;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= pc.size || id.y >= pc.size) {
        return;
    }

    const float2 texel = float2(id.xy);

    // A rotated 2x2 grid, spelled out: an unrolled loop over an array here crashed spirv-opt's BlockMergePass.
    float3 radiance = sample_equirect(texel, float2(-0.125, -0.375)) + sample_equirect(texel, float2(0.375, -0.125)) +
                      sample_equirect(texel, float2(0.125, 0.375)) + sample_equirect(texel, float2(-0.375, 0.125));
    radiance *= 0.25;

    // Filtering already works on values clamped to 65000 at load; this guards against NaN from a damaged source.
    radiance = min(max(radiance, float3(0.0)), float3(65000.0));

    storage_2d[pc.dst_storage_index][id.xy] = float4(radiance, 1.0);
}
