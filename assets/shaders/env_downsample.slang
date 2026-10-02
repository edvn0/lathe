module env_downsample;

import bindless;

// One face of one level of the radiance mip chain: a 2x2 box filter of the level below, within the face. The source is
// that face's single-mip view, so this never samples a subresource that is still being written.
struct EnvDownsamplePushConstants
{
    uint src_texture_index;
    uint sampler_index;
    uint dst_storage_index;
    uint dst_size;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] EnvDownsamplePushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= pc.dst_size || id.y >= pc.dst_size) {
        return;
    }

    const Texture2D<float4> source = sampled_2d[pc.src_texture_index];
    const SamplerState sampler = samplers[pc.sampler_index];

    // The centre of a destination texel is the corner of its 2x2 source footprint, where bilinear filtering averages
    // exactly those four texels.
    const float2 uv = (float2(id.xy) + 0.5) / float(pc.dst_size);

    storage_2d[pc.dst_storage_index][id.xy] = float4(source.SampleLevel(sampler, uv, 0).rgb, 1.0);
}
