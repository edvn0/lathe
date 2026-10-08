module env_sh_project;

import bindless;
import scene_types;
import environment;

struct EnvShProjectPushConstants
{
    uint cube_texture_index;
    uint sampler_index;
    uint level;
    uint level_size;
    Ptr<GpuEnvironmentSh> destination;
};

[vk::push_constant] EnvShProjectPushConstants pc;

static const uint group_size = 256;

groupshared float partial_sums[group_size][27];

[shader("compute")]
[numthreads(group_size, 1, 1)]
void main_cs(uint lane: SV_GroupIndex)
{
    const TextureCube<float4> cube = sampled_cube[pc.cube_texture_index];
    const SamplerState sampler = samplers[pc.sampler_index];

    const uint texels_per_face = pc.level_size * pc.level_size;
    const uint total = texels_per_face * 6;

    float sums[27];

    [unroll]
    for (uint i = 0; i < 27; ++i) {
        sums[i] = 0.0;
    }

    for (uint texel = lane; texel < total; texel += group_size) {
        const uint face = texel / texels_per_face;
        const uint within = texel % texels_per_face;
        const uint x = within % pc.level_size;
        const uint y = within / pc.level_size;

        const float2 uv = (float2(x, y) + 0.5) / float(pc.level_size);
        const float3 direction = cube_texel_direction(face, uv);
        const float3 radiance = cube.SampleLevel(sampler, direction, float(pc.level)).rgb;
        const float weight = cube_texel_solid_angle(x, y, pc.level_size);

        const Sh9Basis basis = sh9_basis(direction);

        [unroll]
        for (uint c = 0; c < 9; ++c) {
            const float w = basis.values[c] * weight;

            sums[c * 3 + 0] += radiance.x * w;
            sums[c * 3 + 1] += radiance.y * w;
            sums[c * 3 + 2] += radiance.z * w;
        }
    }

    [unroll]
    for (uint i = 0; i < 27; ++i) {
        partial_sums[lane][i] = sums[i];
    }

    GroupMemoryBarrierWithGroupSync();

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            [unroll]
            for (uint i = 0; i < 27; ++i) {
                partial_sums[lane][i] += partial_sums[lane + stride][i];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0) {
        const float band_factor[9] = {1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25};

        [unroll]
        for (uint c = 0; c < 9; ++c) {
            pc.destination.coefficients[c] = float4(partial_sums[0][c * 3 + 0] * band_factor[c],
                                                    partial_sums[0][c * 3 + 1] * band_factor[c],
                                                    partial_sums[0][c * 3 + 2] * band_factor[c], 0.0);
        }
    }
}
