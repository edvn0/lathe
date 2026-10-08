module env_prefilter;

import bindless;
import environment;

struct EnvPrefilterPushConstants
{
    uint cube_texture_index;
    uint sampler_index;
    uint dst_storage_index;
    uint face;

    uint dst_size;
    uint sample_count;
    uint radiance_size;
    float roughness;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] EnvPrefilterPushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= pc.dst_size || id.y >= pc.dst_size) {
        return;
    }

    const TextureCube<float4> cube = sampled_cube[pc.cube_texture_index];
    const SamplerState sampler = samplers[pc.sampler_index];

    const float2 uv = (float2(id.xy) + 0.5) / float(pc.dst_size);
    const float3 n = cube_texel_direction(pc.face, uv);

    float3 result;

    if (pc.roughness <= 0.0) {
        result = cube.SampleLevel(sampler, n, max(log2(float(pc.radiance_size) / 256.0), 0.0)).rgb;
    } else {
        float3 tangent;
        float3 bitangent;
        tangent_frame(n, tangent, bitangent);

        const float alpha = pc.roughness * pc.roughness;
        const float texel_solid_angle = 4.0 * env_pi / (6.0 * float(pc.radiance_size) * float(pc.radiance_size));

        float3 sum = float3(0.0);
        float weight = 0.0;

        for (uint i = 0; i < pc.sample_count; ++i) {
            const float3 h_local = importance_sample_ggx(hammersley(i, pc.sample_count), alpha);
            const float3 h = normalize(tangent * h_local.x + bitangent * h_local.y + n * h_local.z);
            const float3 l = 2.0 * dot(n, h) * h - n;

            const float ndotl = dot(n, l);

            if (ndotl > 0.0) {
                const float pdf = distribution_ggx_alpha(h_local.z, alpha) * 0.25;
                const float sample_solid_angle = 1.0 / (float(pc.sample_count) * max(pdf, 1e-6));
                const float lod = max(0.5 * log2(sample_solid_angle / texel_solid_angle) + 1.0, 0.0);

                sum += cube.SampleLevel(sampler, l, lod).rgb * ndotl;
                weight += ndotl;
            }
        }

        result = sum / max(weight, 1e-4);
    }

    storage_2d[pc.dst_storage_index][id.xy] = float4(result, 1.0);
}
