module env_brdf_lut;

import bindless;
import environment;

struct EnvBrdfLutPushConstants
{
    uint dst_storage_index;
    uint size;
    uint sample_count;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] EnvBrdfLutPushConstants pc;

float2 integrate_brdf(float ndotv, float roughness)
{
    const float alpha = roughness * roughness;
    const float3 v = float3(sqrt(1.0 - ndotv * ndotv), 0.0, ndotv);

    float a = 0.0;
    float b = 0.0;

    for (uint i = 0; i < pc.sample_count; ++i) {
        const float2 xi = hammersley(i, pc.sample_count);
        const float3 h = importance_sample_ggx(xi, alpha);
        const float3 l = 2.0 * dot(v, h) * h - v;

        const float ndotl = saturate(l.z);
        const float ndoth = saturate(h.z);
        const float vdoth = saturate(dot(v, h));

        if (ndotl > 0.0) {
            const float a2 = alpha * alpha;
            const float lambda_v = ndotl * sqrt(ndotv * ndotv * (1.0 - a2) + a2);
            const float lambda_l = ndotv * sqrt(ndotl * ndotl * (1.0 - a2) + a2);
            const float visibility = 0.5 / max(lambda_v + lambda_l, 1e-6);

            const float weighted = visibility * 4.0 * ndotl * vdoth / max(ndoth, 1e-6);
            const float fresnel = pow(1.0 - vdoth, 5.0);

            a += (1.0 - fresnel) * weighted;
            b += fresnel * weighted;
        }
    }

    return float2(a, b) / float(pc.sample_count);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= pc.size || id.y >= pc.size) {
        return;
    }

    const float ndotv = max((float(id.x) + 0.5) / float(pc.size), 1e-3);
    const float roughness = max((float(id.y) + 0.5) / float(pc.size), 0.045);

    storage_2d[pc.dst_storage_index][id.xy] = float4(integrate_brdf(ndotv, roughness), 0.0, 1.0);
}
