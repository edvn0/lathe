module gtao_denoise;

import bindless;
import scene_types;

// Depth-aware spatial blur over the raw GTAO output. GTAO's per-pixel
// dithered slice rotation (see gtao.slang) trades a dense sample count for
// noise; this pass cleans that noise up in one 5x5 dispatch rather than
// accumulating over time (this renderer has no TAA/history buffer to
// accumulate into), weighting neighbours by how close their *linear
// view-space* depth is to the centre pixel's so the blur doesn't bleed
// across silhouette edges. Linear (not raw hardware) depth is used for the
// weight because reverse-Z hardware depth is extremely non-linear -- two
// points at very different world-space depth can sit almost on top of each
// other in raw depth value near the far end of the range, which would make
// a raw-depth bilateral weight blur straight through real edges out there.
struct GtaoDenoisePushConstants
{
    Ptr<UBO> ubo;
    uint input_texture_index;
    uint depth_texture_index;
    uint output_storage_index;
    uint point_sampler_index;
    uint width;
    uint height;
    float2 inv_extent;
    float depth_sigma;

    // See the matching field's comment in gtao.slang's GtaoPushConstants.
    uint _padding;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] GtaoDenoisePushConstants pc;

float linear_view_z(float4x4 inv_proj, float2 uv, float depth)
{
    float2 ndc_xy = uv * 2.0 - 1.0;
    float ndc_z = 1.0 - depth;
    float4 clip = mul(inv_proj, float4(ndc_xy, ndc_z, 1.0));
    return clip.z / clip.w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void mainCs(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint2 pixel = dispatch_id.xy;

    if (pixel.x >= pc.width || pixel.y >= pc.height)
    {
        return;
    }

    float4x4 inv_proj = pc.ubo.inverse_projection;

    Texture2D<float4> ao_tex = sampled_2d[pc.input_texture_index];
    Texture2D<float> depth_tex = sampled_2d_depth[pc.depth_texture_index];
    SamplerState point_sampler = samplers[pc.point_sampler_index];

    float2 uv = (float2(pixel) + 0.5) * pc.inv_extent;

    float center_depth = depth_tex.SampleLevel(point_sampler, uv, 0);

    if (center_depth <= 0.0)
    {
        storage_2d[pc.output_storage_index][pixel] = float4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    float center_view_z = linear_view_z(inv_proj, uv, center_depth);
    float center_ao = ao_tex.SampleLevel(point_sampler, uv, 0).r;

    float total_weight = 1.0;
    float total_ao = center_ao;

    static const int radius = 2;

    [unroll]
    for (int y = -radius; y <= radius; ++y)
    {
        [unroll]
        for (int x = -radius; x <= radius; ++x)
        {
            if (x == 0 && y == 0)
            {
                continue;
            }

            float2 sample_uv = uv + float2(float(x), float(y)) * pc.inv_extent;

            if (any(sample_uv < 0.0) || any(sample_uv > 1.0))
            {
                continue;
            }

            float sample_depth = depth_tex.SampleLevel(point_sampler, sample_uv, 0);

            if (sample_depth <= 0.0)
            {
                continue;
            }

            float sample_view_z = linear_view_z(inv_proj, sample_uv, sample_depth);
            float depth_diff = (sample_view_z - center_view_z) * pc.depth_sigma;
            float weight = exp(-depth_diff * depth_diff);

            float sample_ao = ao_tex.SampleLevel(point_sampler, sample_uv, 0).r;

            total_ao += sample_ao * weight;
            total_weight += weight;
        }
    }

    float denoised = total_ao / max(total_weight, 1e-4);
    storage_2d[pc.output_storage_index][pixel] = float4(denoised, denoised, denoised, 1.0);
}
