module gtao;

import bindless;
import scene_types;

struct GtaoPushConstants
{
    Ptr<UBO> ubo;
    uint depth_texture_index;
    uint output_storage_index;
    uint point_sampler_index;
    uint width;
    uint height;
    float2 inv_extent;
    float radius_view;
    float falloff_range;
    uint slice_count;
    uint step_count;

    uint _padding;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] GtaoPushConstants pc;

static const float pi = 3.14159265358979323846;

float3 view_pos_from_depth(float4x4 inv_proj, float2 uv, float depth)
{
    float2 ndc_xy = uv * 2.0 - 1.0;
    float ndc_z = 1.0 - depth;
    float4 clip = mul(inv_proj, float4(ndc_xy, ndc_z, 1.0));
    return clip.xyz / clip.w;
}

float horizon_cos(Texture2D<float> depth_tex, SamplerState point_sampler, float4x4 inv_proj, float2 uv,
                   float2 dir_uv, float2 texel, float3 p, float3 view_dir, float step_pixels, uint step_count,
                   float radius_view, float falloff_start)
{
    float max_cos = 0.0;

    for (uint i = 1; i <= step_count; ++i)
    {
        float2 sample_uv = uv + dir_uv * (step_pixels * float(i)) * texel;

        if (any(sample_uv < 0.0) || any(sample_uv > 1.0))
        {
            break;
        }

        float sample_depth = depth_tex.SampleLevel(point_sampler, sample_uv, 0);

        if (sample_depth <= 0.0)
        {
            continue;
        }

        float3 sample_pos = view_pos_from_depth(inv_proj, sample_uv, sample_depth);
        float3 horizon_vec = sample_pos - p;
        float horizon_len = length(horizon_vec);

        if (horizon_len < 1e-4)
        {
            continue;
        }

        float sample_cos = dot(horizon_vec, view_dir) / horizon_len;

        float range_weight = saturate((radius_view - horizon_len) / max(radius_view - falloff_start, 1e-4));
        float weighted_cos = lerp(0.0, sample_cos, range_weight);

        max_cos = max(max_cos, weighted_cos);
    }

    return max_cos;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint2 pixel = dispatch_id.xy;

    if (pixel.x >= pc.width || pixel.y >= pc.height)
    {
        return;
    }

    float4x4 inv_proj = pc.ubo.inverse_projection;
    float4x4 proj = pc.ubo.projection;

    Texture2D<float> depth_tex = sampled_2d_depth[pc.depth_texture_index];
    SamplerState point_sampler = samplers[pc.point_sampler_index];

    float2 texel = pc.inv_extent;
    float2 uv = (float2(pixel) + 0.5) * texel;

    float center_depth = depth_tex.SampleLevel(point_sampler, uv, 0);

    if (center_depth <= 0.0)
    {
        storage_2d[pc.output_storage_index][pixel] = float4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    float3 p = view_pos_from_depth(inv_proj, uv, center_depth);

    float2 dx = float2(texel.x, 0.0);
    float2 dy = float2(0.0, texel.y);

    float d_l = depth_tex.SampleLevel(point_sampler, uv - dx, 0);
    float d_r = depth_tex.SampleLevel(point_sampler, uv + dx, 0);
    float d_u = depth_tex.SampleLevel(point_sampler, uv - dy, 0);
    float d_d = depth_tex.SampleLevel(point_sampler, uv + dy, 0);

    float3 p_l = view_pos_from_depth(inv_proj, uv - dx, max(d_l, 1e-6));
    float3 p_r = view_pos_from_depth(inv_proj, uv + dx, max(d_r, 1e-6));
    float3 p_u = view_pos_from_depth(inv_proj, uv - dy, max(d_u, 1e-6));
    float3 p_d = view_pos_from_depth(inv_proj, uv + dy, max(d_d, 1e-6));

    float3 tangent_x = (abs(p_l.z - p.z) < abs(p_r.z - p.z)) ? (p - p_l) : (p_r - p);
    float3 tangent_y = (abs(p_u.z - p.z) < abs(p_d.z - p.z)) ? (p - p_u) : (p_d - p);

    float3 n = normalize(cross(tangent_y, tangent_x));

    float3 view_dir = normalize(-p);

    if (dot(n, view_dir) < 0.0)
    {
        n = -n;
    }

    float3 radius_probe = p + float3(pc.radius_view, 0.0, 0.0);
    float4 radius_clip = mul(proj, float4(radius_probe, 1.0));
    float2 radius_ndc = radius_clip.xy / max(radius_clip.w, 1e-5);
    float2 radius_uv = radius_ndc * 0.5 + 0.5;
    float radius_pixels = length((radius_uv - uv) * float2(pc.width, pc.height));
    radius_pixels = clamp(radius_pixels, float(pc.step_count), 256.0);

    float step_pixels = radius_pixels / float(pc.step_count);
    float falloff_start = pc.radius_view * pc.falloff_range;

    float ign = frac(52.9829189 * frac(dot(float2(pixel), float2(0.06711056, 0.00583715))));

    float visibility = 0.0;

    for (uint slice = 0; slice < pc.slice_count; ++slice)
    {
        float slice_angle = (float(slice) + ign) * (pi / float(pc.slice_count));
        float2 slice_dir_uv = float2(cos(slice_angle), sin(slice_angle));

        float3 slice_dir_view = normalize(float3(slice_dir_uv, 0.0));
        float3 plane_normal = normalize(cross(slice_dir_view, view_dir));

        float3 n_proj = n - plane_normal * dot(n, plane_normal);
        float n_proj_len = max(length(n_proj), 1e-5);
        float3 n_proj_dir = n_proj / n_proj_len;

        float sin_theta_n = clamp(dot(n_proj_dir, view_dir), 0.0, 1.0);
        float cos_theta_n_neg = -dot(n_proj_dir, slice_dir_view);
        float theta_n = atan2(sin_theta_n, cos_theta_n_neg);

        float max_cos_1 = horizon_cos(depth_tex, point_sampler, inv_proj, uv, slice_dir_uv, texel, p, view_dir,
                                      step_pixels, pc.step_count, pc.radius_view, falloff_start);
        float max_cos_2 = horizon_cos(depth_tex, point_sampler, inv_proj, uv, -slice_dir_uv, texel, p, view_dir,
                                      step_pixels, pc.step_count, pc.radius_view, falloff_start);

        float theta_h1 = pi - asin(clamp(max_cos_1, -1.0, 1.0));
        float theta_h2 = asin(clamp(max_cos_2, -1.0, 1.0));

        theta_h1 = clamp(theta_h1, theta_n, min(theta_n + pi * 0.5, pi));
        theta_h2 = clamp(theta_h2, max(theta_n - pi * 0.5, 0.0), theta_n);

        float slice_visibility = (sin(theta_h1 - theta_n) - sin(theta_h2 - theta_n)) / (1.0 + sin_theta_n);
        visibility += saturate(slice_visibility);
    }

    visibility /= float(pc.slice_count);

    storage_2d[pc.output_storage_index][pixel] = float4(visibility, visibility, visibility, 1.0);
}
