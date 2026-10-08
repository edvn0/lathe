module hiz_occlusion;

import bindless;

public static const uint occlusion_view_disabled = 0u;
public static const uint occlusion_view_enabled = 1u;
public static const uint occlusion_view_force_occluded = 2u;

public struct OcclusionView
{
    public float4x4 view_projection;

    public Ptr<uint> meshlet_visibility;
    public Ptr<uint> stats;

    public uint hiz_texture_index;
    public uint hiz_mip_count;
    public uint depth_width;
    public uint depth_height;

    public uint enabled;
    public float depth_epsilon;
    public float guard_pixels;
    public uint _pad0;
};

public bool aabb_occluded(OcclusionView view, float3 world_min, float3 world_max)
{
    if (view.enabled == occlusion_view_force_occluded) {
        return true;
    }

    if (view.enabled != occlusion_view_enabled) {
        return false;
    }

    float2 ndc_min = float2(3.402823466e38);
    float2 ndc_max = float2(-3.402823466e38);
    float nearest = 0.0;

    [unroll]
    for (uint corner = 0; corner < 8; ++corner) {
        float3 position = float3((corner & 1u) != 0 ? world_max.x : world_min.x,
                                 (corner & 2u) != 0 ? world_max.y : world_min.y,
                                 (corner & 4u) != 0 ? world_max.z : world_min.z);

        float4 clip = mul(view.view_projection, float4(position, 1.0));

        if (clip.w <= 0.0 || clip.z < 0.0) {
            return false;
        }

        float3 ndc = clip.xyz / clip.w;

        ndc_min = min(ndc_min, ndc.xy);
        ndc_max = max(ndc_max, ndc.xy);

        nearest = max(nearest, 1.0 - ndc.z);
    }

    float width = float(view.depth_width);
    float height = float(view.depth_height);

    float x0 = (ndc_min.x * 0.5 + 0.5) * width - view.guard_pixels;
    float x1 = (ndc_max.x * 0.5 + 0.5) * width + view.guard_pixels;
    float y0 = (0.5 - ndc_max.y * 0.5) * height - view.guard_pixels;
    float y1 = (0.5 - ndc_min.y * 0.5) * height + view.guard_pixels;

    if (x1 < 0.0 || y1 < 0.0 || x0 >= width || y0 >= height) {
        return false;
    }

    int pixel_x0 = int(clamp(floor(x0), 0.0, width - 1.0));
    int pixel_y0 = int(clamp(floor(y0), 0.0, height - 1.0));
    int pixel_x1 = int(clamp(floor(x1), 0.0, width - 1.0));
    int pixel_y1 = int(clamp(floor(y1), 0.0, height - 1.0));

    uint span = uint(max(pixel_x1 - pixel_x0, pixel_y1 - pixel_y0));
    uint level = span == 0 ? 0 : firstbithigh(span);
    level = min(level, max(view.hiz_mip_count, 1u) - 1u);

    int shift = int(level + 1u);
    int texel_x0 = pixel_x0 >> shift;
    int texel_y0 = pixel_y0 >> shift;
    int texel_x1 = pixel_x1 >> shift;
    int texel_y1 = pixel_y1 >> shift;

    Texture2D<float> hiz = sampled_2d_depth[view.hiz_texture_index];
    int mip = int(level);

    float farthest = min(min(hiz.Load(int3(texel_x0, texel_y0, mip)), hiz.Load(int3(texel_x1, texel_y0, mip))),
                         min(hiz.Load(int3(texel_x0, texel_y1, mip)), hiz.Load(int3(texel_x1, texel_y1, mip))));

    return nearest + view.depth_epsilon < farthest;
}

public bool sphere_occluded(OcclusionView view, float3 centre, float radius)
{
    return aabb_occluded(view, centre - radius, centre + radius);
}
