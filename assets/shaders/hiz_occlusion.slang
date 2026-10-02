module hiz_occlusion;

import bindless;

// Conservative Hi-Z occlusion test for world-space boxes. Mirrored on the CPU by include/rendering/hiz_occlusion.hxx,
// whose unit tests cover it; keep the two in lockstep. See docs/occlusion-culling.md.
//
// Stored depth is 1 - ndc.z (reverse-Z through the viewport), so larger is nearer. The pyramid keeps the farthest
// (MIN) depth per texel; logical level L is ceil(depth / 2^(L+1)) and pixel p maps to texel p >> (L+1).

// OcclusionView::enabled values.
public static const uint occlusion_view_disabled = 0u;       // never occluded: no history, or culling is off
public static const uint occlusion_view_enabled = 1u;        // the Hi-Z test
public static const uint occlusion_view_force_occluded = 2u; // debug stub (OcclusionTestMode::always_defer)

// Mirrors GpuOcclusionView in renderer.hxx (112 bytes). The renderer's occlusion_views_buffer holds two: [0] last
// frame's pyramid with the view-projection it was built with (phase 1), [1] this frame's (phase 2).
public struct OcclusionView
{
    public float4x4 view_projection;

    // Reserved for meshlet-level occlusion (docs/occlusion-culling-m2.md); null until then.
    public Ptr<uint> meshlet_visibility;
    public Ptr<uint> stats;

    // The whole mip chain through one sampled_2d_depth view; levels are read with Load(int3(texel, level)).
    public uint hiz_texture_index;
    public uint hiz_mip_count;
    public uint depth_width;
    public uint depth_height;

    public uint enabled;
    public float depth_epsilon;
    public float guard_pixels;
    public uint _pad0;
};

// True only when the box is certainly hidden behind depth already in the pyramid. Boxes the test can't decide (a
// corner behind the eye or nearer than the near plane, or off screen) are visible.
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

        // Behind the eye or nearer than the near plane: the projection can't bound the box.
        if (clip.w <= 0.0 || clip.z < 0.0) {
            return false;
        }

        float3 ndc = clip.xyz / clip.w;

        ndc_min = min(ndc_min, ndc.xy);
        ndc_max = max(ndc_max, ndc.xy);

        // The viewport maps ndc.z 0..1 to depth 1..0.
        nearest = max(nearest, 1.0 - ndc.z);
    }

    float width = float(view.depth_width);
    float height = float(view.depth_height);

    // The viewport flips Y: pixel row 0 is ndc.y = +1.
    float x0 = (ndc_min.x * 0.5 + 0.5) * width - view.guard_pixels;
    float x1 = (ndc_max.x * 0.5 + 0.5) * width + view.guard_pixels;
    float y0 = (0.5 - ndc_max.y * 0.5) * height - view.guard_pixels;
    float y1 = (0.5 - ndc_min.y * 0.5) * height + view.guard_pixels;

    // Off screen is the frustum test's job.
    if (x1 < 0.0 || y1 < 0.0 || x0 >= width || y0 >= height) {
        return false;
    }

    int pixel_x0 = int(clamp(floor(x0), 0.0, width - 1.0));
    int pixel_y0 = int(clamp(floor(y0), 0.0, height - 1.0));
    int pixel_x1 = int(clamp(floor(x1), 0.0, width - 1.0));
    int pixel_y1 = int(clamp(floor(y1), 0.0, height - 1.0));

    // The finest level where the rect spans at most 2 x 2 texels: span < 2^(level + 1).
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

    // Strictly behind everything in the footprint; ties are visible.
    return nearest + view.depth_epsilon < farthest;
}

public bool sphere_occluded(OcclusionView view, float3 centre, float radius)
{
    return aabb_occluded(view, centre - radius, centre + radius);
}
