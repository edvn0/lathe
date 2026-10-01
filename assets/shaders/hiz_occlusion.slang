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

    return false;
}

public bool sphere_occluded(OcclusionView view, float3 centre, float radius)
{
    return aabb_occluded(view, centre - radius, centre + radius);
}
