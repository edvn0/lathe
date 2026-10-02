module hiz_build;

import bindless;

// One level of the Hi-Z pyramid (docs/occlusion-culling.md): each texel is the farthest, i.e. MIN under reverse-Z,
// of its 2 x 2 children in the level below (or in the depth buffer for level 0). Children outside the source's
// logical extent are skipped, so the edges of odd-sized levels stay conservative. Mirrors build_hiz_reference() in
// test/hiz_occlusion_test.cxx; render_pass::build_hiz() places a barrier between levels.
struct HizBuildPushConstants
{
    // sampled_2d_depth index of the source: the depth buffer, or the single-mip view of the level below.
    uint src_texture_index;

    // storage_2d index of the single-mip view being written.
    uint dst_storage_index;

    // Logical extents; the image itself is a power of two.
    uint src_width;
    uint src_height;
    uint dst_width;
    uint dst_height;
};

// The pyramid is R32_SFLOAT; the explicit format avoids needing shaderStorageImageWriteWithoutFormat.
[[vk::binding(4, 0)]] [format("r32f")] RWTexture2D<float> storage_r32f[];

[vk::push_constant] HizBuildPushConstants pc;

// Reverse-Z: 1.0 is the nearest possible depth, the neutral value for MIN.
static const float hiz_neutral_depth = 1.0;

// One child's depth, or the neutral value past the source's logical extent. The fetch is clamped into the source so it
// never reads outside it, and there is no branch around it: nested [unroll]ed conditional accumulation made
// spirv-opt's performance passes (BlockMergePass) segfault on this shader.
float hiz_child_depth(Texture2D<float> source, uint2 child)
{
    uint2 last = uint2(pc.src_width - 1, pc.src_height - 1);
    float depth = source.Load(int3(int2(min(child, last)), 0));
    bool inside = child.x < pc.src_width && child.y < pc.src_height;

    return inside ? depth : hiz_neutral_depth;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 dispatch_id: SV_DispatchThreadID)
{
    if (dispatch_id.x >= pc.dst_width || dispatch_id.y >= pc.dst_height) {
        return;
    }

    Texture2D<float> source = sampled_2d_depth[pc.src_texture_index];
    uint2 base = dispatch_id.xy * 2;

    float farthest = min(min(hiz_child_depth(source, base), hiz_child_depth(source, base + uint2(1, 0))),
                         min(hiz_child_depth(source, base + uint2(0, 1)), hiz_child_depth(source, base + uint2(1, 1))));

    storage_r32f[pc.dst_storage_index][dispatch_id.xy] = farthest;
}
