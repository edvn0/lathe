module hiz_build;

import bindless;

struct HizBuildPushConstants
{
    uint src_texture_index;

    uint dst_storage_index;

    uint src_width;
    uint src_height;
    uint dst_width;
    uint dst_height;
};

[[vk::binding(4, 0)]] [format("r32f")] RWTexture2D<float> storage_r32f[];

[vk::push_constant] HizBuildPushConstants pc;

static const float hiz_neutral_depth = 1.0;

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
