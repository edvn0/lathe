module text_composite;

import bindless;
import text_common;

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

// Mirrors CompositeParams in text_renderer.cxx.
struct CompositePushConstants
{
    uint scene;
    uint result;
    Ptr<uint> atlas;
    Ptr<uint4> rects;
    Ptr<uint> counts;
    Ptr<uint> lists;
    Ptr<GlyphInstance> instances;
    uint tiles_x;
    uint width;
    uint height;
};

[vk::push_constant] CompositePushConstants pc;

groupshared uint s_list[tile_capacity];
groupshared uint s_count;

// One 16x16 workgroup per screen tile. The group sorts the tile's glyph list into draw order, then every pixel blends
// the glyphs that cover it, in that order, over the scene colour. The edge is the distance field's zero crossing,
// anti-aliased over one screen pixel, so text is sharp at any size.
[shader("compute")]
[numthreads(16, 16, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint group_index : SV_GroupIndex)
{
    uint tile = group_id.y * pc.tiles_x + group_id.x;

    if (group_index == 0u)
    {
        uint count = min(pc.counts[tile], tile_capacity);

        for (uint i = 0u; i < count; i++)
        {
            s_list[i] = pc.lists[tile * tile_capacity + i];
        }

        for (uint i = 1u; i < count; i++)
        {
            uint value = s_list[i];
            int j = int(i) - 1;

            while (j >= 0 && s_list[j] > value)
            {
                s_list[j + 1] = s_list[j];
                j--;
            }

            s_list[j + 1] = value;
        }

        s_count = count;
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread_id.x >= pc.width || thread_id.y >= pc.height)
    {
        return;
    }

    float4 colour = sampled_2d[pc.scene].Load(int3(thread_id.xy, 0));
    float2 pixel = float2(thread_id.xy) + 0.5;

    for (uint k = 0u; k < s_count; k++)
    {
        GlyphInstance glyph = pc.instances[s_list[k]];

        if (any(pixel < glyph.min_corner) || any(pixel >= glyph.max_corner))
        {
            continue;
        }

        uint4 rect = pc.rects[glyph.slot];

        if (rect.z == 0u)
        {
            continue;
        }

        float2 extent = glyph.max_corner - glyph.min_corner;
        float2 texel = (pixel - glyph.min_corner) / extent * float2(rect.zw);
        float pixels_per_texel = extent.x / float(rect.z);

        float coverage = saturate(sample_distance(pc.atlas, rect, texel) * pixels_per_texel + 0.5);

        float4 ink = unpack_colour(glyph.colour);

        colour.rgb = lerp(colour.rgb, ink.rgb, coverage * ink.a);
    }

    storage_2d[pc.result][thread_id.xy] = colour;
}
