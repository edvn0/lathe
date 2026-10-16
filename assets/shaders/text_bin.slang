module text_bin;

import text_common;

// Mirrors BinParams in text_renderer.cxx.
struct BinPushConstants
{
    Ptr<GlyphInstance> instances;
    Ptr<uint> counts;
    Ptr<uint> lists;
    uint instance_count;
    uint tiles_x;
    uint tiles_y;
};

[vk::push_constant] BinPushConstants pc;

// One thread per glyph quad: appends the quad's index to the list of every screen tile it touches. The lists are in
// arrival order, which is not draw order; the compositor sorts a tile's list before it blends.
[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;

    if (index >= pc.instance_count)
    {
        return;
    }

    GlyphInstance glyph = pc.instances[index];

    int2 first = int2(floor(glyph.min_corner / float(screen_tile)));
    int2 last = int2(floor((glyph.max_corner - 0.001) / float(screen_tile)));

    first = max(first, int2(0, 0));
    last = min(last, int2(int(pc.tiles_x) - 1, int(pc.tiles_y) - 1));

    for (int y = first.y; y <= last.y; y++)
    {
        for (int x = first.x; x <= last.x; x++)
        {
            uint tile = uint(y) * pc.tiles_x + uint(x);
            uint position;

            InterlockedAdd(pc.counts[tile], 1u, position);

            if (position < tile_capacity)
            {
                pc.lists[tile * tile_capacity + position] = index;
            }
        }
    }
}
