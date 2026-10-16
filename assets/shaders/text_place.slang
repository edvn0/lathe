module text_place;

import text_common;

// Mirrors PlaceParams in text_renderer.cxx.
struct PlacePushConstants
{
    Ptr<GlyphJob> jobs;
    Ptr<uint4> rects;
    Ptr<uint> state;
    uint job_count;
    uint reset;
};

[vk::push_constant] PlacePushConstants pc;

// Shelf-packs this frame's new glyph tiles into the atlas. Tiles go left to right along a shelf and a tile that does
// not fit starts the next shelf. The shelf state lives in a persistent buffer, so later frames carry on where this
// one stopped; `reset` starts the atlas over. The result is the rectangle table the compositor reads: slot ->
// (x, y, width, height), with a zero width for a tile that did not fit.
//
// Placement is a sequential walk, so it is one thread. A frame places at most a few hundred tiles.
// ShelfPlanner in rendering/text_gpu.hxx takes the same steps on the CPU, which is how it knows the atlas is full.
[shader("compute")]
[numthreads(1, 1, 1)]
void main_cs()
{
    uint cursor_x = pc.state[0];
    uint cursor_y = pc.state[1];
    uint shelf_height = pc.state[2];

    if (pc.reset != 0u)
    {
        cursor_x = 0u;
        cursor_y = 0u;
        shelf_height = 0u;
    }

    for (uint i = 0u; i < pc.job_count; i++)
    {
        GlyphJob job = pc.jobs[i];

        uint x = cursor_x;
        uint y = cursor_y;
        uint shelf = shelf_height;

        if (x + job.tile_width > atlas_size)
        {
            y += shelf;
            x = 0u;
            shelf = 0u;
        }

        if (y + job.tile_height > atlas_size || job.tile_width > atlas_size)
        {
            pc.rects[job.slot] = uint4(0u, 0u, 0u, 0u);
            continue;
        }

        pc.rects[job.slot] = uint4(x, y, job.tile_width, job.tile_height);

        cursor_x = x + job.tile_width;
        cursor_y = y;
        shelf_height = max(shelf, job.tile_height);
    }

    pc.state[0] = cursor_x;
    pc.state[1] = cursor_y;
    pc.state[2] = shelf_height;
}
