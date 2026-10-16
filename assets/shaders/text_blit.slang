module text_blit;

import text_common;

// Mirrors BlitParams in text_renderer.cxx.
struct BlitPushConstants
{
    Ptr<float> tiles;
    Ptr<GlyphJob> jobs;
    Ptr<uint4> rects;
    Ptr<uint> atlas;
    uint job_count;
    uint quad_count;
};

[vk::push_constant] BlitPushConstants pc;

// Copies the rasterised tiles into the atlas at the rectangles text_place chose, turning each distance into a byte:
// 128 on the outline, and `spread` texels either side map to the 127 steps below and above it.
// A thread writes four texels, one 32-bit word. Tiles are a multiple of four wide and so are the atlas offsets, so
// the four always lie in one row of one tile, and no two threads share a word.
[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID)
{
    uint quad = thread_id.x;

    if (quad >= pc.quad_count)
    {
        return;
    }

    uint texel = quad * 4u;
    GlyphJob job = pc.jobs[find_job(pc.jobs, pc.job_count, texel)];
    uint4 rect = pc.rects[job.slot];

    if (rect.z == 0u)
    {
        return;
    }

    uint local = texel - job.texel_offset;
    uint tx = local % job.tile_width;
    uint ty = local / job.tile_width;

    uint packed = 0u;

    for (uint k = 0u; k < 4u; k++)
    {
        float value = clamp(128.0 + pc.tiles[texel + k] * (127.0 / spread), 0.0, 255.0);
        packed |= uint(value + 0.5) << (8u * k);
    }

    uint address = (rect.y + ty) * atlas_size + rect.x + tx;

    pc.atlas[address >> 2u] = packed;
}
