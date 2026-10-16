module text_raster;

import text_common;

// Mirrors RasterParams in text_renderer.cxx.
struct RasterPushConstants
{
    Ptr<float4> segments;
    Ptr<GlyphJob> jobs;
    Ptr<float> tiles;
    uint job_count;
    uint texel_count;
};

[vk::push_constant] RasterPushConstants pc;

// One thread per texel of every new glyph's tile: the signed distance, in texels, from the texel's centre to the
// glyph's outline, positive inside. The outline is a list of line segments; the distance is to the nearest, and
// "inside" is a non-zero winding number of the outline around the point.
[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID)
{
    uint texel = thread_id.x;

    if (texel >= pc.texel_count)
    {
        return;
    }

    GlyphJob job = pc.jobs[find_job(pc.jobs, pc.job_count, texel)];

    uint local = texel - job.texel_offset;
    uint tx = local % job.tile_width;
    uint ty = local / job.tile_width;

    float2 p = float2(job.origin_x + (float(tx) + 0.5) / raster_em, job.origin_y - (float(ty) + 0.5) / raster_em);

    float nearest = 1.0e30;
    int winding = 0;

    for (uint i = 0u; i < job.segment_count; i++)
    {
        float4 segment = pc.segments[job.segment_offset + i];
        float2 a = segment.xy;
        float2 b = segment.zw;
        float2 ab = b - a;

        float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1.0e-12), 0.0, 1.0);
        float2 offset = p - (a + ab * t);
        nearest = min(nearest, dot(offset, offset));

        // A ray from p towards +x crosses this segment: count it by the direction the segment goes.
        if ((a.y <= p.y) != (b.y <= p.y))
        {
            float crossing = a.x + (p.y - a.y) / (b.y - a.y) * ab.x;

            if (crossing > p.x)
            {
                winding += (b.y > a.y) ? 1 : -1;
            }
        }
    }

    float distance = sqrt(nearest) * raster_em;

    pc.tiles[texel] = (winding != 0) ? distance : -distance;
}
