module text_common;

// Mirrors rendering/text_gpu.hxx.
public static const uint atlas_size = 2048u;
public static const float raster_em = 64.0;
// Texels the distance field reaches outside (and inside) the outline.
public static const float spread = 8.0;
public static const uint screen_tile = 16u;
public static const uint tile_capacity = 64u;

public struct GlyphJob
{
    public uint segment_offset;
    public uint segment_count;
    public uint tile_width;
    public uint tile_height;
    public float origin_x;
    public float origin_y;
    public uint slot;
    public uint texel_offset;
};

public struct GlyphInstance
{
    public float2 min_corner;
    public float2 max_corner;
    public uint slot;
    public uint colour;
    public uint padding0;
    public uint padding1;
};

// The job a texel of the scratch buffer belongs to: the last whose texel_offset is not past it. Jobs are in
// ascending texel_offset order.
public uint find_job(Ptr<GlyphJob> jobs, uint job_count, uint texel)
{
    uint low = 0u;
    uint high = job_count;

    while (high - low > 1u)
    {
        uint middle = (low + high) / 2u;

        if (jobs[middle].texel_offset <= texel)
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }

    return low;
}

public struct WorldGlyph
{
    public float2 min_corner;
    public float2 max_corner;
    public uint slot;
    public uint colour;
    public uint transform;
    public float intensity;
};

// A distance-field texel of a tile as a signed distance in texels, positive inside. The atlas holds one byte per
// texel, 128 on the outline.
public float atlas_distance(Ptr<uint> atlas, uint4 rect, int x, int y)
{
    x = clamp(x, 0, int(rect.z) - 1);
    y = clamp(y, 0, int(rect.w) - 1);

    uint address = (rect.y + uint(y)) * atlas_size + rect.x + uint(x);
    uint word = atlas[address >> 2u];
    uint value = (word >> ((address & 3u) * 8u)) & 255u;

    return (float(value) - 128.0) * (spread / 127.0);
}

// The distance at a point of a tile given in texels, bilinearly filtered.
public float sample_distance(Ptr<uint> atlas, uint4 rect, float2 texel)
{
    float2 corner = texel - 0.5;
    int2 base = int2(floor(corner));
    float2 f = corner - float2(base);

    float d00 = atlas_distance(atlas, rect, base.x, base.y);
    float d10 = atlas_distance(atlas, rect, base.x + 1, base.y);
    float d01 = atlas_distance(atlas, rect, base.x, base.y + 1);
    float d11 = atlas_distance(atlas, rect, base.x + 1, base.y + 1);

    return lerp(lerp(d00, d10, f.x), lerp(d01, d11, f.x), f.y);
}

// Unpacks a colour from 8-bit RGBA, red in the low byte.
public float4 unpack_colour(uint packed)
{
    return float4(packed & 255u, (packed >> 8u) & 255u, (packed >> 16u) & 255u, packed >> 24u) / 255.0;
}
