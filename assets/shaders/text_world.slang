module text_world;

import text_common;

// Mirrors DrawPush in text_renderer.cxx.
struct WorldPushConstants
{
    float4x4 view_proj;
    Ptr<uint> atlas;
    Ptr<uint4> rects;
    Ptr<WorldGlyph> glyphs;
    Ptr<float4x4> transforms;
};

[vk::push_constant] WorldPushConstants pc;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    nointerpolation uint glyph : TEXCOORD1;
}

// (0, 0) is the quad's top-left, as the tile's first texel row is.
static const float2 corners[6] = {
    float2(0.0, 0.0), float2(1.0, 0.0), float2(1.0, 1.0),
    float2(0.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0),
};

// Six vertices per glyph: its quad in the text's local space, moved into the world by the text's transform.
[shader("vertex")]
VertexOutput main_vs(uint vertex_id: SV_VertexID)
{
    uint index = vertex_id / 6u;
    WorldGlyph glyph = pc.glyphs[index];
    float2 corner = corners[vertex_id % 6u];

    float2 local = float2(lerp(glyph.min_corner.x, glyph.max_corner.x, corner.x),
                          lerp(glyph.max_corner.y, glyph.min_corner.y, corner.y));

    float4 world = mul(pc.transforms[glyph.transform], float4(local, 0.0, 1.0));

    VertexOutput output;
    output.position = mul(pc.view_proj, world);
    output.uv = corner;
    output.glyph = index;

    return output;
}

// Drawn with source-alpha blending, depth tested and not written, after the opaque scene. The edge is the distance
// field's zero crossing; dividing the distance by how fast it changes across the screen gives it in pixels, so the
// edge is one pixel wide however the text is scaled or tilted.
[shader("fragment")]
float4 main_fs(VertexOutput input) : SV_Target0
{
    WorldGlyph glyph = pc.glyphs[input.glyph];
    uint4 rect = pc.rects[glyph.slot];

    if (rect.z == 0u)
    {
        discard;
    }

    float distance = sample_distance(pc.atlas, rect, input.uv * float2(rect.zw));
    float coverage = saturate(distance / max(fwidth(distance), 1.0e-4) + 0.5);

    float4 ink = unpack_colour(glyph.colour);

    return float4(ink.rgb * glyph.intensity, ink.a * coverage);
}
