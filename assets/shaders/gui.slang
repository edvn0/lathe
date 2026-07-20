import bindless;

struct Vertex
{
    float2 xy;
    float2 uv;
    uint rgba;
}

struct PC
{
    float4 LRTB;
    Ptr<Vertex> vertices;
    uint base_vertex;
    uint texture_id;
    uint sampler_id;
    uint pad;
}

struct VertexOutput
{
    float4 position : SV_Position;
    float4 colour : TEXCOORD0;
    float2 uv : TEXCOORD1;
}

[shader("vertex")]
VertexOutput vs_main(uniform PC pc, uint vert_id: SV_VertexID)
{
    float L = pc.LRTB.x;
    float R = pc.LRTB.y;
    float T = pc.LRTB.z;
    float B = pc.LRTB.w;
    float4x4 proj = transpose(float4x4(
        2.0 / (R - L), 0.0, 0.0, 0.0,
        0.0, 2.0 / (T - B), 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        (R + L) / (L - R), (T + B) / (B - T), 0.0, 1.0));
    Vertex v = pc.vertices[pc.base_vertex + vert_id];
    var out_color = unpackUnorm4x8ToFloat(v.rgba);
    var out_uv = float2(v.uv.x, v.uv.y);
    var out_position = mul(proj, float4(v.xy.x, v.xy.y, 0, 1));

    return VertexOutput(out_position, out_color, out_uv);
}

float3 srgb_to_linear(float3 c) {
    float3 lo = c / 12.92;
    float3 hi = pow((c + 0.055) / 1.055, 2.4);
    return lerp(lo, hi, step(0.04045, c));
}

[shader("fragment")]
float4 fs_main(VertexOutput input, uniform PC pc) : SV_Target0
{
    var sampled = sampled_2d[NonUniformResourceIndex(pc.texture_id)]
                          .Sample(samplers[NonUniformResourceIndex(pc.sampler_id)], input.uv);

    float4 colour = sampled * input.colour;
    colour.rgb = srgb_to_linear(colour.rgb);

    return colour;
}

