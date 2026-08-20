struct Vertex {
    float3 position;
    uint rgba;
}

struct PC {
    float4x4 view_proj;
    Ptr<Vertex> vertices;
}

struct VertexOutput {
    float4 position : SV_Position;
    float4 colour : TEXCOORD0;
}

[shader("vertex")]
VertexOutput vs_main(uniform PC pc, uint vert_id : SV_VertexID)
{
    Vertex v = pc.vertices[vert_id];
    var out_colour = unpackUnorm4x8ToFloat(v.rgba);
    var out_position = mul(pc.view_proj, float4(v.position, 1.0));
    return VertexOutput(out_position, out_colour);
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target0
{
    return input.colour;
}