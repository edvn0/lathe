struct Particle
{
    float4 position_life;
    float4 velocity_size;
}

struct DrawPushConstants
{
    float4x4 view_proj;
    Ptr<Particle> particles;
};

[vk::push_constant] DrawPushConstants pc;

struct VertexOutput
{
    float4 position : SV_Position;
    float4 colour : TEXCOORD0;
}

static const float2 corners[6] = {
    float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
    float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0),
};

// Six vertices per particle: a small screen-facing square. A dead particle collapses to a point and is dropped.
[shader("vertex")]
VertexOutput main_vs(uint vertex_id: SV_VertexID)
{
    Particle p = pc.particles[vertex_id / 6];
    float life = p.position_life.w;

    if (life <= 0.0) {
        return VertexOutput(float4(0.0, 0.0, 0.0, 0.0), float4(0.0));
    }

    float4 clip = mul(pc.view_proj, float4(p.position_life.xyz, 1.0));
    clip.xy += corners[vertex_id % 6] * p.velocity_size.w * clip.w;

    float fade = saturate(life);
    return VertexOutput(clip, float4(1.0, 0.45 + 0.4 * fade, 0.1, 1.0));
}

[shader("fragment")]
float4 main_fs(VertexOutput input) : SV_Target0
{
    return input.colour;
}
