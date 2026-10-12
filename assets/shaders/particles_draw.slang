// Draws the particles of one ParticleEmitter as screen-facing squares inside the forward pass.
struct Particle
{
    float4 position_life;
    float4 velocity_lifetime;
}

// Mirrors DrawPush in particle_system.cxx.
struct DrawPushConstants
{
    float4x4 view_proj;
    Ptr<Particle> particles;
    float size_start;
    float size_end;
    float4 colour_start;
    float4 colour_end;
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

// Six vertices per particle. A dead particle collapses to a point and is dropped. The size is in world units: a
// view-space offset becomes a clip-space one through the projection's x and y scales, which are the lengths of the
// first two rows of the view-projection matrix (the view matrix is a rotation and a translation).
[shader("vertex")]
VertexOutput main_vs(uint vertex_id: SV_VertexID)
{
    Particle p = pc.particles[vertex_id / 6];
    float life = p.position_life.w;

    if (life <= 0.0) {
        return VertexOutput(float4(0.0, 0.0, 0.0, 0.0), float4(0.0));
    }

    float t = saturate(1.0 - life / max(p.velocity_lifetime.w, 1e-4));
    float size = lerp(pc.size_start, pc.size_end, t);

    float4 clip = mul(pc.view_proj, float4(p.position_life.xyz, 1.0));
    float2 scale = float2(length(pc.view_proj[0].xyz), length(pc.view_proj[1].xyz));
    clip.xy += corners[vertex_id % 6] * size * scale;

    return VertexOutput(clip, lerp(pc.colour_start, pc.colour_end, t));
}

// Drawn with source-alpha blending, depth tested and not written, after the opaque scene.
[shader("fragment")]
float4 main_fs(VertexOutput input) : SV_Target0
{
    return input.colour;
}
