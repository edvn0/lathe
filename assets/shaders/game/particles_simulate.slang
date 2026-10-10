struct Particle
{
    float4 position_life;
    float4 velocity_size;
}

struct SimulatePushConstants
{
    Ptr<Particle> particles;
    uint count;
    uint frame;
    float delta_time;
    float emitter_x;
    float emitter_y;
    float emitter_z;
};

[vk::push_constant] SimulatePushConstants pc;

uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float unit(uint x)
{
    return float(hash(x) & 0xffffffu) / 16777215.0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 dispatch_id: SV_DispatchThreadID)
{
    if (dispatch_id.x >= pc.count) {
        return;
    }

    Particle p = pc.particles[dispatch_id.x];

    if (p.position_life.w <= 0.0) {
        uint seed = hash(dispatch_id.x * 747796405u + pc.frame * 2891336453u);
        float3 emitter = float3(pc.emitter_x, pc.emitter_y, pc.emitter_z);
        float3 offset = float3(unit(seed) - 0.5, 0.0, unit(seed + 1u) - 0.5) * 2.0;
        float3 velocity = float3(unit(seed + 2u) - 0.5, 0.0, unit(seed + 3u) - 0.5) * 3.0;
        velocity.y = 4.0 + unit(seed + 4u) * 4.0;

        p.position_life = float4(emitter + offset, 1.0 + unit(seed + 5u) * 2.0);
        p.velocity_size = float4(velocity, 0.004 + unit(seed + 6u) * 0.006);
    } else {
        float3 velocity = p.velocity_size.xyz;
        velocity.y -= 9.81 * pc.delta_time;

        p.position_life.xyz += velocity * pc.delta_time;
        p.position_life.w -= pc.delta_time;
        p.velocity_size.xyz = velocity;
    }

    pc.particles[dispatch_id.x] = p;
}
