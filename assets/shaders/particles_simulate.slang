// Simulates the particles of one ParticleEmitter. The buffer is a ring: the `spawn_count` slots starting at `head`
// are (re)spawned this frame, every other live particle just integrates.
struct Particle
{
    // xyz: world position, w: seconds left to live (<= 0 is dead).
    float4 position_life;
    // xyz: velocity, w: the lifetime the particle was spawned with.
    float4 velocity_lifetime;
}

// Mirrors the Layout of SimulateParams in particle_system.cxx.
struct SimulatePushConstants
{
    Ptr<Particle> particles;
    uint count;
    uint head;
    uint spawn_count;
    uint seed;
    float delta_time;
    float lifetime;
    float gravity_x;
    float gravity_y;
    float gravity_z;
    uint shape;
    float shape_size;
    float speed;
    float speed_variance;
    float cone_radians;
    float position_x;
    float position_y;
    float position_z;
    float rotation_x;
    float rotation_y;
    float rotation_z;
    float rotation_w;
};

[vk::push_constant] SimulatePushConstants pc;

static const uint shape_point = 0u;
static const uint shape_sphere = 1u;
static const uint shape_box = 2u;
static const uint shape_cone = 3u;

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

float3 rotate(float3 v)
{
    float3 q = float3(pc.rotation_x, pc.rotation_y, pc.rotation_z);
    return v + 2.0 * cross(q, cross(q, v) + pc.rotation_w * v);
}

// A direction uniformly distributed on the sphere.
float3 sphere_direction(uint seed)
{
    float z = unit(seed) * 2.0 - 1.0;
    float angle = unit(seed + 1u) * 6.2831853;
    float r = sqrt(max(0.0, 1.0 - z * z));
    return float3(r * cos(angle), z, r * sin(angle));
}

// A direction within `cone_radians` of the emitter's +Y axis.
float3 cone_direction(uint seed)
{
    float cos_theta = lerp(1.0, cos(pc.cone_radians), unit(seed));
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float angle = unit(seed + 1u) * 6.2831853;
    return float3(sin_theta * cos(angle), cos_theta, sin_theta * sin(angle));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 dispatch_id: SV_DispatchThreadID)
{
    uint index = dispatch_id.x;
    if (index >= pc.count) {
        return;
    }

    Particle p = pc.particles[index];

    uint since_head = (index + pc.count - pc.head) % pc.count;
    if (since_head < pc.spawn_count) {
        uint seed = hash(index * 747796405u + pc.seed);

        float3 offset = float3(0.0);
        float3 direction = float3(0.0, 1.0, 0.0);

        if (pc.shape == shape_point) {
            direction = sphere_direction(seed + 2u);
        } else if (pc.shape == shape_sphere) {
            direction = sphere_direction(seed + 2u);
            offset = direction * pc.shape_size * pow(unit(seed + 4u), 1.0 / 3.0);
        } else if (pc.shape == shape_box) {
            offset = (float3(unit(seed + 2u), unit(seed + 3u), unit(seed + 4u)) * 2.0 - 1.0) * pc.shape_size;
            direction = cone_direction(seed + 5u);
        } else {
            direction = cone_direction(seed + 2u);
        }

        float speed = pc.speed * (1.0 + (unit(seed + 7u) * 2.0 - 1.0) * pc.speed_variance);
        float3 position = float3(pc.position_x, pc.position_y, pc.position_z) + rotate(offset);

        p.position_life = float4(position, pc.lifetime);
        p.velocity_lifetime = float4(rotate(direction) * speed, pc.lifetime);
    } else if (p.position_life.w > 0.0) {
        float3 velocity = p.velocity_lifetime.xyz + float3(pc.gravity_x, pc.gravity_y, pc.gravity_z) * pc.delta_time;

        p.position_life.xyz += velocity * pc.delta_time;
        p.position_life.w -= pc.delta_time;
        p.velocity_lifetime.xyz = velocity;
    }

    pc.particles[index] = p;
}
