// Probe GI, phase 2 (docs/probe-gi.md): one ray per thread from a probe of the camera following grid, against the scene
// TLAS (binding 6, SceneAccelerationStructure), with VK_KHR_ray_query. Hits are shaded with the sun (a shadow ray) and the
// sky only, with a flat albedo; misses return the sky. Output is (radiance, signed hit distance) per ray, packed to
// 8 bytes; the distance is negative when the ray hit a back face, which is how gi_probe_classify finds probes inside
// geometry. Workgroup x is a slice of 64 rays, group id x the probe slot, group id y the slice.
import bindless;
import scene_types;

struct RtInstanceData
{
    Ptr<CompressedVertex> vertices;
    Ptr<uint> indices;
    uint material_index;
    uint index_count;
    uint2 reserved;
};

// Mirrors TraceParams in probe_volume.cxx.
struct PushConstants
{
    Ptr<float4> probe_state;
    Ptr<uint2> ray_data;
    Ptr<RtInstanceData> instances;
    int base_x;
    int base_y;
    int base_z;
    uint count_x;
    uint count_y;
    uint count_z;
    float spacing;
    uint rays_per_probe;
    float4 rotation;
    float sun_x;
    float sun_y;
    float sun_z;
    float sun_r;
    float sun_g;
    float sun_b;
    uint sky_cube;
    uint sky_sampler;
    float env_cos;
    float env_sin;
    float sky_intensity;
};

[vk::push_constant] PushConstants pc;

public [[vk::binding(6, 0)]] RaytracingAccelerationStructure scene_tlas;

static const float gi_max_distance = 500.0;
static const float gi_albedo = 0.5;
static const float gi_sky_mip = 1.0;
static const float gi_sky_irradiance_mip = 4.0;
static const float pi = 3.14159265358979;

float3 rotate_by_quaternion(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Fibonacci sphere: evenly spread directions for any count.
float3 fibonacci_direction(uint index, uint count)
{
    float z = 1.0 - (2.0 * float(index) + 1.0) / float(count);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.399963229728653 * float(index);
    return float3(r * cos(phi), r * sin(phi), z);
}

float3 sky(float3 direction, float mip)
{
    float3 d = float3(pc.env_cos * direction.x + pc.env_sin * direction.z, direction.y,
                      -pc.env_sin * direction.x + pc.env_cos * direction.z);
    return sampled_cube[pc.sky_cube].SampleLevel(samplers[pc.sky_sampler], d, mip).rgb * pc.sky_intensity;
}

float3 vertex_position(Ptr<CompressedVertex> vertices, uint index)
{
    CompressedVertex v = vertices[index];
    return float3(f16tof32(v.position_x), f16tof32(v.position_y), f16tof32(v.position_z));
}

bool occluded(float3 origin, float3 direction, float t_max)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = t_max;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene_tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    query.Proceed();

    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 group_id : SV_GroupID, uint3 local_id : SV_GroupThreadID)
{
    uint slot = group_id.x;
    uint ray_index = group_id.y * 64 + local_id.x;
    if (ray_index >= pc.rays_per_probe)
    {
        return;
    }

    // The probe in this slot is the one world index, inside the current window, that maps to it (toroidal addressing).
    uint3 counts = uint3(pc.count_x, pc.count_y, pc.count_z);
    int3 slot3 = int3(slot % counts.x, (slot / counts.x) % counts.y, slot / (counts.x * counts.y));
    int3 base = int3(pc.base_x, pc.base_y, pc.base_z);
    int3 icounts = int3(counts);
    int3 relative = ((slot3 - base) % icounts + icounts) % icounts;
    float3 origin = float3(base + relative) * pc.spacing + pc.probe_state[slot * 2].xyz;

    float3 direction = normalize(rotate_by_quaternion(pc.rotation, fibonacci_direction(ray_index, pc.rays_per_probe)));
    float3 sun = float3(pc.sun_x, pc.sun_y, pc.sun_z);
    float3 sun_radiance = float3(pc.sun_r, pc.sun_g, pc.sun_b);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = gi_max_distance;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();

    float3 radiance;
    float distance;
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        RtInstanceData data = pc.instances[query.CommittedInstanceID()];
        uint triangle = query.CommittedPrimitiveIndex();
        float3 a = vertex_position(data.vertices, data.indices[triangle * 3 + 0]);
        float3 b = vertex_position(data.vertices, data.indices[triangle * 3 + 1]);
        float3 c = vertex_position(data.vertices, data.indices[triangle * 3 + 2]);

        float3x4 world_to_object = query.CommittedWorldToObject3x4();
        float3x3 linear = float3x3(world_to_object[0].xyz, world_to_object[1].xyz, world_to_object[2].xyz);
        float3 normal = normalize(mul(cross(b - a, c - a), linear));

        // A face seen from behind is the sign of the probe sitting inside geometry. The sign is of the winding, which
        // is as good as the mesh is consistent; classification only needs a fraction.
        bool back_face = dot(normal, direction) > 0.0;
        if (back_face)
        {
            normal = -normal;
        }

        float t = query.CommittedRayT();
        float3 point = origin + direction * t;
        float bias = 0.002 + 0.0005 * t;
        float n_dot_l = max(dot(normal, sun), 0.0);
        float visible = n_dot_l > 0.0 && !occluded(point + normal * bias, sun, 1.0e4) ? 1.0 : 0.0;

        float3 irradiance = sun_radiance * (n_dot_l * visible) + sky(normal, gi_sky_irradiance_mip);
        radiance = gi_albedo * irradiance;
        distance = back_face ? -t : t;
    }
    else
    {
        radiance = sky(direction, gi_sky_mip);
        distance = gi_max_distance;
    }

    uint index = slot * pc.rays_per_probe + ray_index;
    uint rg = f32tof16(radiance.r) | (f32tof16(radiance.g) << 16);
    uint bd = f32tof16(radiance.b) | (f32tof16(distance) << 16);
    pc.ray_data[index] = uint2(rg, bd);
}
