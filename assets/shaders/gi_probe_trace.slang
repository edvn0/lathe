// Probe GI, phase 2 (docs/probe-gi.md): one ray per thread from a probe of the camera following grid, against the scene
// TLAS (binding 6, SceneAccelerationStructure), with VK_KHR_ray_query. Hits are shaded with the sun (a shadow ray) and the
// sky only, with a flat albedo; misses return the sky. Output is (radiance, signed hit distance) per ray, packed to
// 8 bytes; the distance is negative when the ray hit a back face, which is how gi_probe_classify finds probes inside
// geometry. Workgroup x is a slice of 64 rays, group id x the probe slot, group id y the slice.
import bindless;
import scene_types;
import gi_probes;

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
    Ptr<Material> materials;
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
static const float gi_sky_mip = 1.0;
static const float pi = 3.14159265358979;

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

float2 vertex_uv(Ptr<CompressedVertex> vertices, uint index)
{
    CompressedVertex v = vertices[index];
    return float2(f16tof32(v.texcoord_u), f16tof32(v.texcoord_v));
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

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random_unit(uint seed)
{
    return float(pcg(seed) & 0xFFFFFFu) / 16777216.0;
}

// A direction over the hemisphere of `normal`, cosine weighted, so a radiance sample along it estimates the
// irradiance (divided by pi) without any further weight.
float3 cosine_direction(float3 normal, float2 u)
{
    float3 t = normalize(abs(normal.x) > 0.9 ? cross(normal, float3(0.0, 1.0, 0.0)) : cross(normal, float3(1.0, 0.0, 0.0)));
    float3 b = cross(normal, t);
    float r = sqrt(u.x);
    float phi = 6.283185307179586 * u.y;
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + normal * sqrt(max(0.0, 1.0 - u.x)));
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
    int3 base = int3(pc.base_x, pc.base_y, pc.base_z);
    int3 icounts = int3(counts);
    int3 world_index = gi_world_index(slot, base, icounts);
    // A slot that was just given to another probe still holds the previous probe's offset until it is classified.
    float4 stored_index = pc.probe_state[slot * gi_state_stride + 2];
    float3 offset = all(int3(stored_index.xyz) == world_index) ? pc.probe_state[slot * gi_state_stride].xyz : float3(0.0);
    float3 origin = float3(world_index) * pc.spacing + offset;

    float3 direction = normalize(gi_rotate(pc.rotation, gi_fibonacci_direction(ray_index, pc.rays_per_probe)));
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

        // The surface's own colour, so light bounces off it tinted: the base colour factor times the texture, which is
        // read at a coarser level the farther the hit is (the ray carries no footprint).
        Material material = pc.materials[data.material_index];
        float2 bary = query.CommittedTriangleBarycentrics();
        uint i0 = data.indices[triangle * 3 + 0];
        uint i1 = data.indices[triangle * 3 + 1];
        uint i2 = data.indices[triangle * 3 + 2];
        float2 uv = vertex_uv(data.vertices, i0) * (1.0 - bary.x - bary.y) + vertex_uv(data.vertices, i1) * bary.x +
                    vertex_uv(data.vertices, i2) * bary.y;
        float lod = clamp(log2(max(t, 1.0)) + 2.0, 0.0, 12.0);
        float3 albedo = float3(material.colour[0], material.colour[1], material.colour[2]) *
                        sampled_2d[NonUniformResourceIndex(material.base_colour_texture)]
                                .SampleLevel(samplers[NonUniformResourceIndex(material.sampler_index)], uv, lod).rgb;

        // Sky light reaching the hit: one cosine weighted ray, so a hit under a roof or in a room gets none. The
        // probes' hysteresis averages these over frames.
        uint seed = slot * 977u + ray_index * 131u + asuint(pc.rotation.x);
        float3 sky_direction = cosine_direction(normal, float2(random_unit(seed), random_unit(seed ^ 0x9e3779b9u)));
        float3 sky_light = occluded(point + normal * bias, sky_direction, 1.0e4) ? float3(0.0) : sky(sky_direction, gi_sky_mip);

        float3 irradiance = sun_radiance * (n_dot_l * visible) + sky_light;
        radiance = albedo * irradiance;
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
