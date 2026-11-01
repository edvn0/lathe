// Probe GI (docs/probe-gi.md): one ray per thread from a probe of the camera following grid, against the scene TLAS
// (binding 6, SceneAccelerationStructure), with VK_KHR_ray_query. Output is (radiance, signed hit distance) per ray,
// packed to 8 bytes; the distance is negative when the ray hit a back face, which is how gi_probe_classify finds probes
// inside geometry. Workgroup x is a slice of 64 rays, group id x the probe slot, group id y the slice.
//
// What a hit radiates (units: pixel radiance, so a diffuse surface lit by E has albedo * E / pi):
//   - its own emission (emissive factor * strength * texture), unless the piece is an explicit emitter (below);
//   - albedo times the incoming light: the sun (shadow ray), the sky through one cosine weighted visibility ray, the
//     explicit emitters and the local lights through one stochastic shadow ray (a light picked in proportion to its
//     unshadowed contribution), and the probe volume itself as last frame left it (every ray that hits something asks
//     the atlases what lights that spot, which is what makes the volume multi-bounce);
//   - misses return the sky.
// Small emissive pieces (a bulb) are too small for random rays to find, so the renderer lists them as explicit emitters
// (spheres), and lights them like point lights.
import bindless;
import scene_types;
import gi_probes;

struct RtInstanceData
{
    Ptr<CompressedVertex> vertices;
    Ptr<uint> indices;
    uint material_index;
    uint index_count;
    uint flags;
    uint reserved;
};

// Mirrors ProbeVolume::Emitter.
struct GpuEmitter
{
    float3 centre;
    float radius;
    uint material_index;
    uint3 padding;
};

// Word offsets into the emitter buffer, mirroring ProbeVolume::grid_*_word.
static const uint emitter_grid_header = 4096 * 8;
static const uint emitter_grid_ranges = emitter_grid_header + 8;
static const uint emitter_grid_indices = emitter_grid_ranges + 16384 * 2;

GpuEmitter load_emitter(Ptr<uint> words, uint e)
{
    uint at = e * 8;
    GpuEmitter emitter;
    emitter.centre = float3(asfloat(words[at]), asfloat(words[at + 1]), asfloat(words[at + 2]));
    emitter.radius = asfloat(words[at + 3]);
    emitter.material_index = words[at + 4];
    emitter.padding = uint3(0);
    return emitter;
}

// Mirrors TracePush in probe_volume.cxx.
struct PushConstants
{
    Ptr<float4> probe_state;
    Ptr<uint2> ray_data;
    Ptr<RtInstanceData> instances;
    Ptr<Material> materials;
    Ptr<UBO> ubo;
    Ptr<uint> emitters; // the emitters and their grid, see ProbeVolume::grid_header_word
    Ptr<GpuLight> lights;
    int base_x;
    int base_y;
    int base_z;
    uint count_x;
    uint count_y;
    uint count_z;
    float spacing;
    uint rays_per_probe;
    float4 rotation;
    uint emitter_count;
    uint light_count;
    uint flags;
    float emitter_intensity;
    uint frame;
    uint sleep_params;
};

[vk::push_constant] PushConstants pc;

public [[vk::binding(6, 0)]] RaytracingAccelerationStructure scene_tlas;

static const float gi_max_distance = 500.0;
static const float gi_sky_mip = 1.0;
static const float pi = 3.14159265358979;
static const uint instance_flag_explicit_emitter = 1;

float3 rotate_y_vector(float3 v, float2 cos_sin)
{
    return float3(cos_sin.x * v.x + cos_sin.y * v.z, v.y, -cos_sin.y * v.x + cos_sin.x * v.z);
}

float3 sky(float3 direction, float mip)
{
    Ptr<UBO> ubo = pc.ubo;
    float3 d = rotate_y_vector(direction, float2(ubo.environment_rotation.x, -ubo.environment_rotation.y));
    return sampled_cube[ubo.prefilter_cube_texture].SampleLevel(samplers[ubo.environment_sampler], d, mip).rgb *
           ubo.environment_intensity.y;
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
    query.TraceRayInline(scene_tlas, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
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

float luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// The texture's average colour tints a material's emission (a coarse level of the texture).
float3 emission_of(Material material, float2 uv, float lod)
{
    float3 factor = float3(material.emissive[0], material.emissive[1], material.emissive[2]) * material.emissive_strength;
    if (factor.x + factor.y + factor.z <= 0.0)
    {
        return float3(0.0);
    }
    return factor * sampled_2d[NonUniformResourceIndex(material.emissive_texture)]
                            .SampleLevel(samplers[NonUniformResourceIndex(material.sampler_index)], uv, lod).rgb;
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

    Ptr<UBO> ubo = pc.ubo;

    // The probe in this slot is the one world index, inside the current window, that maps to it (toroidal addressing).
    uint3 counts = uint3(pc.count_x, pc.count_y, pc.count_z);
    int3 base = int3(pc.base_x, pc.base_y, pc.base_z);
    int3 icounts = int3(counts);
    int3 world_index = gi_world_index(slot, base, icounts);
    // A slot that was just given to another probe still holds the previous probe's offset until it is classified.
    float4 stored_index = pc.probe_state[slot * gi_state_stride + 2];
    if (gi_probe_skipped(pc.probe_state, slot, world_index, pc.frame, pc.sleep_params))
    {
        return;
    }
    float3 offset = all(int3(stored_index.xyz) == world_index) ? pc.probe_state[slot * gi_state_stride].xyz : float3(0.0);
    float3 origin = float3(world_index) * pc.spacing + offset;

    float3 direction = normalize(gi_rotate(pc.rotation, gi_fibonacci_direction(ray_index, pc.rays_per_probe)));
    float3 sun = ubo.light_direction;
    float3 sun_radiance = ubo.light_colour * ubo.light_intensity;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = gi_max_distance;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene_tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    query.Proceed();

    float3 radiance;
    float distance;
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        RtInstanceData data = pc.instances[query.CommittedInstanceID()];
        uint triangle = query.CommittedPrimitiveIndex();
        uint i0 = data.indices[triangle * 3 + 0];
        uint i1 = data.indices[triangle * 3 + 1];
        uint i2 = data.indices[triangle * 3 + 2];
        float3 a = vertex_position(data.vertices, i0);
        float3 b = vertex_position(data.vertices, i1);
        float3 c = vertex_position(data.vertices, i2);

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
        float3 shadow_origin = point + normal * bias;

        Material material = pc.materials[data.material_index];
        float2 bary = query.CommittedTriangleBarycentrics();
        float2 uv = vertex_uv(data.vertices, i0) * (1.0 - bary.x - bary.y) + vertex_uv(data.vertices, i1) * bary.x +
                    vertex_uv(data.vertices, i2) * bary.y;
        // The ray carries no footprint: the farther the hit, the coarser the texture level.
        float lod = clamp(log2(max(t, 1.0)) + 2.0, 0.0, 12.0);
        float3 albedo = float3(material.colour[0], material.colour[1], material.colour[2]) *
                        sampled_2d[NonUniformResourceIndex(material.base_colour_texture)]
                                .SampleLevel(samplers[NonUniformResourceIndex(material.sampler_index)], uv, lod).rgb;

        // Light reaching the hit, as the mean radiance over the cosine lobe (irradiance / pi).
        float3 incoming = float3(0.0);

        float n_dot_l = max(dot(normal, sun), 0.0);
        if (n_dot_l > 0.0 && !occluded(shadow_origin, sun, 1.0e4))
        {
            incoming += sun_radiance * (n_dot_l / pi);
        }

        // Sky light: one cosine weighted ray, so a hit under a roof or in a room gets none. Hysteresis averages these.
        uint seed = slot * 977u + ray_index * 131u + asuint(pc.rotation.x);
        float3 sky_direction = cosine_direction(normal, float2(random_unit(seed), random_unit(seed ^ 0x9e3779b9u)));
        if (!occluded(shadow_origin, sky_direction, 1.0e4))
        {
            incoming += sky(sky_direction, gi_sky_mip);
        }

        // Explicit emitters (spheres) and local lights: pick one in proportion to its unshadowed contribution, trace a
        // shadow ray to it, and weigh it by the sum over the picked one's share.
        float3 selected_contribution = float3(0.0);
        float3 selected_target = float3(0.0);
        float selected_distance = 0.0;
        float total_weight = 0.0;
        float selected_weight = 0.0;
        uint pick_seed = seed * 2654435761u + 12345u;
        // Only the emitters of the hit's grid cell can reach it.
        uint cell_first = 0;
        uint cell_count = 0;
        if (pc.emitter_count > 0)
        {
            Ptr<uint> words = pc.emitters;
            float3 grid_origin = float3(asfloat(words[emitter_grid_header]), asfloat(words[emitter_grid_header + 1]),
                                        asfloat(words[emitter_grid_header + 2]));
            float grid_scale = asfloat(words[emitter_grid_header + 3]);
            int3 grid_dims = int3(int(words[emitter_grid_header + 4]), int(words[emitter_grid_header + 5]),
                                  int(words[emitter_grid_header + 6]));
            int3 cell = int3(floor((point - grid_origin) * grid_scale));
            if (words[emitter_grid_header + 7] != 0 && all(cell >= 0) && all(cell < grid_dims))
            {
                uint cell_index = uint(cell.x + grid_dims.x * (cell.y + grid_dims.y * cell.z));
                cell_first = words[emitter_grid_ranges + cell_index * 2];
                cell_count = words[emitter_grid_ranges + cell_index * 2 + 1];
            }
        }
        for (uint k = 0; k < cell_count; ++k)
        {
            GpuEmitter emitter = load_emitter(pc.emitters, pc.emitters[emitter_grid_indices + cell_first + k]);
            float3 to_emitter = emitter.centre - point;
            float dist = length(to_emitter);
            float range = max(80.0 * emitter.radius, 1.0);
            if (dist >= range)
            {
                continue;
            }
            float3 l = to_emitter / max(dist, 1e-6);
            float cos_theta = max(dot(normal, l), 0.0);
            if (cos_theta <= 0.0)
            {
                continue;
            }
            // Treated as a point light (like the point lights the loader synthesises from emissive pieces): a bulb's
            // physical emission is far too small to light anything, so its colour is the average emission colour and
            // its strength a setting. Same falloff as the forward shader's point lights.
            Material emitter_material = pc.materials[emitter.material_index];
            float3 emitted = emission_of(emitter_material, float2(0.5), 12.0);
            float peak = max(max(emitted.x, emitted.y), emitted.z);
            if (peak <= 0.0)
            {
                continue;
            }
            float falloff = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
            falloff = falloff * falloff / max(dist * dist, 1e-4);
            float3 contribution = (emitted / peak) * pc.emitter_intensity * falloff * (cos_theta / pi);
            float w = luminance(contribution);
            if (w <= 1e-6)
            {
                continue;
            }
            total_weight += w;
            pick_seed = pcg(pick_seed);
            if (float(pick_seed & 0xFFFFFFu) / 16777216.0 * total_weight < w)
            {
                selected_contribution = contribution;
                selected_target = emitter.centre;
                selected_distance = sqrt(dot(to_emitter, to_emitter));
                selected_weight = w;
            }
        }
        for (uint li = 0; li < pc.light_count; ++li)
        {
            GpuLight light = pc.lights[li];
            float3 to_light = light.position - point;
            float dist = length(to_light);
            if (dist >= light.range)
            {
                continue;
            }
            float3 l = to_light / max(dist, 1e-6);
            float cos_theta = max(dot(normal, l), 0.0);
            if (cos_theta <= 0.0)
            {
                continue;
            }
            float falloff = clamp(1.0 - pow(dist / max(light.range, 1e-4), 4.0), 0.0, 1.0);
            falloff = falloff * falloff / max(dist * dist, 1e-4);
            if (light.type == LightType.spot)
            {
                float cd = dot(normalize(light.direction), -l);
                float cone = clamp(cd * light.spot_scale + light.spot_offset, 0.0, 1.0);
                falloff *= cone * cone;
            }
            float3 contribution = light.colour * light.intensity * falloff * (cos_theta / pi);
            float w = luminance(contribution);
            if (w <= 1e-6)
            {
                continue;
            }
            total_weight += w;
            pick_seed = pcg(pick_seed);
            if (float(pick_seed & 0xFFFFFFu) / 16777216.0 * total_weight < w)
            {
                selected_contribution = contribution;
                selected_target = light.position;
                selected_distance = dist;
                selected_weight = w;
            }
        }
        if (selected_weight > 0.0)
        {
            float3 to_target = selected_target - shadow_origin;
            float target_distance = length(to_target);
            // Stop a little short of the emitter itself so its own geometry does not shadow it.
            if (!occluded(shadow_origin, to_target / max(target_distance, 1e-6), max(target_distance - 0.15, 0.0)))
            {
                incoming += selected_contribution * (total_weight / selected_weight);
            }
        }

        // Light that already bounced around: the probe volume as last frame left it.
        // (A cascade bounces on its own light: the coarse one has to stand without the near one's atlases.)
        GpuGiBlock own = (pc.flags & 1) != 0 ? ubo.gi_far : ubo.gi;
        if ((own.flags & gi_flag_enabled) != 0)
        {
            float coverage;
            float3 indirect = gi_sample_block(ubo, own, point, normal, -direction, coverage);
            incoming += indirect * coverage;
        }

        float3 emission = (data.flags & instance_flag_explicit_emitter) != 0 ? float3(0.0) : emission_of(material, uv, lod);
        radiance = albedo * incoming + emission;
        distance = back_face ? -t : t;
    }
    else
    {
        radiance = sky(direction, gi_sky_mip);
        distance = gi_max_distance;
    }

    // A NaN here would poison the atlases for good (hysteresis keeps a share of the old value).
    if (any(isnan(radiance)) || any(isinf(radiance)))
    {
        radiance = float3(0.0);
    }
    uint index = slot * pc.rays_per_probe + ray_index;
    uint rg = f32tof16(radiance.r) | (f32tof16(radiance.g) << 16);
    uint bd = f32tof16(radiance.b) | (f32tof16(distance) << 16);
    pc.ray_data[index] = uint2(rg, bd);
}
