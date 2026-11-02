// Ray traced sun shadows (docs/ray-traced-shadows.md): one compute pass between the depth prepass and the forward pass.
// Each pixel's surface point is rebuilt from the depth buffer and a ray is traced from it toward the sun against the
// scene TLAS (SceneAccelerationStructure, bound in the bindless set at binding 6). The result, 1 lit and 0 shadowed, is a
// mask the forward shader multiplies the sun by where the cascaded shadow map used to be.
//
// The ray leaves the sun's disc at a random point (a small cone), so shadows have a contact-hardening penumbra; the
// forward shader averages the mask over a pixel and its neighbours. Alpha tested geometry (instances marked
// flag_alpha_tested, not opaque in the TLAS) is walked through ray query candidates: a candidate counts as a blocker only
// where the material's alpha test passes, so leaves cast the shadow of their leaves, not of their cards.
import bindless;
import common;
import ray_tracing;
import scene_types;
import light_clusters;


// Mirrors Push in ray_traced_shadows.cxx.
struct PushConstants
{
    float4x4 inverse_view_projection;
    Ptr<RtInstanceData> instances;
    Ptr<Material> materials;
    uint depth;
    uint result;
    uint width;
    uint height;
    float3 sun;
    // Half the angle of the cone the sun's rays are drawn from, in radians (0 gives hard shadows).
    float sun_spread;
    uint frame;
    uint light_count;
    Ptr<UBO> ubo;
    Ptr<GpuLight> lights;
    // The per cluster light lists, as the forward pass reads them (after the stats).
    Ptr<uint> cluster_lights;
    // Bit 0: trace the sun. Bit 1: trace local lights.
    uint flags;
};

static const uint flag_sun = 1u;
static const uint flag_local = 2u;

[vk::push_constant] PushConstants pc;

static const uint shadow_rays = 4;

// The renderer draws with a negative height viewport, so NDC y points up while the pixel row points down; the depth
// buffer is reversed (1 at the near plane, 0 at the far plane and the sky), so NDC z is 1 - depth.
float3 unproject(int2 pixel, float depth)
{
    float2 uv = (float2(pixel) + 0.5) / float2(pc.width, pc.height);
    float4 world = mul(pc.inverse_view_projection, float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0 - depth, 1.0));
    return world.xyz / world.w;
}

float depth_at(int2 pixel)
{
    int2 clamped = clamp(pixel, int2(0), int2(int(pc.width) - 1, int(pc.height) - 1));
    return sampled_2d_depth[pc.depth].Load(int3(clamped, 0));
}

// Interleaved gradient noise (Jimenez): cheap, well spread over a pixel's neighbours.
float gradient_noise(float2 pixel, uint frame)
{
    pixel += 5.588238 * float(frame & 63u);
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

// A cone of directions around `axis`, `spread` radians across, drawn with two noise values.
float3 cone_direction(float3 axis, float spread, float2 u)
{
    float3 up = abs(axis.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, axis));
    float3 bitangent = cross(axis, tangent);
    float radius = spread * sqrt(u.x);
    float angle = 6.28318530718 * u.y;
    return normalize(axis + (tangent * cos(angle) + bitangent * sin(angle)) * radius);
}

// Whether the material lets this candidate through: the alpha test of a leaf card.
bool alpha_blocks(RayQuery<RAY_FLAG_NONE> query)
{
    RtInstanceData data = pc.instances[query.CandidateInstanceID()];
    uint triangle = query.CandidatePrimitiveIndex();
    uint i0 = data.indices[triangle * 3 + 0];
    uint i1 = data.indices[triangle * 3 + 1];
    uint i2 = data.indices[triangle * 3 + 2];
    float2 bary = query.CandidateTriangleBarycentrics();
    float2 uv = vertex_uv(data.vertices, i0) * (1.0 - bary.x - bary.y) + vertex_uv(data.vertices, i1) * bary.x +
                vertex_uv(data.vertices, i2) * bary.y;
    Material material = pc.materials[data.material_index];
    float lod = clamp(log2(max(query.CandidateTriangleRayT(), 1.0)), 0.0, 6.0);
    float alpha = material.colour[3] *
                  sampled_2d[NonUniformResourceIndex(material.base_colour_texture)]
                          .SampleLevel(samplers[NonUniformResourceIndex(material.sampler_index)], uv, lod).a;
    return alpha >= material.alpha_cutoff;
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
    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE && alpha_blocks(query))
        {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

// Visibility of one light from the point: 1 lit (or out of reach, which shades to nothing anyway), 0 blocked.
float local_light_visibility(GpuLight light, float3 origin)
{
    float3 to_light = light.position - origin;
    float distance = length(to_light);
    if (distance >= light.range)
    {
        return 1.0;
    }
    float3 direction = to_light / max(distance, 1e-6);
    if (light.type == LightType.spot && dot(normalize(light.direction), -direction) * light.spot_scale + light.spot_offset <= 0.0)
    {
        return 1.0;
    }
    return occluded(origin, direction, max(distance - 0.02, 0.0)) ? 0.0 : 1.0;
}

float3 trace_local_lights(float3 p, float3 normal, float bias)
{
    const Ptr<UBO> ubo = pc.ubo;
    float3 origin = p + normal * bias;
    float visibility[rt_local_shadow_slots] = { 1.0, 1.0, 1.0 };
    uint slot = 0;

    if (ubo.clustered_lighting != 0)
    {
        float view_z = mul(ubo.view, float4(p, 1.0)).z;
        uint cluster = cluster_index(ubo, mul(ubo.view_projection, float4(p, 1.0)), view_z);
        uint capacity = ubo.cluster_light_capacity;
        uint stored_count = min(pc.cluster_lights[cluster], capacity);
        uint list_base = cluster_count(ubo) + cluster * capacity;
        for (uint i = 0; i < stored_count && slot < rt_local_shadow_slots; ++i)
        {
            GpuLight light = pc.lights[pc.cluster_lights[list_base + i]];
            if (light.shadow_tile != no_shadow_tile)
            {
                visibility[slot++] = local_light_visibility(light, origin);
            }
        }
    }
    else
    {
        for (uint i = 0; i < pc.light_count && slot < rt_local_shadow_slots; ++i)
        {
            GpuLight light = pc.lights[i];
            if (light.shadow_tile != no_shadow_tile)
            {
                visibility[slot++] = local_light_visibility(light, origin);
            }
        }
    }
    return float3(visibility[0], visibility[1], visibility[2]);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID)
{
    if (thread_id.x >= pc.width || thread_id.y >= pc.height)
    {
        return;
    }
    int2 pixel = int2(thread_id.xy);

    float depth = depth_at(pixel);
    if (depth <= 0.0)
    {
        // Sky: nothing to shadow.
        storage_2d[pc.result][thread_id.xy] = float4(1.0);
        return;
    }

    float3 p = unproject(pixel, depth);
    float3 eye = unproject(pixel, 1.0);

    // The face normal from the neighbours' positions, taking on each axis the side whose depth is closer so a silhouette
    // does not tilt the normal.
    float3 p_l = unproject(pixel + int2(-1, 0), max(depth_at(pixel + int2(-1, 0)), 1e-6));
    float3 p_r = unproject(pixel + int2(1, 0), max(depth_at(pixel + int2(1, 0)), 1e-6));
    float3 p_u = unproject(pixel + int2(0, -1), max(depth_at(pixel + int2(0, -1)), 1e-6));
    float3 p_d = unproject(pixel + int2(0, 1), max(depth_at(pixel + int2(0, 1)), 1e-6));
    float3 tangent_x = length(p - p_l) < length(p_r - p) ? p - p_l : p_r - p;
    float3 tangent_y = length(p - p_u) < length(p_d - p) ? p - p_u : p_d - p;
    float3 normal = normalize(cross(tangent_x, tangent_y));
    if (dot(normal, p - eye) > 0.0)
    {
        normal = -normal;
    }

    // The offset grows with distance: the position is only as exact as the depth buffer, and the TLAS has half precision
    // vertices.
    float distance = length(p - eye);
    float bias = 0.01 + 0.0015 * distance;
    float3 origin = p + normal * bias + pc.sun * (bias * 0.5);

    // Rays through the sun's disc, one in each quarter of the cone's radius (stratified) turned by a per pixel angle. Two
    // go first, the middle of the disc and a random point: where they agree (almost every pixel, lit or shadowed) that is
    // the answer; where they do not the pixel is in a penumbra and two more settle it into 0, 1/4 ... 1.
    float lit = 1.0;
    if ((pc.flags & flag_sun) == 0)
    {
        // The sun is off (no sun, or only local lights are traced): nothing to shadow.
    }
    else if (pc.sun_spread > 0.0)
    {
        float angle_offset = gradient_noise(float2(pixel), pc.frame);
        float radial_offset = gradient_noise(float2(pixel).yx + 17.0, pc.frame);
        float centre = occluded(origin, pc.sun, 1.0e4) ? 0.0 : 1.0;
        float2 u1 = float2((1.0 + radial_offset) / float(shadow_rays), frac(angle_offset + 0.618034));
        float edge = occluded(origin, cone_direction(pc.sun, pc.sun_spread, u1), 1.0e4) ? 0.0 : 1.0;
        lit = centre + edge;
        if (centre != edge)
        {
            for (uint i = 2; i < shadow_rays; ++i)
            {
                float2 u = float2((float(i) + radial_offset) / float(shadow_rays), frac(angle_offset + float(i) * 0.618034));
                lit += occluded(origin, cone_direction(pc.sun, pc.sun_spread, u), 1.0e4) ? 0.0 : 1.0;
            }
            lit /= float(shadow_rays);
        }
        else
        {
            lit *= 0.5;
        }
    }
    else
    {
        lit = occluded(origin, pc.sun, 1.0e4) ? 0.0 : 1.0;
    }

    // Local lights: one ray from the point to each shadow casting light of the pixel's cluster, the first three in list
    // order into G, B and A. forward_geom.slang counts the same lights the same way.
    float3 local_visibility = float3(1.0);
    if ((pc.flags & flag_local) != 0)
    {
        local_visibility = trace_local_lights(p, normal, bias);
    }

    storage_2d[pc.result][thread_id.xy] = float4(lit, local_visibility);
}
