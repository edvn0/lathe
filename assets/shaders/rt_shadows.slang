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
import scene_types;

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

// Mirrors SceneAccelerationStructure::InstanceData.
struct RtInstanceData
{
    Ptr<CompressedVertex> vertices;
    Ptr<uint> indices;
    uint material_index;
    uint index_count;
    uint flags;
    uint reserved;
};

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
};

[vk::push_constant] PushConstants pc;

public [[vk::binding(6, 0)]] RaytracingAccelerationStructure scene_tlas;

static const uint shadow_rays = 4;

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

bool occluded(float3 origin, float3 direction)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 1.0e4;

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
    float lit = 0.0;
    if (pc.sun_spread > 0.0)
    {
        float angle_offset = gradient_noise(float2(pixel), pc.frame);
        float radial_offset = gradient_noise(float2(pixel).yx + 17.0, pc.frame);
        float centre = occluded(origin, pc.sun) ? 0.0 : 1.0;
        float2 u1 = float2((1.0 + radial_offset) / float(shadow_rays), frac(angle_offset + 0.618034));
        float edge = occluded(origin, cone_direction(pc.sun, pc.sun_spread, u1)) ? 0.0 : 1.0;
        lit = centre + edge;
        if (centre != edge)
        {
            for (uint i = 2; i < shadow_rays; ++i)
            {
                float2 u = float2((float(i) + radial_offset) / float(shadow_rays), frac(angle_offset + float(i) * 0.618034));
                lit += occluded(origin, cone_direction(pc.sun, pc.sun_spread, u)) ? 0.0 : 1.0;
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
        lit = occluded(origin, pc.sun) ? 0.0 : 1.0;
    }

    storage_2d[pc.result][thread_id.xy] = float4(lit);
}
