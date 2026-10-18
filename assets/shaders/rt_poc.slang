// Ray query proof of concept (docs/ray-tracing.md). One compute pass, no ray tracing pipeline: it traces the camera's
// rays against a one-mesh TLAS, drawing the mesh into the scene colour, and traces a ray from every scene pixel towards
// the sun, darkening the pixels the mesh shadows.
import bindless;

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

// Mirrors RayTracingPoc::Push in rt_poc.cxx.
struct PushConstants
{
    float4x4 inverse_view_projection;
    Ptr<float> vertices;
    Ptr<uint> indices;
    uint hdr;
    uint depth;
    uint result;
    float sun_x;
    float sun_y;
    float sun_z;
    uint mode;
};

[vk::push_constant] PushConstants pc;

// The top level acceleration structure, bound once in the bindless set by RayTracingPoc::create.
public [[vk::binding(6, 0)]] RaytracingAccelerationStructure scene_tlas;

float3 vertex(uint index)
{
    return float3(pc.vertices[index * 3 + 0], pc.vertices[index * 3 + 1], pc.vertices[index * 3 + 2]);
}

// Debug views, chosen with --rt-poc-view (RayTracingPoc::view_names lists them).
static const uint view_beauty = 0;
static const uint view_normals = 1;
static const uint view_distance = 2;
static const uint view_shadow_mask = 3;
static const uint view_triangle = 4;
static const uint view_barycentrics = 5;
static const uint view_depth_test = 6;

float3 hash_colour(uint id)
{
    uint h = id * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return float3(h & 255u, (h >> 8u) & 255u, (h >> 16u) & 255u) / 255.0;
}

// Blue at 0, green at 0.5, red at 1.
float3 heat(float t)
{
    return saturate(float3(2.0 * t - 1.0, 1.0 - abs(2.0 * t - 1.0), 1.0 - 2.0 * t));
}

float3 unproject(float2 ndc, float ndc_z)
{
    float4 world = mul(pc.inverse_view_projection, float4(ndc, ndc_z, 1.0));
    return world.xyz / world.w;
}

// True when anything in the TLAS is between `origin` and `origin + direction * t_max`.
bool occluded(RaytracingAccelerationStructure tlas, float3 origin, float3 direction, float t_max)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.02;
    ray.TMax = t_max;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    query.Proceed();

    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID)
{
    uint width, height;
    sampled_2d[pc.hdr].GetDimensions(width, height);

    if (thread_id.x >= width || thread_id.y >= height)
    {
        return;
    }

    float4 scene = sampled_2d[pc.hdr].Load(int3(thread_id.xy, 0));
    float2 uv = (float2(thread_id.xy) + 0.5) / float2(width, height);
    // The renderer draws with a negative height viewport, so NDC y points up while the pixel row points down.
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // The depth buffer is reversed: 1 at the near plane, 0 at the far plane and the sky. NDC z is 1 - depth.
    float scene_depth = sampled_2d_depth[pc.depth].Load(int3(thread_id.xy, 0));
    float3 sun = float3(pc.sun_x, pc.sun_y, pc.sun_z);
    RaytracingAccelerationStructure tlas = scene_tlas;

    float3 near_point = unproject(ndc, 0.0);
    float3 ray_direction = normalize(unproject(ndc, 1.0) - near_point);
    float scene_distance = 1.0e30;
    bool scene_shadowed = false;

    if (scene_depth > 0.0)
    {
        float3 scene_point = unproject(ndc, 1.0 - scene_depth);
        scene_distance = length(scene_point - near_point);

        // Shadow ray from the scene surface towards the sun (TMin keeps it off the surface).
        scene_shadowed = occluded(tlas, scene_point, sun, 1.0e4);
    }

    // Primary ray: the mesh, if it is in front of the scene.
    RayDesc ray;
    ray.Origin = near_point;
    ray.Direction = ray_direction;
    ray.TMin = 0.0;
    ray.TMax = scene_distance;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();
    bool hit = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;

    // The scene, dimmed, for the views that draw something over it.
    float3 backdrop = scene.rgb * 0.25;
    float3 colour = scene.rgb;

    if (hit)
    {
        // The instance is a translation and a uniform scale, so the object space face normal is the world one.
        uint triangle = query.CommittedPrimitiveIndex();
        float3 a = vertex(pc.indices[triangle * 3 + 0]);
        float3 b = vertex(pc.indices[triangle * 3 + 1]);
        float3 c = vertex(pc.indices[triangle * 3 + 2]);
        float3 normal = normalize(cross(b - a, c - a));

        if (dot(normal, ray_direction) > 0.0)
        {
            normal = -normal;
        }

        float t = query.CommittedRayT();
        float3 point = near_point + ray_direction * t;
        float shade = max(dot(normal, sun), 0.0);
        bool lit = !occluded(tlas, point + normal * 0.02, sun, 1.0e4);
        float visibility = lit ? 1.0 : 0.0;

        switch (pc.mode)
        {
        case view_normals:
            colour = normal * 0.5 + 0.5;
            break;
        case view_distance:
            // Blue (near) to red (30 m), with a darker line every metre.
            colour = heat(saturate(t / 30.0)) * (frac(t) < 0.06 ? 0.4 : 1.0);
            break;
        case view_shadow_mask:
            colour = float3(shade > 0.0 ? visibility : 0.0);
            break;
        case view_triangle:
            colour = hash_colour(triangle);
            break;
        case view_barycentrics:
        {
            float2 bary = query.CommittedRayBarycentrics();
            colour = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
            break;
        }
        case view_depth_test:
            colour = float3(0.1, 0.9, 0.2);
            break;
        default:
            colour = float3(0.9, 0.55, 0.2) * (0.15 + 3.0 * shade * visibility);
            break;
        }
    }
    else
    {
        switch (pc.mode)
        {
        case view_normals:
        case view_distance:
        case view_triangle:
        case view_barycentrics:
            colour = backdrop;
            break;
        case view_shadow_mask:
            // White: the sun reaches the surface. Black: the sphere is in the way. Grey: sky, nothing to shade.
            colour = scene_depth > 0.0 ? float3(scene_shadowed ? 0.0 : 1.0) : float3(0.5);
            break;
        case view_depth_test:
        {
            // Red: the sphere is on this ray but behind the scene, so the depth test rejects it.
            RayDesc unbounded = ray;
            unbounded.TMax = 1.0e30;
            RayQuery<RAY_FLAG_NONE> behind;
            behind.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, unbounded);
            behind.Proceed();
            colour = behind.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? float3(0.9, 0.1, 0.1) : backdrop;
            break;
        }
        default:
            colour = scene.rgb * (scene_shadowed ? 0.35 : 1.0);
            break;
        }
    }

    storage_2d[pc.result][thread_id.xy] = float4(colour, scene.a);
}
