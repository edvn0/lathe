// Ray query checks of the real scene acceleration structure (docs/ray-tracing.md, docs/probe-gi.md phase 1). One compute
// pass, no ray tracing pipeline: a ray per pixel from the camera against the scene TLAS (bound in the bindless set,
// binding 6, by SceneAccelerationStructure), and from the hit a ray towards the sun.
//
// Views (RayTracingPoc::view_names):
//   beauty       the scene colour untouched (the shadow map shadows)
//   shadow-mask  ray traced sun visibility of the primary hit: white lit, black shadowed, grey no hit
//   split        left half: shadow map beauty; right half: the ray traced shadow mask
//   depth-match  green where the primary ray hits at the depth buffer's distance, red where it does not, grey sky
//   normals      world space face normal of the primary hit (instance transform, scale included)
//   instance     a colour per TLAS instance
//   overlay      beauty darkened where the ray traced sun ray is blocked
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
    uint2 reserved;
};

// Mirrors Params in ray_tracing_poc.cxx.
struct PushConstants
{
    float4x4 inverse_view_projection;
    Ptr<RtInstanceData> instances;
    uint hdr;
    uint depth;
    uint result;
    float sun_x;
    float sun_y;
    float sun_z;
    uint mode;
};

[vk::push_constant] PushConstants pc;

public [[vk::binding(6, 0)]] RaytracingAccelerationStructure scene_tlas;

static const uint rt_view_beauty = 0;
static const uint rt_view_shadow_mask = 1;
static const uint rt_view_split = 2;
static const uint rt_view_depth_match = 3;
static const uint rt_view_normals = 4;
static const uint rt_view_instance = 5;
static const uint rt_view_overlay = 6;

float3 hash_colour(uint id)
{
    uint h = id * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return float3(h & 255u, (h >> 8u) & 255u, (h >> 16u) & 255u) / 255.0;
}

float3 unproject(float2 ndc, float ndc_z)
{
    float4 world = mul(pc.inverse_view_projection, float4(ndc, ndc_z, 1.0));
    return world.xyz / world.w;
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

    float3 near_point = unproject(ndc, 0.0);
    float3 ray_direction = normalize(unproject(ndc, 1.0) - near_point);
    float scene_distance = 0.0;
    if (scene_depth > 0.0)
    {
        scene_distance = length(unproject(ndc, 1.0 - scene_depth) - near_point);
    }

    RayDesc ray;
    ray.Origin = near_point;
    ray.Direction = ray_direction;
    ray.TMin = 0.0;
    ray.TMax = 1.0e5;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();
    bool hit = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;

    float3 colour = scene.rgb;
    float3 mask = float3(0.5);
    float3 normal = float3(0.0);
    float t = 0.0;

    if (hit)
    {
        uint instance = query.CommittedInstanceID();
        uint triangle = query.CommittedPrimitiveIndex();
        RtInstanceData data = pc.instances[instance];

        float3 a = vertex_position(data.vertices, data.indices[triangle * 3 + 0]);
        float3 b = vertex_position(data.vertices, data.indices[triangle * 3 + 1]);
        float3 c = vertex_position(data.vertices, data.indices[triangle * 3 + 2]);
        float3 object_normal = cross(b - a, c - a);

        // The inverse transpose of the instance's linear part, so scale and shear are carried.
        float3x4 world_to_object = query.CommittedWorldToObject3x4();
        float3x3 linear = float3x3(world_to_object[0].xyz, world_to_object[1].xyz, world_to_object[2].xyz);
        normal = normalize(mul(object_normal, linear));
        if (dot(normal, ray_direction) > 0.0)
        {
            normal = -normal;
        }

        t = query.CommittedRayT();
        float3 point = near_point + ray_direction * t;
        // The offset scales with distance: half precision positions and the hit point itself are only so exact.
        float bias = 0.002 + 0.0005 * t;
        bool lit = dot(normal, sun) > 0.0 && !occluded(point + normal * bias, sun, 1.0e4);
        mask = float3(lit ? 1.0 : 0.0);

        if (pc.mode == rt_view_instance)
        {
            colour = hash_colour(instance);
        }
    }

    switch (pc.mode)
    {
    case rt_view_shadow_mask:
        colour = mask;
        break;
    case rt_view_split:
        if (thread_id.x >= width / 2)
        {
            colour = mask;
        }
        break;
    case rt_view_depth_match:
        if (scene_depth <= 0.0)
        {
            colour = hit ? float3(0.9, 0.9, 0.1) : float3(0.4);
        }
        else
        {
            colour = hit && abs(t - scene_distance) <= 0.01 * scene_distance + 0.02 ? float3(0.1, 0.8, 0.2)
                                                                                     : float3(0.9, 0.1, 0.1);
        }
        break;
    case rt_view_normals:
        colour = hit ? normal * 0.5 + 0.5 : float3(0.0);
        break;
    case rt_view_overlay:
        colour = scene.rgb * (hit && mask.x < 0.5 ? 0.3 : 1.0);
        break;
    default:
        break;
    }

    storage_2d[pc.result][thread_id.xy] = float4(colour, scene.a);
}
