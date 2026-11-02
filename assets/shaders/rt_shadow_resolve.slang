// Temporal filter of the ray traced shadow mask (docs/ray-traced-shadows.md): one compute pass between rt_shadows and the
// forward pass. rt_shadows traces a few noisy rays per pixel, a different set every frame, so its raw mask flickers even
// with a still camera. This pass reprojects the last resolved mask (the history) onto the pixel's current surface point,
// clamps it to the raw mask's 5x5 neighbourhood so a moving shadow does not leave a trail, and blends the raw value in at a
// fixed rate: the result converges over about ten frames.
//
// A pixel whose history cannot be trusted (it was off screen, or hidden behind something else a frame ago, or the camera
// cut, or the history was just created) falls back to a depth aware blur of the raw mask instead of showing the noise.
//
// The mask is RGBA: G, B and A are the local light slots, filtered like the sun in R. The history's distance image keeps
// each pixel's view depth (clip w) so a reprojected tap can be rejected where the surface differs.
import bindless;
import common;

// Mirrors Push in ray_traced_shadows.cxx.
struct PushConstants
{
    float4x4 inverse_view_projection;
    float4x4 view_projection;
    float4x4 previous_view_projection;
    uint depth;
    uint raw;
    uint history_mask;
    uint history_depth;
    uint result_mask;
    uint result_depth;
    uint width;
    uint height;
    // The frame number rt_shadows traced with: it picks which pixel of its block a mask pixel stands for.
    uint trace_frame;
    uint resolution_shift;
    // Bit 0: the history holds a mask made with previous_view_projection.
    uint flags;
    // How much of the raw mask each frame brings in once the history is valid.
    float blend;
};

[vk::push_constant] PushConstants pc;

static const uint flag_history_valid = 1u;

float depth_at(int2 pixel)
{
    int2 clamped = clamp(pixel, int2(0), int2(int(pc.width) - 1, int(pc.height) - 1));
    return sampled_2d_depth[pc.depth].Load(int3(clamped, 0));
}

// The renderer draws with a negative height viewport, so NDC y points up while the pixel row points down; the depth
// buffer is reversed (1 at the near plane, 0 at the far plane and the sky), so NDC z is 1 - depth (see rt_shadows.slang).
float3 unproject(int2 pixel, float depth)
{
    float2 uv = (float2(pixel) + 0.5) / float2(pc.width, pc.height);
    float4 world = mul(pc.inverse_view_projection, float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0 - depth, 1.0));
    return world.xyz / world.w;
}

uint2 mask_size()
{
    uint block = 1u << pc.resolution_shift;
    return (uint2(pc.width, pc.height) + (block - 1u)) >> pc.resolution_shift;
}

// The screen pixel a mask pixel traced from this frame (rt_shadows.slang's choice).
int2 traced_pixel(int2 mask_pixel)
{
    uint block = 1u << pc.resolution_shift;
    uint offset_index = (pc.trace_frame * 5u) % (block * block);
    return min(int2(uint2(mask_pixel) * block + uint2(offset_index % block, offset_index / block)),
               int2(int(pc.width) - 1, int(pc.height) - 1));
}

float4 raw_at(int2 mask_pixel)
{
    int2 size = int2(mask_size());
    return sampled_2d[pc.raw].Load(int3(clamp(mask_pixel, int2(0), size - 1), 0));
}

// The view depth of a mask pixel's surface (clip w), or 0 for the sky.
float view_depth_at(int2 mask_pixel)
{
    int2 pixel = traced_pixel(clamp(mask_pixel, int2(0), int2(mask_size()) - 1));
    float depth = depth_at(pixel);
    if (depth <= 0.0)
    {
        return 0.0;
    }
    return mul(pc.view_projection, float4(unproject(pixel, depth), 1.0)).w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID)
{
    uint2 size = mask_size();
    if (thread_id.x >= size.x || thread_id.y >= size.y)
    {
        return;
    }
    int2 mask_pixel = int2(thread_id.xy);
    int2 pixel = traced_pixel(mask_pixel);

    float depth = depth_at(pixel);
    float4 raw = raw_at(mask_pixel);
    if (depth <= 0.0)
    {
        // Sky: nothing to shadow, and no history to keep.
        storage_2d[pc.result_mask][thread_id.xy] = raw;
        storage_2d[pc.result_depth][thread_id.xy] = float4(0.0);
        return;
    }

    float3 position = unproject(pixel, depth);
    float view_depth = mul(pc.view_projection, float4(position, 1.0)).w;

    // The history: the four taps around where this surface point was last frame, each kept only where the surface it
    // held is at the depth this point had then.
    float4 history = float4(0.0);
    float history_weight = 0.0;
    if ((pc.flags & flag_history_valid) != 0u)
    {
        float4 previous = mul(pc.previous_view_projection, float4(position, 1.0));
        if (previous.w > 0.0)
        {
            float2 uv = float2(previous.x / previous.w * 0.5 + 0.5, (1.0 - previous.y / previous.w) * 0.5);
            if (all(uv >= float2(0.0)) && all(uv < float2(1.0)))
            {
                float2 at = uv * float2(size) - 0.5;
                int2 base = int2(floor(at));
                float2 f = at - float2(base);
                float tolerance = 0.02 * previous.w + 0.02;
                for (int i = 0; i < 4; ++i)
                {
                    int2 tap = base + int2(i & 1, i >> 1);
                    if (any(tap < int2(0)) || any(tap >= int2(size)))
                    {
                        continue;
                    }
                    float tap_depth = sampled_2d[pc.history_depth].Load(int3(tap, 0)).x;
                    if (tap_depth <= 0.0 || abs(tap_depth - previous.w) > tolerance)
                    {
                        continue;
                    }
                    float w = ((i & 1) != 0 ? f.x : 1.0 - f.x) * ((i >> 1) != 0 ? f.y : 1.0 - f.y);
                    history += sampled_2d[pc.history_mask].Load(int3(tap, 0)) * w;
                    history_weight += w;
                }
            }
        }
    }

    float4 result;
    if (history_weight > 0.05)
    {
        history /= history_weight;
        // A shadow that moved on must not leave its old value behind: keep the history inside what the raw mask shows
        // around this pixel, 5x5 (in a penumbra the raw values span the range, so the clamp does not bite there).
        float4 low = raw;
        float4 high = raw;
        for (int y = -2; y <= 2; ++y)
        {
            for (int x = -2; x <= 2; ++x)
            {
                float4 neighbour = raw_at(mask_pixel + int2(x, y));
                low = min(low, neighbour);
                high = max(high, neighbour);
            }
        }
        result = lerp(clamp(history, low, high), raw, pc.blend);
    }
    else
    {
        // No usable history: blur the raw mask over the neighbours on the same surface.
        float tolerance = 0.03 * view_depth + 0.03;
        float4 sum = raw;
        float total = 1.0;
        for (int y = -2; y <= 2; ++y)
        {
            for (int x = -2; x <= 2; ++x)
            {
                if (x == 0 && y == 0)
                {
                    continue;
                }
                int2 neighbour = mask_pixel + int2(x, y);
                if (any(neighbour < int2(0)) || any(neighbour >= int2(size)))
                {
                    continue;
                }
                float neighbour_depth = view_depth_at(neighbour);
                if (neighbour_depth > 0.0 && abs(neighbour_depth - view_depth) <= tolerance)
                {
                    sum += raw_at(neighbour);
                    total += 1.0;
                }
            }
        }
        result = sum / total;
    }

    storage_2d[pc.result_mask][thread_id.xy] = result;
    storage_2d[pc.result_depth][thread_id.xy] = float4(view_depth, 0.0, 0.0, 0.0);
}
