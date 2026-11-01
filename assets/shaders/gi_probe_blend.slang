// Probe GI: folds this frame's rays into the atlases (gi_probes.slang for the layout).
//   irradiance_cs  one thread per interior irradiance texel: the cosine weighted mean ray radiance in the texel's
//                  direction, blended into the old value
//   distance_cs    the same for the distance atlas: mean and mean square of the ray distance, weight cos^64
//   border_cs      one thread per texel of an atlas: copies the border from the opposite interior edge so bilinear
//                  filtering across the octahedron's edge is seamless (a separate pass, the interior is final by then)
// New and just reassigned probes take the new value outright (hysteresis 0), young ones blend fast.
import bindless;
import gi_probes;

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

struct Push
{
    Ptr<float4> probe_state;
    Ptr<uint2> ray_data;
    uint atlas;
    uint count_x;
    uint count_y;
    uint count_z;
    float spacing;
    uint rays_per_probe;
    float hysteresis;
    uint tile;
    float4 rotation;
};

[vk::push_constant] Push pc;

static const float gi_max_radiance = 64.0;

float probe_hysteresis(uint slot)
{
    float age = pc.probe_state[slot * gi_state_stride + 2].w;
    if (age < 0.5)
    {
        return 0.0;
    }
    float h = age < 8.0 ? min(0.85, pc.hysteresis) : pc.hysteresis;
    // The probe's lighting just changed a lot: follow it.
    return pc.probe_state[slot * gi_state_stride + 3].w > 0.5 ? min(h, 0.4) : h;
}

void unpack(uint2 packed, out float3 radiance, out float distance)
{
    radiance = float3(f16tof32(packed.x & 0xFFFF), f16tof32(packed.x >> 16), f16tof32(packed.y & 0xFFFF));
    distance = f16tof32(packed.y >> 16);
}

// The tile and texel of a thread of an atlas with `tile` texel tiles; false outside the atlas.
bool locate(uint2 id, uint tile, out uint slot, out int2 texel)
{
    uint per_row = pc.count_x * pc.count_y;
    uint rows = pc.count_z;
    slot = 0;
    texel = int2(0);
    if (id.x >= per_row * tile || id.y >= rows * tile)
    {
        return false;
    }
    slot = (id.y / tile) * per_row + id.x / tile;
    texel = int2(id.x % tile, id.y % tile);
    return true;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void irradiance_cs(uint3 id : SV_DispatchThreadID)
{
    uint slot;
    int2 texel;
    if (!locate(id.xy, gi_irradiance_tile, slot, texel))
    {
        return;
    }
    int2 interior = texel - 1;
    if (any(interior < 0) || any(interior >= int(gi_irradiance_interior)))
    {
        return;
    }

    float3 direction = gi_oct_decode((float2(interior) + 0.5) / float(gi_irradiance_interior) * 2.0 - 1.0);
    float3 sum = float3(0.0);
    float weight_sum = 0.0;
    for (uint i = 0; i < pc.rays_per_probe; ++i)
    {
        float3 ray_direction = gi_rotate(pc.rotation, gi_fibonacci_direction(i, pc.rays_per_probe));
        float weight = max(0.0, dot(direction, ray_direction));
        if (weight <= 0.0)
        {
            continue;
        }
        float3 radiance;
        float distance;
        unpack(pc.ray_data[slot * pc.rays_per_probe + i], radiance, distance);
        sum += min(radiance, float3(gi_max_radiance)) * weight;
        weight_sum += weight;
    }
    float3 fresh = weight_sum > 1e-4 ? sum / weight_sum : float3(0.0);

    float4 old = storage_2d[pc.atlas][id.xy];
    // A new probe takes the fresh value outright: the atlas may hold anything (NaN included) for it.
    float h = probe_hysteresis(slot);
    storage_2d[pc.atlas][id.xy] = float4(h <= 0.0 ? fresh : lerp(fresh, old.rgb, h), 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void distance_cs(uint3 id : SV_DispatchThreadID)
{
    uint slot;
    int2 texel;
    if (!locate(id.xy, gi_distance_tile, slot, texel))
    {
        return;
    }
    int2 interior = texel - 1;
    if (any(interior < 0) || any(interior >= int(gi_distance_interior)))
    {
        return;
    }

    float3 direction = gi_oct_decode((float2(interior) + 0.5) / float(gi_distance_interior) * 2.0 - 1.0);
    float cap = 5.0 * pc.spacing;
    float2 sum = float2(0.0);
    float weight_sum = 0.0;
    for (uint i = 0; i < pc.rays_per_probe; ++i)
    {
        float3 ray_direction = gi_rotate(pc.rotation, gi_fibonacci_direction(i, pc.rays_per_probe));
        float weight = pow(max(0.0, dot(direction, ray_direction)), 64.0);
        if (weight <= 1e-6)
        {
            continue;
        }
        float3 radiance;
        float distance;
        unpack(pc.ray_data[slot * pc.rays_per_probe + i], radiance, distance);
        float d = min(abs(distance), cap);
        sum += float2(d, d * d) * weight;
        weight_sum += weight;
    }
    float2 fresh = weight_sum > 1e-6 ? sum / weight_sum : float2(cap, cap * cap);

    float4 old = storage_2d[pc.atlas][id.xy];
    float h = probe_hysteresis(slot);
    storage_2d[pc.atlas][id.xy] = float4(h <= 0.0 ? fresh : lerp(fresh, old.rg, h), 0.0, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void border_cs(uint3 id : SV_DispatchThreadID)
{
    uint tile = pc.tile;
    uint slot;
    int2 texel;
    if (!locate(id.xy, tile, slot, texel))
    {
        return;
    }
    int last = int(tile) - 1;
    bool on_x = texel.x == 0 || texel.x == last;
    bool on_y = texel.y == 0 || texel.y == last;
    if (!on_x && !on_y)
    {
        return;
    }

    int2 source;
    if (on_x && on_y)
    {
        source = int2(texel.x == 0 ? last - 1 : 1, texel.y == 0 ? last - 1 : 1);
    }
    else if (on_x)
    {
        source = int2(texel.x == 0 ? 1 : last - 1, last - texel.y);
    }
    else
    {
        source = int2(last - texel.x, texel.y == 0 ? 1 : last - 1);
    }

    uint2 origin = id.xy - uint2(texel);
    storage_2d[pc.atlas][id.xy] = storage_2d[pc.atlas][origin + uint2(source)];
}
