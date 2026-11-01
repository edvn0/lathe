module gi_probes;

import bindless;
import scene_types;

// Probe GI (docs/probe-gi.md). The probe volume is a grid of probes addressed toroidally: the probe with world index i
// (an integer cell, the probe sits at i * spacing + its relocation offset) lives in slot (i mod counts), slot =
// x + counts.x * (y + counts.y * z). Each probe owns a tile of two atlases: irradiance (8x8 octahedral texels plus a
// one texel border) and distance (16x16 plus a border), tiles laid out count.x * count.y to a row.
//
// Per probe state (probe_state, four float4 per probe):
//   [0] relocation offset xyz, w: 1 active, 0 off (inside geometry)
//   [1] mean ray radiance rgb, w: the fraction of rays that hit a back face
//   [2] the world index the slot currently holds (exact in a float), w: frames since the slot was (re)assigned
//   [3] slow average of the mean ray radiance rgb, w: 1 while the probe's lighting just changed a lot (a moving light or
//       sun, a door): the blend then lets the new rays in faster
public static const uint gi_irradiance_interior = 8;
public static const uint gi_irradiance_tile = 10;
public static const uint gi_distance_interior = 16;
public static const uint gi_distance_tile = 18;
public static const uint gi_state_stride = 4;

// Bits of GpuGiBlock::flags.
public static const uint gi_flag_enabled = 1;
public static const uint gi_flag_debug_shift = 1;

public float2 gi_oct_encode(float3 n)
{
    float3 a = abs(n);
    float2 p = n.xy / (a.x + a.y + a.z);
    if (n.z < 0.0)
    {
        p = (1.0 - abs(p.yx)) * float2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
    }
    return p;
}

public float3 gi_oct_decode(float2 e)
{
    float3 v = float3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0)
    {
        v.xy = (1.0 - abs(v.yx)) * float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(v);
}

public int3 gi_wrap(int3 index, int3 counts)
{
    return ((index % counts) + counts) % counts;
}

public uint gi_slot_of(int3 index, int3 counts)
{
    int3 s = gi_wrap(index, counts);
    return uint(s.x + counts.x * (s.y + counts.y * s.z));
}

public int3 gi_slot3(uint slot, int3 counts)
{
    return int3(int(slot) % counts.x, (int(slot) / counts.x) % counts.y, int(slot) / (counts.x * counts.y));
}

// The world index of the probe a slot holds while the window starts at `base`.
public int3 gi_world_index(uint slot, int3 base, int3 counts)
{
    return base + gi_wrap(gi_slot3(slot, counts) - base, counts);
}

// Texture coordinate of `direction` in the tile of `slot` of an atlas with square tiles of `tile` texels (`interior`
// of them octahedral).
public float2 gi_atlas_uv(uint slot, int3 counts, uint tile, uint interior, float3 direction, float2 atlas_size)
{
    uint per_row = uint(counts.x * counts.y);
    float2 origin = float2(float(slot % per_row) * float(tile), float(slot / per_row) * float(tile));
    float2 e = gi_oct_encode(direction) * 0.5 + 0.5;
    return (origin + 1.0 + e * float(interior)) / atlas_size;
}

public float3 gi_fibonacci_direction(uint index, uint count)
{
    float z = 1.0 - (2.0 * float(index) + 1.0) / float(count);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.399963229728653 * float(index);
    return float3(r * cos(phi), r * sin(phi), z);
}

public float3 gi_rotate(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Diffuse irradiance (as the SH term: the mean radiance over the cosine lobe, so albedo * result is the outgoing
// radiance) from the probe volume at `position` for a surface facing `normal`. `coverage` is 0 outside the volume (and
// where every neighbouring probe is off) and 1 well inside; the caller blends the result with its fallback by it.
public float3 gi_sample_irradiance(Ptr<UBO> ubo, float3 position, float3 normal, float3 view_direction,
                                   out float coverage)
{
    GpuGiBlock gi = ubo.gi;
    coverage = 0.0;

    int3 base = int3(gi.base_x, gi.base_y, gi.base_z);
    int3 counts = int3(gi.count_x, gi.count_y, gi.count_z);
    float3 biased = position + normal * gi.normal_bias + view_direction * gi.view_bias;
    float3 cell = biased / gi.spacing;
    int3 i0 = int3(floor(cell));
    float3 alpha = saturate(cell - float3(i0));

    // Fade over the outermost probe layer so the volume's edge does not pop.
    float3 low = cell - float3(base) - 1.0;
    float3 high = float3(base + counts) - 2.0 - cell;
    float edge = min(min(min(low.x, low.y), low.z), min(min(high.x, high.y), high.z));
    float fade = saturate(edge);
    if (fade <= 0.0)
    {
        return float3(0.0);
    }

    SamplerState linear_sampler = samplers[ubo.environment_sampler];
    uint2 irradiance_size;
    sampled_2d[gi.irradiance_texture].GetDimensions(irradiance_size.x, irradiance_size.y);
    uint2 distance_size;
    sampled_2d[gi.distance_texture].GetDimensions(distance_size.x, distance_size.y);

    float3 sum = float3(0.0);
    float weight_sum = 0.0;
    for (uint c = 0; c < 8; ++c)
    {
        int3 offset = int3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
        int3 index = i0 + offset;
        uint slot = gi_slot_of(index, counts);
        float4 state = gi.probe_state[slot * gi_state_stride];
        if (state.w < 0.5)
        {
            continue;
        }

        float3 probe_position = float3(index) * gi.spacing + state.xyz;
        float3 trilinear3 = lerp(1.0 - alpha, alpha, float3(offset));
        float trilinear = trilinear3.x * trilinear3.y * trilinear3.z;

        // Probes behind the surface contribute less.
        float3 to_probe = normalize(probe_position - position);
        float facing = (dot(to_probe, normal) + 1.0) * 0.5;
        float weight = facing * facing + 0.2;

        // Chebyshev visibility from the distance atlas: is the point behind what the probe sees in its direction?
        float3 from_probe = biased - probe_position;
        float distance_to_probe = length(from_probe);
        float2 distance_uv = gi_atlas_uv(slot, counts, gi_distance_tile, gi_distance_interior,
                                         from_probe / max(distance_to_probe, 1e-4), float2(distance_size));
        float2 moments = sampled_2d[gi.distance_texture].SampleLevel(linear_sampler, distance_uv, 0).rg;
        if (distance_to_probe > moments.x)
        {
            float variance = abs(moments.x * moments.x - moments.y);
            float d = distance_to_probe - moments.x;
            float chebyshev = variance / (variance + d * d);
            weight *= max(chebyshev * chebyshev * chebyshev, 0.0);
        }

        weight = max(1e-6, weight);
        if (weight < 0.2)
        {
            weight *= weight * weight * (1.0 / 0.04);
        }
        weight *= trilinear;

        float2 irradiance_uv = gi_atlas_uv(slot, counts, gi_irradiance_tile, gi_irradiance_interior, normal,
                                           float2(irradiance_size));
        sum += weight * sampled_2d[gi.irradiance_texture].SampleLevel(linear_sampler, irradiance_uv, 0).rgb;
        weight_sum += weight;
    }

    if (weight_sum <= 1e-5)
    {
        return float3(0.0);
    }
    coverage = fade;
    return sum / weight_sum * gi.intensity;
}

// What the probe nearest to `position` says about itself, for the debug views: green active, red off.
public float3 gi_debug_state_colour(Ptr<UBO> ubo, float3 position)
{
    GpuGiBlock gi = ubo.gi;
    int3 base = int3(gi.base_x, gi.base_y, gi.base_z);
    int3 counts = int3(gi.count_x, gi.count_y, gi.count_z);
    int3 index = int3(floor(position / gi.spacing + 0.5));
    if (any(index < base) || any(index >= base + counts))
    {
        return float3(0.2);
    }
    float4 state = gi.probe_state[gi_slot_of(index, counts) * gi_state_stride];
    return state.w > 0.5 ? float3(0.1, 0.8, 0.2) : float3(0.9, 0.1, 0.1);
}
