module wind;

// Foliage sway. The weight comes from local Y (blade base at 0, tip at 1) so bases stay planted; the phase
// comes from world XZ so neighbours don't move in lockstep. Must be applied identically in forward,
// depth_prepass and shadow_depth.
public float2 wind_offset(float3 local_position, float3 world_position, float wind_strength, float time)
{
    if (wind_strength <= 0.0) {
        return float2(0.0, 0.0);
    }

    float height_weight = saturate(local_position.y);
    height_weight *= height_weight;

    float phase = world_position.x * 0.15 + world_position.z * 0.17;

    float sway_x = sin(time * 1.6 + phase) * 0.7 + sin(time * 3.1 + phase * 1.7) * 0.3;
    float sway_z = cos(time * 1.3 + phase * 1.3) * 0.7 + cos(time * 2.7 + phase * 0.9) * 0.3;

    return float2(sway_x, sway_z) * wind_strength * height_weight;
}
