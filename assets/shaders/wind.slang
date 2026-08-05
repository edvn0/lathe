module wind;

// Shared wind-sway displacement for foliage. Height weight comes from the
// *local* vertex Y (grass meshes are authored with the blade base at y=0
// and the tip at y=1), so only the tip sways while the base stays planted;
// the phase comes from *world* XZ so neighbouring blades don't move in
// lockstep. Applied identically in forward_geom, depth_prepass and
// shadow_depth -- skipping it in any one of those would desync depth/shadow
// from what's actually drawn.
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
