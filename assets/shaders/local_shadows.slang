module local_shadows;

import bindless;
import scene_types;

// Cube face of a direction, in the order the tiles were rendered: +X, -X, +Y, -Y, +Z, -Z.
public uint point_shadow_face(float3 direction)
{
    const float3 magnitude = abs(direction);

    if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) {
        return direction.x >= 0.0 ? 0u : 1u;
    }

    if (magnitude.y >= magnitude.z) {
        return direction.y >= 0.0 ? 2u : 3u;
    }

    return direction.z >= 0.0 ? 4u : 5u;
}

// Fraction of the light that reaches world_position: 1 lit, 0 shadowed. Lights without a tile are always lit.
public float local_shadow(Ptr<UBO> ubo, GpuLight light, float3 world_position, float3 geometric_normal)
{
    if (light.shadow_tile == no_shadow_tile || ubo.local_shadow_tile_count == 0) {
        return 1.0;
    }

    const float3 to_fragment = world_position - light.position;
    const float distance = max(length(to_fragment), 1e-4);

    uint tile_index = light.shadow_tile;

    if (light.type == LightType.point) {
        tile_index += point_shadow_face(to_fragment);
    }

    const GpuShadowTile tile = ubo.local_shadow_tiles[tile_index];

    const float texel_world = tile.texel_world * distance;
    const float3 biased = world_position + geometric_normal * (texel_world * ubo.local_shadow_normal_offset_texels);

    const float4 clip = mul(tile.view_projection, float4(biased, 1.0));

    if (clip.w <= 0.0) {
        return 1.0;
    }

    const float3 ndc = clip.xyz / clip.w;

    if (any(abs(ndc.xy) > 1.0) || ndc.z <= 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    const var atlas = sampled_2d_depth[NonUniformResourceIndex(ubo.local_shadow_atlas_texture)];
    const var cmp = comparison_samplers[NonUniformResourceIndex(ubo.shadow_sampler)];

    const float2 tile_uv = clamp(ndc.xy * 0.5 + 0.5, float2(tile.tile_texel), float2(1.0 - tile.tile_texel));
    const float2 uv = tile.atlas_rect.xy + tile_uv * tile.atlas_rect.zw;
    const float reference = saturate(ndc.z + tile.depth_bias);

    // The comparison sampler already filters bilinearly; four taps half a texel apart widen that to a soft edge.
    const float2 half_texel = tile.atlas_rect.zw * (0.5 * tile.tile_texel);

    float sum = 0.0;
    sum += atlas.SampleCmpLevelZero(cmp, uv + float2(-half_texel.x, -half_texel.y), reference);
    sum += atlas.SampleCmpLevelZero(cmp, uv + float2(half_texel.x, -half_texel.y), reference);
    sum += atlas.SampleCmpLevelZero(cmp, uv + float2(-half_texel.x, half_texel.y), reference);
    sum += atlas.SampleCmpLevelZero(cmp, uv + float2(half_texel.x, half_texel.y), reference);

    return sum * 0.25;
}
