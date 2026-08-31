import bindless;
import scene_types;
import wind;
import pbr;

struct VertexOutput
{
    float4 position : SV_Position;

    float3 world_position;
    float3 world_normal;
    float4 world_tangent; // xyz = tangent, w = handedness sign
    float2 texture_coordinate;

    nointerpolation uint material_index;
};

[shader("vertex")]
VertexOutput mainVs(
uint vertex_id: SV_VulkanVertexID,
                 uint draw_id: SV_DrawIndex,
                 uint inst_base: SV_VulkanInstanceID,
    uniform PC pc)
{
    GpuDraw draw = pc.draws[inst_base];
    const var ubo = pc.ubo;

    const var vertex = decode_vertex(draw.vertex_address[vertex_id]);

    float4x4 model = pc.transforms[inst_base];

    float4 world_position = mul(
        model,
        float4(vertex.position, 1.0)
    );

    const var wind_material = pc.materials[draw.material_index];
    world_position.xz += wind_offset(vertex.position, world_position.xyz, wind_material.wind_strength, ubo.time);

    // This is correct for uniform scale and rotation.
    // Use inverse-transpose when non-uniform scaling is supported.
    float3x3 normal_matrix = (float3x3)model;

    float3 world_normal = normalize(
        mul(
            normal_matrix,
            vertex.normal
        )
    );

    float3 world_tangent = normalize(
        mul(
            normal_matrix,
            vertex.tangent.xyz
        )
    );

    VertexOutput output;
    output.position = mul(ubo.view_projection, world_position);
    output.world_position = world_position.xyz;
    output.world_normal = world_normal;
    output.world_tangent = float4(world_tangent, vertex.tangent.w);
    output.texture_coordinate = vertex.uvs;
    output.material_index = draw.material_index;

    return output;
}

struct OutputFS
{
    float4 colour : SV_Target;
};

// 3x3 hardware-PCF tap. Each tap is clamped to a 1-texel inset of this
// cascade's tile before the atlas remap: without the inset, taps near a
// tile edge would leave the tile and CLAMP_TO_BORDER on the wide atlas
// would silently read the neighbouring cascade in x.
float sample_shadow_atlas(Ptr<UBO> ubo, uint cascade, float2 tile_uv, float reference)
{
    const var atlas = sampled_2d_depth[NonUniformResourceIndex(ubo.shadow_atlas_texture)];
    const var cmp = comparison_samplers[NonUniformResourceIndex(ubo.shadow_sampler)];

    const float step_u = ubo.shadow_atlas_texel_u * ubo.shadow_pcf_radius_texels;
    const float step_v = ubo.shadow_atlas_texel_v * ubo.shadow_pcf_radius_texels;

    const float inset_u = ubo.shadow_atlas_texel_u;
    const float inset_v = ubo.shadow_atlas_texel_v;
    const float2 lo = float2(inset_u, inset_v);
    const float2 hi = float2(1.0 - inset_u, 1.0 - inset_v);

    float sum = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            float2 tap = clamp(tile_uv + float2(float(x) * step_u, float(y) * step_v), lo, hi);
            float2 atlas_uv = float2(ubo.cascade_atlas_offset_u[cascade] + tap.x * ubo.cascade_atlas_scale_u[cascade],
                                     tap.y * ubo.cascade_atlas_scale_v[cascade]);
            sum += atlas.SampleCmpLevelZero(cmp, atlas_uv, reference);
        }
    }

    return sum * (1.0 / 9.0);
}

[shader("fragment")]
OutputFS mainFs(VertexOutput input, uniform PC pc)
{
    const var ubo = pc.ubo;

    float3 normal = normalize(input.world_normal);
    float3 tangent = normalize(input.world_tangent.xyz);

    // Re-orthogonalize in case interpolation skewed the basis.
    tangent = normalize(tangent - normal * dot(normal, tangent));
    float3 bitangent = cross(normal, tangent) * input.world_tangent.w;

    float3x3 tbn = float3x3(tangent, bitangent, normal);

    float3 light_direction = normalize(ubo.light_direction);

    const var material = pc.materials[input.material_index];
    const var texture = sampled_2d[NonUniformResourceIndex(material.base_colour_texture)];
    const var emissive_texture = sampled_2d[NonUniformResourceIndex(material.emissive_texture)];
    const var ambient_occlusion_texture = sampled_2d[NonUniformResourceIndex(material.occlusion_texture)];
    const var normal_texture = sampled_2d[NonUniformResourceIndex(material.normal_texture)];
    const var metallic_roughness_texture = sampled_2d[NonUniformResourceIndex(material.metallic_roughness_texture)];
    const var sampler = samplers[NonUniformResourceIndex(material.sampler_index)];

    float2 uv = float2(input.texture_coordinate.x, input.texture_coordinate.y);

    // Normal maps are stored BC5 (2 channels, X/Y only) -- Z is reconstructed
    // here rather than sampled, since BC5 has nowhere to store it.
    float2 tangent_normal_xy = normal_texture.Sample(sampler, uv).xy * 2.0 - 1.0;
    tangent_normal_xy *= material.normal_scale;
    float tangent_normal_z = sqrt(saturate(1.0 - dot(tangent_normal_xy, tangent_normal_xy)));
    float3 tangent_normal = normalize(float3(tangent_normal_xy, tangent_normal_z));

    float3 shading_normal = normalize(mul(tangent_normal, tbn));

    float ndotl = saturate(dot(shading_normal, light_direction));

    float view_z = mul(ubo.view, float4(input.world_position, 1.0)).z;

    // Cascade selection and the cascade_view_projection[] lookup are inlined
    // here rather than factored into a helper: passing Ptr<UBO> into a
    // non-entry-point function and then dynamically indexing an array of
    // matrices crashes the Slang SPIR-V backend (assert failure in
    // slang-ir.cpp, structType). sample_shadow_atlas() below is safe
    // because it never indexes a matrix array through the pointer.
    uint cascade = ubo.cascade_count;
    [unroll]
    for (uint i = 0; i < shadow_cascade_count; ++i) {
        if (i < ubo.cascade_count && view_z <= ubo.cascade_split_far[i]) {
            cascade = i;
            break;
        }
    }

    float shadow = 1.0;

    if (ndotl > 0.0 && cascade < ubo.cascade_count) {
        // Normal-offset bias along the geometric (not normal-mapped) normal
        // -- a perturbed normal can push the sample through geometry.
        // sqrt(2) covers a texel footprint's diagonal.
        float texel = ubo.cascade_texel_world[cascade];
        float3 biased = input.world_position + normal * (texel * ubo.shadow_normal_offset_texels * 1.41421356);

        float4 clip = mul(ubo.cascade_view_projection[cascade], float4(biased, 1.0));
        float3 ndc = clip.xyz / clip.w;

        if (any(abs(ndc.xy) > 1.0) || ndc.z <= 0.0 || ndc.z > 1.0) {
            shadow = 1.0;
        } else {
            float2 tile_uv = ndc.xy * 0.5 + 0.5;
            float reference = saturate(ndc.z + ubo.shadow_depth_bias_world * ubo.cascade_depth_scale[cascade]);
            shadow = sample_shadow_atlas(ubo, cascade, tile_uv, reference);
        }
    }

    float ambient_occlusion = ambient_occlusion_texture.Sample(sampler, uv).r;
    ambient_occlusion = lerp(1.0, ambient_occlusion, material.occlusion_strength);

    // Screen-space GTAO term (see gtao.slang/gtao_denoise.slang), sampled at
    // this fragment's own screen position and blended in on top of the
    // material's baked occlusion texture above -- the two account for
    // different things (macro-scale, cross-mesh occlusion the baked texture
    // can't see, vs. per-texel occlusion baked into the asset) and multiply
    // together rather than replacing one another.
    float2 screen_uv = input.position.xy / pc.screen_size;
    const var gtao_texture = sampled_2d[NonUniformResourceIndex(pc.ao_texture_index)];
    const var gtao_sampler = samplers[NonUniformResourceIndex(pc.ao_sampler_index)];
    float screen_space_ao = gtao_texture.Sample(gtao_sampler, screen_uv).r;
    ambient_occlusion *= lerp(1.0, screen_space_ao, ubo.ao_intensity);

const var base_sample = texture.Sample(sampler, uv);

if (material.alpha_mode == AlphaMode.mask && base_sample.a < material.alpha_cutoff) {
    discard;
}

var base_colour = base_sample.xyz *
                  float3(material.colour[0], material.colour[1], material.colour[2]);

    // glTF metallic-roughness convention: roughness in G, metallic in B.
    float2 metallic_roughness = metallic_roughness_texture.Sample(sampler, uv).bg *
                                float2(material.metallic_factor, material.roughness_factor);
    float metallic = metallic_roughness.x;
    float roughness = clamp(metallic_roughness.y, 0.045, 1.0);

    float3 view_direction = normalize(ubo.camera_position - input.world_position);

    float3 directional_contribution = shade_punctual(
            shading_normal, view_direction, light_direction, ubo.light_colour * ubo.light_intensity, base_colour,
            metallic, roughness) * shadow;

    float3 punctual_contribution = float3(0.0);

    for (uint light_index = 0; light_index < pc.light_count; ++light_index) {
        const var punctual_light = pc.lights[light_index];

        float3 to_light = punctual_light.position - input.world_position;
        float distance = length(to_light);
        float3 light_dir = to_light / max(distance, 1e-6);

        // Smooth range window on top of inverse-square falloff so lights
        // reach exactly zero at `range` instead of cutting off with a hard
        // edge -- same shape as glTF KHR_lights_punctual's recommended
        // attenuation.
        float distance_falloff = clamp(1.0 - pow(distance / max(punctual_light.range, 1e-4), 4.0), 0.0, 1.0);
        distance_falloff = (distance_falloff * distance_falloff) / max(distance * distance, 1e-4);

        float cone_falloff = 1.0;
        if (punctual_light.type == LightType.spot) {
            float cd = dot(normalize(punctual_light.direction), -light_dir);
            cone_falloff = clamp(cd * punctual_light.spot_scale + punctual_light.spot_offset, 0.0, 1.0);
            cone_falloff *= cone_falloff;
        }

        float3 radiance = punctual_light.colour * punctual_light.intensity * distance_falloff * cone_falloff;

        punctual_contribution += shade_punctual(shading_normal, view_direction, light_dir, radiance, base_colour,
                                                metallic, roughness);
    }

    float3 ambient = base_colour * ambient_occlusion * ubo.ambient_intensity;
    float3 emissive = emissive_texture.Sample(sampler, uv).xyz *
                  float3(material.emissive[0], material.emissive[1], material.emissive[2]) *
                  material.emissive_strength;

    float3 shaded_colour = ambient + directional_contribution + punctual_contribution + emissive;

    if (ubo.shadow_debug_cascade_tint != 0) {
        const float3 tints[5] = {
            float3(1.0, 0.4, 0.4), float3(0.4, 1.0, 0.4), float3(0.4, 0.4, 1.0),
            float3(1.0, 1.0, 0.4), float3(1.0, 1.0, 1.0),
        };
        shaded_colour *= tints[min(cascade, 4u)];
    }

    float distance_to_camera = length(ubo.camera_position - input.world_position);

    float transmittance = exp(-distance_to_camera * ubo.fog_extinction);
    float3 scattered = ubo.fog_colour * ubo.fog_inscattering * (1.0 - transmittance);

    float3 fogged_colour = shaded_colour * transmittance + scattered;

    OutputFS output;
    output.colour = float4(fogged_colour, base_sample.a);

    return output;
}