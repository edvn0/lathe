import bindless;
import scene_types;
import wind;
import pbr;
import meshlet_task;
import light_clusters;
import environment;

struct VertexOutput
{
    float4 position : SV_Position;

    float3 world_position;
    float3 world_normal;
    float4 world_tangent;
    float2 texture_coordinate;

    nointerpolation uint material_index;

    nointerpolation uint debug_cluster_id;
};

static const uint debug_instance_salt = 0x9E3779B9u;

[shader("amplification")]
[numthreads(meshlets_per_task, 1, 1)]
void main_task(uint3 group_id: SV_GroupID, uint lane: SV_GroupIndex, uint draw_index: SV_DrawIndex, uniform PC pc)
{
    run_meshlet_task(pc, 0, draw_index, group_id, lane);
}

VertexOutput forward_vertex(PC pc, GpuDraw draw, float4x4 model, float wind_strength, uint vertex_id)
{
    const var ubo = pc.ubo;
    const var vertex = decode_vertex(draw.vertex_address[vertex_id]);

    float4 world_position = mul(model, float4(vertex.position, 1.0));
    world_position.xz += wind_offset(vertex.position, world_position.xyz, wind_strength, ubo.time);

    float3x3 normal_matrix = (float3x3)model;

    float3 world_normal = normalize(mul(normal_matrix, vertex.normal));
    float3 world_tangent = normalize(mul(normal_matrix, vertex.tangent.xyz));

    VertexOutput output;
    output.position = mul(ubo.view_projection, world_position);
    output.world_position = world_position.xyz;
    output.world_normal = world_normal;
    output.world_tangent = float4(world_tangent, vertex.tangent.w);
    output.texture_coordinate = vertex.uvs;
    output.material_index = draw.material_index;
    output.debug_cluster_id = 0;

    return output;
}

[shader("vertex")]
VertexOutput main_vs(uint vertex_id: SV_VulkanVertexID, uint instance_index: SV_VulkanInstanceID, uniform PC pc)
{
    const GpuDraw draw = pc.draws[instance_index];
    const float wind_strength = pc.materials[draw.material_index].wind_strength;

    VertexOutput output = forward_vertex(pc, draw, pc.transforms[instance_index], wind_strength, vertex_id);
    output.debug_cluster_id = instance_index ^ debug_instance_salt;
    return output;
}

[shader("mesh")]
[numthreads(mesh_group_size, 1, 1)]
[outputtopology("triangle")]
void main_mesh(in payload MeshletPayload payload, uint3 group_id: SV_GroupID, uint lane: SV_GroupIndex,
               uniform PC pc, out vertices VertexOutput verts[meshlet_max_vertices],
               out indices uint3 tris[meshlet_max_triangles])
{
    const GpuDraw draw = pc.draws[payload.instance_index];
    const uint meshlet_index = payload_meshlet_index(payload, group_id.x);
    const Meshlet meshlet = draw.meshlets[meshlet_index];

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);

    const float4x4 model = pc.transforms[payload.instance_index];
    const float wind_strength = pc.materials[draw.material_index].wind_strength;

    for (uint v = lane; v < meshlet.vertex_count; v += mesh_group_size) {
        VertexOutput output = forward_vertex(pc, draw, model, wind_strength, meshlet_vertex_index(draw, meshlet, v));
        output.debug_cluster_id = meshlet_index;
        verts[v] = output;
    }

    for (uint t = lane; t < meshlet.triangle_count; t += mesh_group_size) {
        tris[t] = meshlet_triangle(draw, meshlet, t);
    }
}

struct OutputFS
{
    float4 colour : SV_Target0;

#ifdef OUTLINE_MASK
    float outline : SV_Target1;
#endif
};

uint pcg_hash(uint value)
{
    const uint state = value * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float3 debug_meshlet_colour(uint cluster_id)
{
    const uint hash = pcg_hash(cluster_id);
    const float hue = float(hash & 0xFFFFu) / 65535.0;
    const float value = 0.75 + 0.25 * (float((hash >> 16u) & 0xFFu) / 255.0);
    const float saturation = 0.65;

    const float3 hue_rgb = saturate(abs(frac(hue + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
    return value * lerp(float3(1.0), hue_rgb, saturation);
}

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

float3 shade_light(GpuLight punctual_light, float3 world_position, float3 shading_normal, float3 view_direction,
                   float3 base_colour, float metallic, float roughness)
{
    float3 to_light = punctual_light.position - world_position;
    float distance = length(to_light);
    float3 light_dir = to_light / max(distance, 1e-6);

    float distance_falloff = clamp(1.0 - pow(distance / max(punctual_light.range, 1e-4), 4.0), 0.0, 1.0);
    distance_falloff = (distance_falloff * distance_falloff) / max(distance * distance, 1e-4);

    float cone_falloff = 1.0;
    if (punctual_light.type == LightType.spot) {
        float cd = dot(normalize(punctual_light.direction), -light_dir);
        cone_falloff = clamp(cd * punctual_light.spot_scale + punctual_light.spot_offset, 0.0, 1.0);
        cone_falloff *= cone_falloff;
    }

    float3 radiance = punctual_light.colour * punctual_light.intensity * distance_falloff * cone_falloff;

    return shade_punctual(shading_normal, view_direction, light_dir, radiance, base_colour, metallic, roughness);
}

float3 cluster_heatmap_colour(uint light_count, uint capacity)
{
    if (light_count == 0) {
        return float3(0.0);
    }

    if (light_count > capacity) {
        return float3(1.0, 0.0, 1.0);
    }

    const float t = saturate(float(light_count) / 64.0);
    return saturate(float3(2.0 * t - 0.5, 1.5 - abs(4.0 * t - 2.0), 1.5 - 3.0 * t));
}

[shader("fragment")]
OutputFS main_fs(VertexOutput input, bool is_front_face: SV_IsFrontFace, uniform PC pc)
{
    const var ubo = pc.ubo;

    const var material = pc.materials[input.material_index];

    float3 normal = normalize(input.world_normal);

    if ((material.flags & material_flag_double_sided) != 0 && !is_front_face) {
        normal = -normal;
    }
    float3 tangent = normalize(input.world_tangent.xyz);

    tangent = normalize(tangent - normal * dot(normal, tangent));
    float3 bitangent = cross(normal, tangent) * input.world_tangent.w;

    float3x3 tbn = float3x3(tangent, bitangent, normal);

    float3 light_direction = normalize(ubo.light_direction);

    const var texture = sampled_2d[NonUniformResourceIndex(material.base_colour_texture)];
    const var emissive_texture = sampled_2d[NonUniformResourceIndex(material.emissive_texture)];
    const var ambient_occlusion_texture = sampled_2d[NonUniformResourceIndex(material.occlusion_texture)];
    const var normal_texture = sampled_2d[NonUniformResourceIndex(material.normal_texture)];
    const var metallic_roughness_texture = sampled_2d[NonUniformResourceIndex(material.metallic_roughness_texture)];
    const var sampler = samplers[NonUniformResourceIndex(material.sampler_index)];

    float2 uv = float2(input.texture_coordinate.x, input.texture_coordinate.y);

    float2 tangent_normal_xy = normal_texture.Sample(sampler, uv).xy * 2.0 - 1.0;
    tangent_normal_xy *= material.normal_scale;
    float tangent_normal_z = sqrt(saturate(1.0 - dot(tangent_normal_xy, tangent_normal_xy)));
    float3 tangent_normal = normalize(float3(tangent_normal_xy, tangent_normal_z));

    float3 shading_normal = normalize(mul(tangent_normal, tbn));

    float ndotl = saturate(dot(shading_normal, light_direction));

    float view_z = mul(ubo.view, float4(input.world_position, 1.0)).z;

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

    float2 screen_uv = input.position.xy / pc.screen_size;
    const var gtao_texture = sampled_2d[NonUniformResourceIndex(pc.ao_texture_index)];
    const var gtao_sampler = samplers[NonUniformResourceIndex(pc.ao_sampler_index)];
    float screen_space_ao = gtao_texture.Sample(gtao_sampler, screen_uv).r;
    ambient_occlusion *= lerp(1.0, screen_space_ao, ubo.ao_intensity);

const var base_sample = texture.Sample(sampler, uv);

if (material.alpha_mode == AlphaMode.mask && (material.flags & material_flag_alpha_to_coverage) == 0 &&
    base_sample.a < material.alpha_cutoff) {
    discard;
}

var base_colour = base_sample.xyz *
                  float3(material.colour[0], material.colour[1], material.colour[2]);

    if (material.debug_meshlet_colours != 0) {
        base_colour = debug_meshlet_colour(input.debug_cluster_id);
    }

    float2 metallic_roughness = metallic_roughness_texture.Sample(sampler, uv).bg *
                                float2(material.metallic_factor, material.roughness_factor);
    float metallic = metallic_roughness.x;
    float roughness = clamp(metallic_roughness.y, 0.045, 1.0);

    float3 view_direction = normalize(ubo.camera_position - input.world_position);

    float3 directional_contribution = shade_punctual(
            shading_normal, view_direction, light_direction, ubo.light_colour * ubo.light_intensity, base_colour,
            metallic, roughness) * shadow;

    float3 punctual_contribution = float3(0.0);
    uint cluster_light_count = 0;

    if (ubo.clustered_lighting != 0) {
        const float4 clip = mul(ubo.view_projection, float4(input.world_position, 1.0));
        const uint cluster = cluster_index(ubo, clip, view_z);

        cluster_light_count = pc.cluster_lights[cluster];

        const uint capacity = ubo.cluster_light_capacity;
        const uint stored_count = min(cluster_light_count, capacity);
        const uint list_base = cluster_count(ubo) + cluster * capacity;
        const uint list_end = 0xFFFFFFFFu;

        uint cursor = 0;
        uint lane_next = stored_count != 0 ? pc.cluster_lights[list_base] : list_end;

        for (;;) {
            const uint light_index = WaveActiveMin(lane_next);

            if (light_index == list_end) {
                break;
            }

            const GpuLight light = pc.lights[light_index];
            punctual_contribution += shade_light(light, input.world_position, shading_normal, view_direction,
                                                 base_colour, metallic, roughness) *
                                     light_screen_fade(ubo, light);

            if (lane_next == light_index) {
                ++cursor;
                lane_next = cursor < stored_count ? pc.cluster_lights[list_base + cursor] : list_end;
            }
        }
    } else {
        for (uint light_index = 0; light_index < pc.light_count; ++light_index) {
            const GpuLight light = pc.lights[light_index];
            punctual_contribution += shade_light(light, input.world_position, shading_normal, view_direction,
                                                 base_colour, metallic, roughness) *
                                     light_screen_fade(ubo, light);
        }
    }

    float3 ambient;

    if ((ubo.environment_flags & env_ibl_valid) != 0) {
        const uint debug_view = env_debug_view(ubo.environment_flags);
        const bool furnace = debug_view == env_debug_white_furnace;

        const float2 unrotate = float2(ubo.environment_rotation.x, -ubo.environment_rotation.y);
        const float3 reflected = reflect(-view_direction, shading_normal);

        const float ndotv = max(dot(shading_normal, view_direction), 1e-4);
        const float3 f0 = lerp(float3(0.04), base_colour, metallic);

        const float2 scale_bias = sampled_2d[ubo.brdf_lut_texture].SampleLevel(samplers[ubo.environment_sampler], float2(ndotv, roughness), 0).rg;
        const float3 specular_albedo = f0 * scale_bias.x + scale_bias.y;

        float3 energy = float3(1.0);
        if ((ubo.environment_flags & env_multi_scatter) != 0) {
            energy = 1.0 + f0 * (1.0 / max(scale_bias.x + scale_bias.y, 1e-4) - 1.0);
        }

        float3 irradiance = sh9_eval(ubo.environment_sh, rotate_y(shading_normal, unrotate));
        float3 prefiltered = sampled_cube[ubo.prefilter_cube_texture].SampleLevel(
            samplers[ubo.environment_sampler], rotate_y(reflected, unrotate), roughness * ubo.environment_rotation.z).rgb;

        if (furnace) {
            irradiance = float3(1.0);
            prefiltered = float3(1.0);
        }

        float so = 1.0;
        if ((ubo.environment_flags & env_specular_occlusion) != 0) {
            so = lerp(1.0, specular_occlusion(ndotv, ambient_occlusion, roughness), ubo.environment_rotation.w);
        }

        const float horizon = saturate(1.0 + dot(reflected, normal));

        const float3 diffuse_ibl = base_colour * (1.0 - metallic) * (1.0 - specular_albedo) * irradiance *
                                   ambient_occlusion * ubo.environment_intensity.x;
        const float3 specular_ibl = prefiltered * specular_albedo * energy * so * horizon * horizon * ubo.environment_intensity.y;

        ambient = (diffuse_ibl + specular_ibl) * ubo.environment_exposure;

        if (debug_view == env_debug_diffuse_only) {
            ambient = diffuse_ibl * ubo.environment_exposure;
        } else if (debug_view == env_debug_specular_only) {
            ambient = specular_ibl * ubo.environment_exposure;
        } else if (debug_view == env_debug_specular_occlusion) {
            ambient = float3(so);
        }
    } else {
        ambient = base_colour * ambient_occlusion * ubo.ambient_intensity;
    }

    float3 emissive = emissive_texture.Sample(sampler, uv).xyz *
                  float3(material.emissive[0], material.emissive[1], material.emissive[2]) *
                  material.emissive_strength;

    float3 shaded_colour = ambient + directional_contribution + punctual_contribution + emissive;

    if (ubo.cluster_debug_heatmap != 0) {
        shaded_colour = lerp(shaded_colour, cluster_heatmap_colour(cluster_light_count, ubo.cluster_light_capacity), 0.75);
    }

    if (ubo.shadow_debug_cascade_tint != 0) {
        const float3 tints[5] = {
            float3(1.0, 0.4, 0.4), float3(0.4, 1.0, 0.4), float3(0.4, 0.4, 1.0),
            float3(1.0, 1.0, 0.4), float3(1.0, 1.0, 1.0),
        };
        shaded_colour *= tints[min(cascade, 4u)];
    }

    float distance_to_camera = length(ubo.camera_position - input.world_position);

    float transmittance = exp(-distance_to_camera * ubo.fog_extinction);
    float3 fog_colour = ubo.fog_colour;

    if ((ubo.environment_flags & env_fog_from_environment) != 0) {
        const float3 view_ray = normalize(input.world_position - ubo.camera_position);
        fog_colour *= environment_fog_colour(ubo, view_ray);
    }

    float3 scattered = fog_colour * ubo.fog_inscattering * (1.0 - transmittance);

    float3 fogged_colour = shaded_colour * transmittance + scattered;

    OutputFS output;
    output.colour = float4(fogged_colour, base_sample.a);
#ifdef OUTLINE_MASK
    output.outline = (material.flags & material_flag_outlined) != 0 ? 1.0 : 0.0;
#endif

    return output;
}