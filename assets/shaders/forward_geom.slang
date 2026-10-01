import bindless;
import scene_types;
import wind;
import pbr;
import meshlet_task;
import light_clusters;

struct VertexOutput
{
    float4 position : SV_Position;

    float3 world_position;
    float3 world_normal;
    float4 world_tangent; // xyz = tangent, w = handedness sign
    float2 texture_coordinate;

    nointerpolation uint material_index;

    // Meshlet index within the mesh on the meshlet path, a salted instance index on the instanced path. Only read
    // for materials with debug_meshlet_colours set.
    nointerpolation uint debug_cluster_id;
};

// Keeps instanced-path ids from hashing to the same colours as meshlet ids.
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

    // Valid for uniform scale and rotation; non-uniform scale would need the inverse-transpose.
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

// Instanced path for batches too small for meshlets. Same vertex function as main_mesh.
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
    float4 colour : SV_Target;
};

// PCG hash: meshoptimizer emits spatially coherent meshlets in order, so consecutive ids must map to unrelated
// colours or neighbours become indistinguishable.
uint pcg_hash(uint value)
{
    const uint state = value * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Saturated, bright hue per id, so every cluster reads clearly under lighting.
float3 debug_meshlet_colour(uint cluster_id)
{
    const uint hash = pcg_hash(cluster_id);
    const float hue = float(hash & 0xFFFFu) / 65535.0;
    const float value = 0.75 + 0.25 * (float((hash >> 16u) & 0xFFu) / 255.0);
    const float saturation = 0.65;

    const float3 hue_rgb = saturate(abs(frac(hue + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
    return value * lerp(float3(1.0), hue_rgb, saturation);
}

// 3x3 hardware PCF. Taps are clamped to a 1-texel inset of the cascade's tile, or edge taps would read the
// neighbouring cascade in the atlas.
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

// Radiance from one punctual light, with KHR_lights_punctual's distance and cone falloff. Exactly zero beyond
// `range` and outside the cone, which the clustered loop relies on.
float3 shade_light(GpuLight punctual_light, float3 world_position, float3 shading_normal, float3 view_direction,
                   float3 base_colour, float metallic, float roughness)
{
    float3 to_light = punctual_light.position - world_position;
    float distance = length(to_light);
    float3 light_dir = to_light / max(distance, 1e-6);

    // KHR_lights_punctual's smooth window on top of inverse-square, reaching zero at `range`.
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

// Black (no lights) through blue, green and yellow to red at 64 or more lights in the cluster, and magenta past
// the cluster's capacity, where it drops lights.
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
OutputFS main_fs(VertexOutput input, uniform PC pc)
{
    const var ubo = pc.ubo;

    float3 normal = normalize(input.world_normal);
    float3 tangent = normalize(input.world_tangent.xyz);

    // Re-orthogonalize after interpolation.
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

    // BC5 normal maps store X/Y only; reconstruct Z.
    float2 tangent_normal_xy = normal_texture.Sample(sampler, uv).xy * 2.0 - 1.0;
    tangent_normal_xy *= material.normal_scale;
    float tangent_normal_z = sqrt(saturate(1.0 - dot(tangent_normal_xy, tangent_normal_xy)));
    float3 tangent_normal = normalize(float3(tangent_normal_xy, tangent_normal_z));

    float3 shading_normal = normalize(mul(tangent_normal, tbn));

    float ndotl = saturate(dot(shading_normal, light_direction));

    float view_z = mul(ubo.view, float4(input.world_position, 1.0)).z;

    // Inlined on purpose: dynamically indexing a matrix array through Ptr<UBO> in a helper function crashes
    // Slang's SPIR-V backend.
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
        // Normal-offset bias along the geometric normal; a normal-mapped one can push the sample through geometry.
        // sqrt(2) covers a texel's diagonal.
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

    // Screen-space GTAO multiplies with the baked occlusion; they capture different scales.
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

    if (material.debug_meshlet_colours != 0) {
        base_colour = debug_meshlet_colour(input.debug_cluster_id);
    }

    // glTF: roughness in G, metallic in B.
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

        // Each lane's list is sorted by light index, so the wave merges them: every iteration shades the smallest
        // pending index across the wave, which keeps the loop uniform and the light load scalar. Lanes whose cluster
        // lacks that light are out of its range and add zero.
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

    float3 ambient = base_colour * ambient_occlusion * ubo.ambient_intensity;
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
    float3 scattered = ubo.fog_colour * ubo.fog_inscattering * (1.0 - transmittance);

    float3 fogged_colour = shaded_colour * transmittance + scattered;

    OutputFS output;
    output.colour = float4(fogged_colour, base_sample.a);

    return output;
}