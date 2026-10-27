import bindless;
import scene_types;
import wind;
import pbr;
import meshlet_task;
import light_clusters;
import environment;
import local_shadows;
#ifdef BLEND_PASS
import water;
#endif

struct VertexOutput
{
    float4 position : SV_Position;

    float3 world_position;
    float3 world_normal;
    float4 world_tangent;
    float2 texture_coordinate;

    nointerpolation uint material_index;

    nointerpolation uint debug_cluster_id;

    // Which faces this draw keeps (blend_faces_*), for blended draws.
    nointerpolation uint blend_faces;
};

static const uint debug_instance_salt = 0x9E3779B9u;

[shader("amplification")]
[numthreads(meshlets_per_task, 1, 1)]
void main_task(uint3 group_id: SV_GroupID, uint lane: SV_GroupIndex, uint draw_index: SV_DrawIndex, uniform PC pc)
{
    run_meshlet_task(pc, 0, draw_index, group_id, lane);
}

VertexOutput forward_vertex(PC pc, GpuDraw draw, float4x4 model, float wind_strength, MeshletDisplace displace,
                            uint vertex_id)
{
    const var ubo = pc.ubo;
    const var vertex = decode_vertex(draw.vertex_address[vertex_id]);

    float4 world_position = mul(model, float4(displace.apply(vertex.position), 1.0));
    world_position.xz += wind_offset(vertex.position, world_position.xyz, wind_strength, ubo.time);

    float3x3 normal_matrix = (float3x3)model;

    float3 world_normal = normalize(mul(normal_matrix, vertex.normal));
    float3 world_tangent = normalize(mul(normal_matrix, vertex.tangent.xyz));

#ifdef BLEND_PASS
    // Water: upward-facing surfaces are lifted by the same wave field the fragment shader shades (water_displacement),
    // so the geometry and the normals agree. The normal stays flat here; main_fs rebuilds it from the field.
    const Material material = pc.materials[draw.material_index];
    if ((material.flags & material_flag_water) != 0) {
        // An ocean's camera-following grid is clipped to its water zone (water_zone: min x, min z, max x, max z):
        // vertices outside fold onto the zone's edge, so the triangles there collapse and draw nothing.
        world_position.xz = clamp(world_position.xz, material.water_zone.xy, material.water_zone.zw);
        // A flat surface moves along +Y, a sphere (a planet's ocean) along its own up.
        const WaterFrame frame = water_frame(material, world_position.xyz);
        if (material.water_shape.x > 0.0 && dot(world_normal, frame.up) > 0.5) {
            world_position.xyz += frame.up * water_displacement(material, frame, ubo.time,
                                                                length(ubo.camera_position - world_position.xyz));
        }
    }
#endif

    VertexOutput output;
    output.position = mul(ubo.view_projection, world_position);
    output.world_position = world_position.xyz;
    output.world_normal = world_normal;
    output.world_tangent = float4(world_tangent, vertex.tangent.w);
    output.texture_coordinate = vertex.uvs;
    output.material_index = draw.material_index;
    output.debug_cluster_id = 0;
    output.blend_faces = blend_faces_both;

    return output;
}

[shader("vertex")]
VertexOutput main_vs(uint vertex_id: SV_VulkanVertexID, uint instance_index: SV_VulkanInstanceID,
                     uint draw_index: SV_DrawIndex, uniform PC pc)
{
    const GpuDraw draw = pc.draws[instance_index];
    const float wind_strength = pc.materials[draw.material_index].wind_strength;

    VertexOutput output = forward_vertex(pc, draw, pc.transforms[instance_index], wind_strength, MeshletDisplace.none(), vertex_id);
    output.debug_cluster_id = instance_index ^ debug_instance_salt;
#ifdef BLEND_PASS
    output.blend_faces = pc.task_commands[draw_index].blend_faces;
#endif
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
    const MeshletDisplace displace = MeshletDisplace.from(pc.materials[draw.material_index], meshlet);

    for (uint v = lane; v < meshlet.vertex_count; v += mesh_group_size) {
        VertexOutput output =
                forward_vertex(pc, draw, model, wind_strength, displace, meshlet_vertex_index(draw, meshlet, v));
        output.debug_cluster_id = meshlet_index;
        output.blend_faces = payload.blend_faces;
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

// Anti-aliased grid line mask for a coordinate measured in cells: 1 on a line, 0 between, fading out when the cells
// get too small on screen to resolve (which would otherwise shimmer).
float grid_line(float2 cell)
{
    const float2 width = max(fwidth(cell), float2(1e-5));
    const float2 to_line = abs(frac(cell - 0.5) - 0.5) / width;
    const float line_mask = 1.0 - saturate(min(to_line.x, to_line.y) - 0.5);
    const float resolvable = saturate(1.0 - 2.0 * max(width.x, width.y));
    return line_mask * resolvable;
}

// What the decals overlapping a fragment contribute, accumulated front to back as premultiplied layers so the result
// is order-correct without knowing the underlying surface yet.
struct DecalLayers
{
    float3 albedo;
    float albedo_coverage;

    float3 normal;
    float normal_coverage;

    float2 metallic_roughness;
    float metallic_roughness_coverage;

    float occlusion;
    float occlusion_coverage;

    float3 emissive;
};

DecalLayers decal_layers_none()
{
    DecalLayers layers;
    layers.albedo = float3(0.0);
    layers.albedo_coverage = 0.0;
    layers.normal = float3(0.0);
    layers.normal_coverage = 0.0;
    layers.metallic_roughness = float2(0.0);
    layers.metallic_roughness_coverage = 0.0;
    layers.occlusion = 0.0;
    layers.occlusion_coverage = 0.0;
    layers.emissive = float3(0.0);
    return layers;
}

// Clustered forward decals: every decal whose box overlaps this fragment's cluster is projected onto the fragment's own
// world position, so no depth read is needed and MSAA works as for any other forward shading.
DecalLayers gather_decals(PC pc, float3 world_position, float3 geometric_normal, float view_z)
{
    const var ubo = pc.ubo;
    DecalLayers layers = decal_layers_none();

    if (ubo.decal_count == 0) {
        return layers;
    }

    // Texture gradients come from the world position's screen-space derivatives, taken here in uniform control flow,
    // so the per-decal samples below can branch freely.
    const float3 dpdx = ddx(world_position);
    const float3 dpdy = ddy(world_position);

    const float4 clip = mul(ubo.view_projection, float4(world_position, 1.0));
    const uint cluster = cluster_index(ubo, clip, view_z);
    const uint count = min(ubo.decal_clusters[cluster], decal_cluster_capacity);
    const uint list_base = cluster_count(ubo) + cluster * decal_cluster_capacity;

    for (uint i = 0; i < count; ++i) {
        const GpuDecal decal = ubo.decals[ubo.decal_clusters[list_base + i]];

        const float3 local = mul(decal.world_to_decal, float4(world_position, 1.0)).xyz;
        const float3 inside = 0.5 - abs(local);

        if (any(inside <= 0.0)) {
            continue;
        }

        const float3 up = decal.normal_opacity.xyz;
        const float facing = dot(geometric_normal, up);
        const float angle_fade = saturate(facing * decal.tangent_fade_scale.w + decal.bitangent_fade_offset.w);
        const float depth_fade = saturate(inside.y * 10.0);
        const float fade = decal.normal_opacity.w * angle_fade * depth_fade;

        if (fade <= 0.0) {
            continue;
        }

        const var material = pc.materials[decal.material_index];
        const float2 uv = float2(local.x + 0.5, local.z + 0.5);

        const float3x3 to_decal = (float3x3)decal.world_to_decal;
        const float3 gradient_x = mul(to_decal, dpdx);
        const float3 gradient_y = mul(to_decal, dpdy);
        const float2 uv_dx = gradient_x.xz;
        const float2 uv_dy = gradient_y.xz;

        const var sampler = samplers[NonUniformResourceIndex(material.sampler_index)];

        const float4 albedo_sample = sampled_2d[NonUniformResourceIndex(material.base_colour_texture)].SampleGrad(
                                             sampler, uv, uv_dx, uv_dy) *
                                     float4(material.colour[0], material.colour[1], material.colour[2], material.colour[3]);
        const float weight = fade * albedo_sample.a;

        layers.albedo = layers.albedo * (1.0 - weight) + albedo_sample.rgb * weight;
        layers.albedo_coverage = layers.albedo_coverage * (1.0 - weight) + weight;

        float2 tangent_xy = sampled_2d[NonUniformResourceIndex(material.normal_texture)]
                                    .SampleGrad(sampler, uv, uv_dx, uv_dy).xy * 2.0 - 1.0;
        tangent_xy *= material.normal_scale;
        const float tangent_z = sqrt(saturate(1.0 - dot(tangent_xy, tangent_xy)));
        const float3 decal_normal = normalize(decal.tangent_fade_scale.xyz * tangent_xy.x +
                                              decal.bitangent_fade_offset.xyz * tangent_xy.y + up * tangent_z);

        layers.normal = layers.normal * (1.0 - weight) + decal_normal * weight;
        layers.normal_coverage = layers.normal_coverage * (1.0 - weight) + weight;

        const float2 metallic_roughness =
                sampled_2d[NonUniformResourceIndex(material.metallic_roughness_texture)]
                        .SampleGrad(sampler, uv, uv_dx, uv_dy).bg *
                float2(material.metallic_factor, material.roughness_factor);

        layers.metallic_roughness = layers.metallic_roughness * (1.0 - weight) + metallic_roughness * weight;
        layers.metallic_roughness_coverage = layers.metallic_roughness_coverage * (1.0 - weight) + weight;

        const float occlusion = lerp(1.0, sampled_2d[NonUniformResourceIndex(material.occlusion_texture)]
                                                  .SampleGrad(sampler, uv, uv_dx, uv_dy).r,
                                     material.occlusion_strength);

        layers.occlusion = layers.occlusion * (1.0 - weight) + occlusion * weight;
        layers.occlusion_coverage = layers.occlusion_coverage * (1.0 - weight) + weight;

        layers.emissive += sampled_2d[NonUniformResourceIndex(material.emissive_texture)]
                                   .SampleGrad(sampler, uv, uv_dx, uv_dy).rgb *
                           float3(material.emissive[0], material.emissive[1], material.emissive[2]) *
                           material.emissive_strength * weight;
    }

    return layers;
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

float3 shade_light(Ptr<UBO> ubo, GpuLight punctual_light, float3 world_position, float3 geometric_normal,
                   float3 shading_normal, float3 view_direction, float3 base_colour, float metallic, float roughness,
                   float fade)
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

    float3 radiance = punctual_light.colour * punctual_light.intensity * distance_falloff * cone_falloff * fade;

    // A light that does not reach this pixel (out of range, outside the cone or faded out) adds nothing: skip its
    // shadow lookup and the BRDF.
    if (radiance.x + radiance.y + radiance.z <= 0.0) {
        return float3(0.0);
    }

    radiance *= local_shadow(ubo, punctual_light, world_position, geometric_normal);

    return shade_punctual(shading_normal, view_direction, light_dir, radiance, base_colour, metallic, roughness);
}

// The overdraw view's colour for a pixel the depth prepass rasterised `count` fragments to: black for none, then blue
// for one, through green and yellow, to red at eight or more.
float3 overdraw_heatmap_colour(uint count)
{
    if (count == 0) {
        return float3(0.0);
    }

    const float t = saturate((float(count) - 1.0) / 7.0);
    return saturate(float3(2.0 * t - 0.5, 1.5 - abs(4.0 * t - 2.0), 1.5 - 3.0 * t));
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

#ifdef BLEND_PASS
    // A double-sided blended mesh is drawn twice, back faces then front faces; a single-sided one keeps its front.
    if ((input.blend_faces == blend_faces_back && is_front_face) ||
        (input.blend_faces == blend_faces_front && !is_front_face)) {
        discard;
    }
#endif

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

    // A planet's surface (material_flag_planet) takes its equirectangular coordinates from the fragment's direction
    // from the planet's centre instead of the vertex (half-precision vertex coordinates step every few texels of an
    // Earth-sized texture). The frame is in the water sphere fields: centre, then the planet's x, y and z axes.
    float2 uv_dx = ddx(uv);
    float2 uv_dy = ddy(uv);
    if ((material.flags & material_flag_planet) != 0) {
        const float3 direction = normalize(input.world_position - material.water_sphere_centre.xyz);
        const float3 local = float3(dot(direction, material.water_east.xyz), dot(direction, material.water_anchor.xyz),
                                    dot(direction, material.water_north.xyz));
        const float pi = 3.14159265358979323846;
        // Longitude grows towards -x (planet::direction_of).
        uv = float2((atan2(-local.x, local.z) + pi) / (2.0 * pi), (0.5 * pi - asin(clamp(local.y, -1.0, 1.0))) / pi);
        uv_dx = ddx(uv);
        uv_dy = ddy(uv);
        // Across the date line u jumps by one; the gradients ignore that jump.
        uv_dx.x -= round(uv_dx.x);
        uv_dy.x -= round(uv_dy.x);
    }

    float2 tangent_normal_xy = normal_texture.Sample(sampler, uv).xy * 2.0 - 1.0;
    tangent_normal_xy *= material.normal_scale;
    float tangent_normal_z = sqrt(saturate(1.0 - dot(tangent_normal_xy, tangent_normal_xy)));
    float3 tangent_normal = normalize(float3(tangent_normal_xy, tangent_normal_z));

    float3 shading_normal = normalize(mul(tangent_normal, tbn));

    float view_z = mul(ubo.view, float4(input.world_position, 1.0)).z;

    const DecalLayers decals = gather_decals(pc, input.world_position, normal, view_z);

    shading_normal = normalize(shading_normal * (1.0 - decals.normal_coverage) + decals.normal);

    float ndotl = saturate(dot(shading_normal, light_direction));

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
    ambient_occlusion = ambient_occlusion * (1.0 - decals.occlusion_coverage) + decals.occlusion;

    float2 screen_uv = input.position.xy / pc.screen_size;
    const var gtao_texture = sampled_2d[NonUniformResourceIndex(pc.ao_texture_index)];
    const var gtao_sampler = samplers[NonUniformResourceIndex(pc.ao_sampler_index)];
    float screen_space_ao = gtao_texture.Sample(gtao_sampler, screen_uv).r;
    ambient_occlusion *= lerp(1.0, screen_space_ao, ubo.ao_intensity);

const var base_sample = texture.SampleGrad(sampler, uv, uv_dx, uv_dy);

if (material.alpha_mode == AlphaMode.mask && (material.flags & material_flag_alpha_to_coverage) == 0 &&
    base_sample.a < material.alpha_cutoff) {
    discard;
}

var base_colour = base_sample.xyz *
                  float3(material.colour[0], material.colour[1], material.colour[2]);

    base_colour = base_colour * (1.0 - decals.albedo_coverage) + decals.albedo;

#ifdef BLEND_PASS
    // Water: the lighting below shades the foam (a rough white surface, lit like anything else); shade_water at the
    // end composes the water itself and mixes the foam in.
    const bool water_surface = (material.flags & material_flag_water) != 0 && ubo.view_mode == view_lit;
    if (water_surface) {
        base_colour = water_foam_albedo;
    }
#endif

    const float3 texture_base_colour = base_colour;

    if (material.debug_meshlet_colours != 0 || ubo.view_mode == view_meshlets) {
        base_colour = debug_meshlet_colour(input.debug_cluster_id);
    } else if (ubo.view_mode == view_lighting_only) {
        base_colour = float3(0.8);
    }

    float3 grid_emission = float3(0.0);

    if ((material.flags & material_flag_grid) != 0) {
        const float2 world_xz = input.world_position.xz;
        const float minor = grid_line(world_xz);
        const float major = grid_line(world_xz / 5.0);
        const float axis_x = saturate(1.0 - abs(world_xz.y) / max(fwidth(world_xz.y), 1e-5));
        const float axis_z = saturate(1.0 - abs(world_xz.x) / max(fwidth(world_xz.x), 1e-5));

        float3 line_colour = float3(0.16, 0.2, 0.28) * minor;
        line_colour = max(line_colour, float3(0.45, 0.55, 0.8) * major);
        line_colour = lerp(line_colour, float3(0.95, 0.3, 0.3), axis_x);
        line_colour = lerp(line_colour, float3(0.3, 0.5, 1.0), axis_z);

        grid_emission = line_colour;
    }

    float2 metallic_roughness = metallic_roughness_texture.Sample(sampler, uv).bg *
                                float2(material.metallic_factor, material.roughness_factor);
    metallic_roughness = metallic_roughness * (1.0 - decals.metallic_roughness_coverage) + decals.metallic_roughness;
    float metallic = metallic_roughness.x;
    float roughness = clamp(metallic_roughness.y, 0.045, 1.0);

#ifdef BLEND_PASS
    if (water_surface) {
        metallic = 0.0;
        roughness = water_foam_roughness;
    }
#endif

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
            punctual_contribution += shade_light(ubo, light, input.world_position, normal, shading_normal,
                                                 view_direction, base_colour, metallic, roughness,
                                                 light_screen_fade(ubo, light));

            if (lane_next == light_index) {
                ++cursor;
                lane_next = cursor < stored_count ? pc.cluster_lights[list_base + cursor] : list_end;
            }
        }
    } else {
        for (uint light_index = 0; light_index < pc.light_count; ++light_index) {
            const GpuLight light = pc.lights[light_index];
            punctual_contribution += shade_light(ubo, light, input.world_position, normal, shading_normal,
                                                 view_direction, base_colour, metallic, roughness,
                                                 light_screen_fade(ubo, light));
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
                  material.emissive_strength + decals.emissive;

    float3 shaded_colour = ambient + directional_contribution + punctual_contribution + emissive + grid_emission;

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

    // Data views replace the lit result outright (no fog), the way a viewport shading mode does in a DCC tool.
    if (ubo.view_mode == view_base_colour) {
        fogged_colour = texture_base_colour;
    } else if (ubo.view_mode == view_normals) {
        fogged_colour = shading_normal * 0.5 + 0.5;
    } else if (ubo.view_mode == view_roughness) {
        fogged_colour = float3(roughness);
    } else if (ubo.view_mode == view_metallic) {
        fogged_colour = float3(metallic);
    } else if (ubo.view_mode == view_ambient_occlusion) {
        fogged_colour = float3(ambient_occlusion);
    } else if (ubo.view_mode == view_overdraw && ubo.overdraw_width != 0) {
        const uint2 pixel = uint2(input.position.xy);
        const uint slot = pixel.y * ubo.overdraw_width + pixel.x;
        fogged_colour = overdraw_heatmap_colour(slot < ubo.overdraw_size ? ubo.overdraw_counts[slot] : 0u);
    }

    OutputFS output;
    output.colour = float4(fogged_colour, base_sample.a * material.colour[3]);

#ifdef BLEND_PASS
    if (water_surface) {
        const WaterShade water = shade_water(pc, material, input.world_position, normal, input.position.xy, view_z,
                                             shadow);
        output.colour = float4(lerp(water.colour, fogged_colour, water.foam), max(water.alpha, water.foam));
    }
#endif
#ifdef OUTLINE_MASK
    output.outline = (material.flags & material_flag_outlined) != 0 ? 1.0 : 0.0;
#endif

    return output;
}