module scene_types;

import hiz_occlusion;

public static const uint shadow_cascade_count = 4;

// Mirror meshlet_max_vertices/meshlet_max_triangles/meshlets_per_task in include/assets/meshlet.hxx.
public static const uint meshlet_max_vertices = 64;
public static const uint meshlet_max_triangles = 124;

// Task workgroup size: one lane culls one meshlet.
public static const uint meshlets_per_task = 32;

// Mesh workgroup size: lanes stride over a meshlet's vertices/triangles.
public static const uint mesh_group_size = 32;

// Mirrors GpuMeshlet in meshlet.hxx. Bounds are local space; cone_cutoff >= 1 disables cone culling.
// vertex_offset/triangle_offset index GpuDraw::meshlet_data in uints.
public struct Meshlet
{
    public float3 centre;
    public float radius;
    public float3 cone_axis;
    public float cone_cutoff;
    public uint vertex_offset;
    public uint triangle_offset;
    public uint vertex_count;
    public uint triangle_count;
};

// Mirrors GpuDrawCommand in meshlet.hxx: a VkDrawMeshTasksIndirectCommandEXT at offset 0 and a
// VkDrawIndexedIndirectCommand at offset 12. Each batch uses one and zeroes the other.
//
// A meshlet batch needs instance_count * ceil(meshlet_count / meshlets_per_task) task groups, spread over X and
// Y so neither exceeds the guaranteed 65535 limit; groups past the total are no-ops.
public struct DrawCommand
{
    public uint group_count_x;
    public uint group_count_y;
    public uint group_count_z;

    public uint index_count;
    public uint instance_count;
    public uint first_index;
    public int vertex_offset;
    public uint first_instance;

    public uint meshlet_count;
    public uint _pad0;
};

// PC::cull_flags bits.
public static const uint cull_frustum_bit = 1u;
// Normal-cone culling. Only valid with back-face culling, and must match between prepass and forward, which
// depth-tests EQUAL.
public static const uint cull_backface_bit = 2u;
// Meshlet-level Hi-Z occlusion (docs/occlusion-culling.md, "Meshlet level"); all need PC::occlusion. Only the main view sets them.
// Test the meshlet's sphere against *PC::occlusion.
public static const uint cull_occlusion_bit = 4u;
// Phase 2: skip meshlets phase 1 already emitted (their bit is set), so the union range draws each once.
public static const uint cull_skip_recorded_bit = 8u;
// Set the meshlet's bit in OcclusionView::meshlet_visibility when it is emitted.
public static const uint cull_record_bit = 16u;
// Forward: emit exactly the meshlets whose bit is set and run no other test, so forward draws what the prepass
// drew.
public static const uint cull_replay_bit = 32u;
// Add this task group's occlusion rejections to PC::occlusion->stats[0].
public static const uint cull_stats_bit = 64u;

public struct GpuDraw
{
    public Ptr<CompressedVertex> vertex_address;
    public Ptr<Meshlet> meshlets;
    public Ptr<uint> meshlet_data;

    public uint material_index;

    // The absolute bit index of the instance's meshlet 0 in the frame's meshlet visibility bitset (instance *
    // meshlet_count further on for each following instance of the batch); 0 and never read for instanced batches.
    public uint meshlet_visibility_offset;
};

public enum class AlphaMode : uint {
    opaque,
    mask,
    blend,
};

public struct Material {
    public float colour[4];
    public float emissive[3];
    public float emissive_strength = 0.0F;
    public float metallic_factor = 1.0F;
    public float roughness_factor = 1.0F;
    public float normal_scale = 1.0F;
    public float occlusion_strength = 1.0F;

    public uint base_colour_texture = 0;
    public uint normal_texture = 2;
    public uint metallic_roughness_texture = 3;
    public uint occlusion_texture = 4;

    public uint emissive_texture = 5;
    public uint sampler_index = 0;

    public AlphaMode alpha_mode = AlphaMode.opaque;
    public float alpha_cutoff = 0.5F;

    // Sway amplitude; 0 for non-foliage materials.
    public float wind_strength = 0.0F;
    public float _pad0 = 0.0F;

    // Non-zero shades with a per-meshlet colour instead of the base colour.
    public uint debug_meshlet_colours = 0;

    // material_flag_* bits. Mirrors GpuMaterial::flags.
    public uint flags = 0;
};

// Drawn without back-face culling; back faces shade with the normal flipped.
public static const uint material_flag_double_sided = 1u;

// Mask materials under MSAA: the prepass writes alpha as sample coverage instead of cutting at alpha_cutoff.
public static const uint material_flag_alpha_to_coverage = 2u;

// Clustered lighting: the grid is set at runtime (UBO::cluster_grid_*, see ClusterGridSettings in cluster_grid.hxx).
// light_cluster.slang sizes its shared per-slice arrays by this. Mirrors cluster_grid_maximum_depth_slices.
public static const uint cluster_grid_maximum_depth_slices = 64;

public enum class LightType : uint {
    point,
    spot,
};

// Mirrors GpuLight in renderer.hxx. direction is unused for point lights. Spot falloff is precomputed
// (KHR_lights_punctual): saturate(dot(spot_dir, -light_to_surface) * spot_scale + spot_offset).
public struct GpuLight {
    public float3 position;
    public float range;

    public float3 colour;
    public float intensity;

    public float3 direction; // normalized, points from the light outward; spot lights only
    public float spot_scale;

    public float spot_offset;
    public LightType type;
    public float _pad0;
    public float _pad1;
};

// Nine RGB spherical-harmonics coefficients, premultiplied by the clamped-cosine lobe and 1/pi, so that sh9_eval()
// returns irradiance / pi. Mirrors GpuEnvironmentSh in environment.hxx.
public struct GpuEnvironmentSh
{
    public float4 coefficients[9];
};

public struct UBO {
    public float4x4 view_projection;
    public float4x4 view;
    public float4x4 projection;

    // Lets screen-space passes unproject depth without assuming a projection convention.
    public float4x4 inverse_projection;

    public float3 camera_position;
    public float3 fog_colour;
    public float fog_extinction;
    public float fog_inscattering;

    // PSSM cascades.
    public float4x4 cascade_view_projection[shadow_cascade_count];
    public float4 cascade_split_far;   // view-space far distance per cascade
    public float4 cascade_texel_world; // world-space size of one shadow texel per cascade
    public float4 cascade_depth_scale; // 1 / (z_far - z_near) per cascade

    // Atlas tiles have variable widths, so these map a cascade-local [0,1] UV into the atlas.
    public float4 cascade_atlas_offset_u; // atlas-normalized U of each tile's left edge
    public float4 cascade_atlas_scale_u;  // tile width / atlas width, per cascade
    public float4 cascade_atlas_scale_v;  // tile height / atlas height, per cascade

    public float3 light_direction;     // normalized, points from surface to light
    public float light_intensity;
    public float3 light_colour;
    public float shadow_normal_offset_texels;

    public uint shadow_atlas_texture;  // bindless sampled_2d index
    public uint shadow_sampler;        // bindless comparison_samplers index
    public float shadow_depth_bias_world;
    public float shadow_pcf_radius_texels;

    public uint cascade_count;
    public float shadow_atlas_texel_u;
    public float shadow_atlas_texel_v;
    public uint shadow_debug_cascade_tint;

    public float time; // seconds since startup

    // Flat multiplier on the ambient term, standing in for IBL.
    public float ambient_intensity;

    // 0 disables screen-space AO, 1 applies it at full strength.
    public float ao_intensity;

    // Exponential depth slicing for the light clusters: slice = floor(log(view_z) * scale + bias).
    public float cluster_z_scale;
    public float cluster_z_bias;

    // 0 shades every light per fragment; otherwise the forward pass reads PC::cluster_lights.
    public uint clustered_lighting;

    // Non-zero tints the forward pass by the number of lights in each fragment's cluster.
    public uint cluster_debug_heatmap;

    // The cluster grid: NDC tiles by exponential depth slices, each cluster a sorted list of up to
    // cluster_light_capacity light indices.
    public uint cluster_grid_x;
    public uint cluster_grid_y;
    public uint cluster_grid_z;
    public uint cluster_light_capacity;

    // Screen-size light LOD, see light_screen_fade() in light_clusters.slang. fade of 0 disables it.
    public float light_lod_pixel_scale;
    public float light_lod_cull_radius_pixels;
    public float light_lod_fade_radius_pixels;

    // Environment lighting and the skybox (docs/ibl-and-skybox.md). Mirrors UBO in renderer.hxx. The flag bits are the
    // env_* constants in environment.slang.
    public uint environment_flags;
    public float environment_exposure;               // 2^EV, for both the skybox and IBL
    public float4 environment_rotation;              // cos(theta), sin(theta), prefilter max LOD, specular occlusion strength
    public float4 environment_intensity;             // diffuse, specular, unused, debug LOD
    public uint radiance_cube_texture;               // bindless sampled_cube indices
    public uint prefilter_cube_texture;
    public uint brdf_lut_texture;                    // bindless sampled_2d index
    public uint environment_sampler;                 // bindless samplers index (linear, clamp)
    public float4 sky_perez[4];                      // Perez A-D for Y, x, y; row 3 = (E_Y, E_x, E_y, 0)
    public float4 sky_zenith;                        // (Y, x, y) over F(0, theta_sun); Y carries the calibration
    public float4 sky_ground;                        // rgb ground albedo
    public float4 sun_direction_cos_radius;          // xyz toward the sun, w = cos(angular radius)
    public float4 sun_disc_radiance;                 // rgb
    public Ptr<GpuEnvironmentSh> environment_sh;     // live SH slot
};

public struct PC
{
    public Ptr<GpuDraw> draws;
    public Ptr<float4x4> transforms;
    public Ptr<Material> materials;
    public Ptr<UBO> ubo;

    // A pointer rather than a UBO array for an unambiguous 16-byte stride.
    public Ptr<GpuLight> lights;
    public uint light_count;

    // Samples per pixel of the target being drawn; only read by the masked depth prepass, for alpha to coverage.
    public uint sample_count;

    // GTAO texture and render resolution; only read by the forward fragment shader. Kept on the shared PC so every
    // pass's push constants stay layout-compatible.
    public uint ao_texture_index;
    public uint ao_sampler_index;
    public float2 screen_size;

    // Task-shader inputs: the indirect commands (indexed by SV_DrawIndex), the cull planes (6 per view) and the
    // enabled culling tests.
    public Ptr<DrawCommand> task_commands;
    public Ptr<float4> cull_planes;
    public uint cull_flags;
    public uint cull_padding;

    // Per-cluster light lists from light_cluster.slang: cluster_count(ubo) counts, then cluster_count(ubo) *
    // UBO::cluster_light_capacity light indices. Only read by the forward fragment shader when
    // UBO::clustered_lighting is set.
    public Ptr<uint> cluster_lights;

    // The Hi-Z view and meshlet visibility bitset read when cull_flags has an occlusion bit; null otherwise (the
    // shadow pass never sets one).
    public Ptr<OcclusionView> occlusion;
};

public struct ShadowPC
{
    public PC base;
    public uint cascade_index;
    public uint padding;
};

public struct CompressedVertex
{
    uint normal_oct;
    uint tangent_oct;
    uint16_t position_x, position_y, position_z;
    uint16_t texcoord_u, texcoord_v;
    uint16_t pad_;
};

public float3 decode_octahedral(float2 encoded)
{
    float3 direction = float3(encoded.x, encoded.y, 1.0 - abs(encoded.x) - abs(encoded.y));
    float t = max(-direction.z, 0.0);

    direction.x += direction.x >= 0.0 ? -t : t;
    direction.y += direction.y >= 0.0 ? -t : t;

    return normalize(direction);
}

public float2 unpack_snorm2x16(uint packed)
{
    int2 signed_value = int2((int)(packed << 16) >> 16, (int)packed >> 16);
    return clamp(float2(signed_value) / 32767.0, -1.0, 1.0);
}

public struct DecodedVertex
{
    public float3 position;
    public float3 normal;
    public float4 tangent; // xyz = direction, w = handedness sign
    public float2 uvs;
};

public DecodedVertex decode_vertex(CompressedVertex vertex)
{
    DecodedVertex decoded;

    decoded.position = float3(f16tof32(vertex.position_x), f16tof32(vertex.position_y), f16tof32(vertex.position_z));
    decoded.uvs = float2(f16tof32(vertex.texcoord_u), f16tof32(vertex.texcoord_v));

    decoded.normal = decode_octahedral(unpack_snorm2x16(vertex.normal_oct));

    bool tangent_sign = (vertex.tangent_oct & 1) != 0;
    decoded.tangent.xyz = decode_octahedral(unpack_snorm2x16(vertex.tangent_oct & 0xFFFFFFFEu));
    decoded.tangent.w = tangent_sign ? -1.0 : 1.0;

    return decoded;
}