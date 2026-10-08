module scene_types;

import hiz_occlusion;

public static const uint shadow_cascade_count = 4;

public static const uint meshlet_max_vertices = 64;
public static const uint meshlet_max_triangles = 124;

public static const uint meshlets_per_task = 32;

public static const uint mesh_group_size = 32;

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

public static const uint cull_frustum_bit = 1u;
public static const uint cull_backface_bit = 2u;
public static const uint cull_occlusion_bit = 4u;
public static const uint cull_skip_recorded_bit = 8u;
public static const uint cull_record_bit = 16u;
public static const uint cull_replay_bit = 32u;
public static const uint cull_stats_bit = 64u;

public struct GpuDraw
{
    public Ptr<CompressedVertex> vertex_address;
    public Ptr<Meshlet> meshlets;
    public Ptr<uint> meshlet_data;

    public uint material_index;

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

    public float wind_strength = 0.0F;
    public float _pad0 = 0.0F;

    public uint debug_meshlet_colours = 0;

    public uint flags = 0;
};

public static const uint material_flag_double_sided = 1u;

public static const uint material_flag_alpha_to_coverage = 2u;

public static const uint material_flag_outlined = 4u;

public static const uint cluster_grid_maximum_depth_slices = 64;

public enum class LightType : uint {
    point,
    spot,
};

public struct GpuLight {
    public float3 position;
    public float range;

    public float3 colour;
    public float intensity;

    public float3 direction;
    public float spot_scale;

    public float spot_offset;
    public LightType type;
    public float _pad0;
    public float _pad1;
};

public struct GpuEnvironmentSh
{
    public float4 coefficients[9];
};

public struct UBO {
    public float4x4 view_projection;
    public float4x4 view;
    public float4x4 projection;

    public float4x4 inverse_projection;

    public float3 camera_position;
    public float3 fog_colour;
    public float fog_extinction;
    public float fog_inscattering;

    public float4x4 cascade_view_projection[shadow_cascade_count];
    public float4 cascade_split_far;
    public float4 cascade_texel_world;
    public float4 cascade_depth_scale;

    public float4 cascade_atlas_offset_u;
    public float4 cascade_atlas_scale_u;
    public float4 cascade_atlas_scale_v;

    public float3 light_direction;
    public float light_intensity;
    public float3 light_colour;
    public float shadow_normal_offset_texels;

    public uint shadow_atlas_texture;
    public uint shadow_sampler;
    public float shadow_depth_bias_world;
    public float shadow_pcf_radius_texels;

    public uint cascade_count;
    public float shadow_atlas_texel_u;
    public float shadow_atlas_texel_v;
    public uint shadow_debug_cascade_tint;

    public float time;

    public float ambient_intensity;

    public float ao_intensity;

    public float cluster_z_scale;
    public float cluster_z_bias;

    public uint clustered_lighting;

    public uint cluster_debug_heatmap;

    public uint cluster_grid_x;
    public uint cluster_grid_y;
    public uint cluster_grid_z;
    public uint cluster_light_capacity;

    public float light_lod_pixel_scale;
    public float light_lod_cull_radius_pixels;
    public float light_lod_fade_radius_pixels;

    public uint environment_flags;
    public float environment_exposure;
    public float4 environment_rotation;
    public float4 environment_intensity;
    public uint radiance_cube_texture;
    public uint prefilter_cube_texture;
    public uint brdf_lut_texture;
    public uint environment_sampler;
    public float4 sky_perez[4];
    public float4 sky_zenith;
    public float4 sky_ground;
    public float4 sun_direction_cos_radius;
    public float4 sun_disc_radiance;
    public Ptr<GpuEnvironmentSh> environment_sh;
};

public struct PC
{
    public Ptr<GpuDraw> draws;
    public Ptr<float4x4> transforms;
    public Ptr<Material> materials;
    public Ptr<UBO> ubo;

    public Ptr<GpuLight> lights;
    public uint light_count;

    public uint sample_count;

    public uint ao_texture_index;
    public uint ao_sampler_index;
    public float2 screen_size;

    public Ptr<DrawCommand> task_commands;
    public Ptr<float4> cull_planes;
    public uint cull_flags;
    public uint cull_padding;

    public Ptr<uint> cluster_lights;

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
    public uint normal_oct;
    public uint tangent_oct;
    public uint16_t position_x, position_y, position_z;
    public uint16_t texcoord_u, texcoord_v;
    public uint16_t pad_;
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
    public float4 tangent;
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