module scene_types;

public static const uint shadow_cascade_count = 4;

public struct Vertex
{
    public float3 position;
    public float3 normal;
    public float4 tangent;
    public float2 uvs;
};

public struct GpuDraw
{
    public Ptr<Vertex> vertex_address;

    public uint material_index;
    public uint transform_index;
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

    // Blade sway amplitude driven by UBO.time in the vertex shader; 0 for
    // every ordinary (non-foliage) material -- see wind.slang.
    public float wind_strength = 0.0F;
    public float _pad0 = 0.0F;
    public float _pad1 = 0.0F;
    public float _pad2 = 0.0F;
};

public struct UBO {
    public float4x4 view_projection;
    public float4x4 view;
    public float4x4 projection;

    public float3 camera_position;
    public float3 fog_colour;
    public float fog_extinction;     // A
    public float fog_inscattering;   // B

    // ---- PSSM cascades ----
    public float4x4 cascade_view_projection[shadow_cascade_count];
    public float4 cascade_split_far;   // view-space far distance per cascade
    public float4 cascade_texel_world; // world-space size of one shadow texel per cascade
    public float4 cascade_depth_scale; // 1 / (z_far - z_near) per cascade

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

    public float time; // seconds since startup -- drives wind sway in vertex shaders
};

public struct PC
{
    public Ptr<GpuDraw> draws;
    public Ptr<float4x4> transforms;
    public Ptr<Material> materials;
    public Ptr<UBO> ubo;
};

public struct ShadowPC
{
    public PC base;
    public uint cascade_index;
    public uint padding;
};
