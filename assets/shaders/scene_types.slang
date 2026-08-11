module scene_types;

public static const uint shadow_cascade_count = 4;

public struct GpuDraw
{
    public Ptr<CompressedVertex> vertex_address;

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

public enum class LightType : uint {
    point,
    spot,
};

// One punctual light. Mirrors GpuLight in include/renderer.hxx. direction is
// unused for point lights. spot_scale/spot_offset are the glTF-style
// precomputed cone falloff terms (see KHR_lights_punctual):
// angular_attenuation = saturate(dot(spot_dir, -light_to_surface) * spot_scale + spot_offset),
// computed on the CPU so the shader never evaluates cos()/acos() per pixel.
public struct GpuLight {
    public float3 position;
    public float range;

    public float3 colour;
    public float intensity;

    public float3 direction; // normalized, points from the light outward -- spot lights only
    public float spot_scale;

    public float spot_offset;
    public LightType type;
    public float _pad0;
    public float _pad1;
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

    // Flat multiplier on the (already-albedo-scaled) ambient term -- see the
    // comment on this field in include/renderer.hxx for why it exists.
    public float ambient_intensity;
};

public struct PC
{
    public Ptr<GpuDraw> draws;
    public Ptr<float4x4> transforms;
    public Ptr<Material> materials;
    public Ptr<UBO> ubo;

    // Punctual (point/spot) lights, in a standalone buffer rather than a
    // trailing UBO array -- same reasoning as frustum_planes_buffer (see
    // include/renderer.hxx): unambiguous 16-byte-stride Ptr<> indexing
    // instead of staking correctness on Slang's struct-array packing.
    public Ptr<GpuLight> lights;
    public uint light_count;
    public uint _padding;
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