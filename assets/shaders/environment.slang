module environment;

import bindless;
import scene_types;

public static const float env_pi = 3.14159265358979323846;

public static const uint env_ibl_valid = 1u << 0;
public static const uint env_skybox = 1u << 1;
public static const uint env_sky_procedural = 1u << 2;
public static const uint env_specular_occlusion = 1u << 3;
public static const uint env_fog_sky = 1u << 4;
public static const uint env_multi_scatter = 1u << 5;
public static const uint env_fog_from_environment = 1u << 6;

public static const uint env_debug_shift = 8;
public static const uint env_debug_none = 0;
public static const uint env_debug_diffuse_only = 1;
public static const uint env_debug_specular_only = 2;
public static const uint env_debug_specular_occlusion = 3;
public static const uint env_debug_sky_prefilter = 4;
public static const uint env_debug_sky_irradiance = 5;
public static const uint env_debug_white_furnace = 6;

public uint env_debug_view(uint flags)
{
    return (flags >> env_debug_shift) & 0xFu;
}

public float3 cube_texel_direction(uint face, float2 uv)
{
    const float s = 2.0 * uv.x - 1.0;
    const float t = 2.0 * uv.y - 1.0;

    float3 direction;

    switch (face) {
    case 0:
        direction = float3(1.0, -t, -s);
        break;
    case 1:
        direction = float3(-1.0, -t, s);
        break;
    case 2:
        direction = float3(s, 1.0, t);
        break;
    case 3:
        direction = float3(s, -1.0, -t);
        break;
    case 4:
        direction = float3(s, -t, 1.0);
        break;
    default:
        direction = float3(-s, -t, -1.0);
        break;
    }

    return normalize(direction);
}

public float2 direction_to_equirect_uv(float3 direction)
{
    const float phi = atan2(direction.x, direction.z);
    return float2(0.5 + phi / (2.0 * env_pi), acos(clamp(direction.y, -1.0, 1.0)) / env_pi);
}

public float3 rotate_y(float3 v, float2 cos_sin)
{
    return float3(cos_sin.x * v.x + cos_sin.y * v.z, v.y, -cos_sin.y * v.x + cos_sin.x * v.z);
}

float corner_solid_angle(float x, float y)
{
    return atan2(x * y, sqrt(x * x + y * y + 1.0));
}

public float cube_texel_solid_angle(uint x, uint y, uint size)
{
    const float step = 2.0 / float(size);
    const float x0 = float(x) * step - 1.0;
    const float y0 = float(y) * step - 1.0;
    const float x1 = x0 + step;
    const float y1 = y0 + step;

    return corner_solid_angle(x1, y1) - corner_solid_angle(x0, y1) - corner_solid_angle(x1, y0) +
           corner_solid_angle(x0, y0);
}

public float2 hammersley(uint index, uint count)
{
    return float2(float(index) / float(count), float(reversebits(index)) * 2.3283064365386963e-10);
}

public float3 importance_sample_ggx(float2 xi, float alpha)
{
    const float phi = 2.0 * env_pi * xi.x;
    const float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    const float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

public float distribution_ggx_alpha(float ndoth, float alpha)
{
    const float a2 = alpha * alpha;
    const float denominator = ndoth * ndoth * (a2 - 1.0) + 1.0;

    return a2 / max(env_pi * denominator * denominator, 1e-8);
}

public void tangent_frame(float3 n, out float3 tangent, out float3 bitangent)
{
    const float3 up = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);

    tangent = normalize(cross(up, n));
    bitangent = cross(n, tangent);
}

public struct Sh9Basis
{
    public float values[9];
};

public Sh9Basis sh9_basis(float3 d)
{
    Sh9Basis basis;

    basis.values[0] = 0.282095;
    basis.values[1] = 0.488603 * d.y;
    basis.values[2] = 0.488603 * d.z;
    basis.values[3] = 0.488603 * d.x;
    basis.values[4] = 1.092548 * d.x * d.y;
    basis.values[5] = 1.092548 * d.y * d.z;
    basis.values[6] = 0.315392 * (3.0 * d.z * d.z - 1.0);
    basis.values[7] = 1.092548 * d.x * d.z;
    basis.values[8] = 0.546274 * (d.x * d.x - d.y * d.y);

    return basis;
}

public float3 sh9_eval(Ptr<GpuEnvironmentSh> sh, float3 normal)
{
    const Sh9Basis basis = sh9_basis(normal);

    float3 result = float3(0.0);

    [unroll]
    for (uint i = 0; i < 9; ++i) {
        result += sh.coefficients[i].xyz * basis.values[i];
    }

    return result;
}

public float3 preetham_sky(Ptr<UBO> ubo, float3 direction, float3 sun_direction)
{
    const float cos_theta = max(direction.y, 1e-2);
    const float cos_gamma = clamp(dot(direction, sun_direction), -1.0, 1.0);
    const float gamma = acos(cos_gamma);

    float3 xyy;

    [unroll]
    for (uint channel = 0; channel < 3; ++channel) {
        const float4 abcd = ubo.sky_perez[channel];
        const float e = ubo.sky_perez[3][channel];

        const float factor = (1.0 + abcd.x * exp(abcd.y / cos_theta)) *
                             (1.0 + abcd.z * exp(abcd.w * gamma) + e * cos_gamma * cos_gamma);

        xyy[channel] = ubo.sky_zenith[channel] * factor;
    }

    const float luminance = max(xyy.x, 0.0);
    const float scale = luminance / max(xyy.z, 1e-4);

    const float big_x = xyy.y * scale;
    const float big_z = (1.0 - xyy.y - xyy.z) * scale;

    return max(float3(3.2406 * big_x - 1.5372 * luminance - 0.4986 * big_z,
                      -0.9689 * big_x + 1.8758 * luminance + 0.0415 * big_z,
                      0.0557 * big_x - 0.2040 * luminance + 1.0570 * big_z),
               float3(0.0));
}

public float3 environment_sky(Ptr<UBO> ubo, float3 direction)
{
    const float3 sun_direction = ubo.sun_direction_cos_radius.xyz;
    const float3 sky = preetham_sky(ubo, direction, sun_direction);

    if (direction.y >= 0.0) {
        return sky;
    }

    const float3 horizon = preetham_sky(ubo, float3(1.0, 0.0, 0.0), sun_direction);
    const float3 ground = ubo.sky_ground.xyz * horizon;

    const float blend = clamp(-direction.y / 0.0349066, 0.0, 1.0);
    const float eased = blend * blend * (3.0 - 2.0 * blend);

    return lerp(horizon, ground, eased);
}

public float specular_occlusion(float ndotv, float ambient_occlusion, float roughness)
{
    const float alpha = roughness * roughness;

    return saturate(pow(ndotv + ambient_occlusion, exp2(-16.0 * alpha - 1.0)) - 1.0 + ambient_occlusion);
}

public float3 environment_fog_colour(Ptr<UBO> ubo, float3 view_ray)
{
    float3 direction = float3(view_ray.x, view_ray.y * 0.25, view_ray.z);
    direction = rotate_y(normalize(direction), float2(ubo.environment_rotation.x, -ubo.environment_rotation.y));

    const TextureCube<float4> prefilter = sampled_cube[ubo.prefilter_cube_texture];
    const SamplerState sampler = samplers[ubo.environment_sampler];

    return prefilter.SampleLevel(sampler, direction, ubo.environment_rotation.z).rgb * ubo.environment_exposure *
           ubo.environment_intensity.y;
}
