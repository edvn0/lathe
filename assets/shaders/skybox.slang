module skybox;

import bindless;
import scene_types;
import environment;

struct SkyboxPushConstants
{
    Ptr<UBO> ubo;
};

[vk::push_constant] SkyboxPushConstants pc;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0;
};

[shader("vertex")]
VertexOutput main_vs(uint vertex_id: SV_VertexID)
{
    float2 position;

    switch (vertex_id) {
    case 0:
        position = float2(-1.0, -1.0);
        break;
    case 1:
        position = float2(-1.0, 3.0);
        break;
    default:
        position = float2(3.0, -1.0);
        break;
    }

    VertexOutput output;
    output.position = float4(position, 1.0, 1.0);
    output.ndc = position;

    return output;
}

struct FragmentOutput
{
    float4 colour : SV_Target0;
};

[shader("fragment")]
FragmentOutput main_fs(VertexOutput input)
{
    const var ubo = pc.ubo;

    const float4 view_point = mul(ubo.inverse_projection, float4(input.ndc, 1.0, 1.0));
    const float3 view_ray = view_point.xyz / view_point.w;
    const float3 ray = normalize(mul(transpose((float3x3)ubo.view), view_ray));

    float3 colour;

    const uint flags = ubo.environment_flags;
    const uint debug_view = env_debug_view(flags);

    if ((flags & env_sky_procedural) != 0) {
        colour = environment_sky(ubo, ray);

        if (dot(ray, ubo.sun_direction_cos_radius.xyz) > ubo.sun_direction_cos_radius.w) {
            colour = ubo.sun_disc_radiance.rgb;
        }
    } else {
        const TextureCube<float4> radiance = sampled_cube[ubo.radiance_cube_texture];
        const SamplerState sampler = samplers[ubo.environment_sampler];

        const float3 direction = rotate_y(ray, float2(ubo.environment_rotation.x, -ubo.environment_rotation.y));

        colour = radiance.SampleLevel(sampler, direction, 0).rgb;
    }

    if (debug_view == env_debug_sky_prefilter) {
        const TextureCube<float4> prefilter = sampled_cube[ubo.prefilter_cube_texture];
        const SamplerState sampler = samplers[ubo.environment_sampler];

        colour = prefilter.SampleLevel(sampler, rotate_y(ray, float2(ubo.environment_rotation.x, -ubo.environment_rotation.y)),
                                       ubo.environment_intensity.w).rgb;
    } else if (debug_view == env_debug_sky_irradiance) {
        colour = sh9_eval(ubo.environment_sh, rotate_y(ray, float2(ubo.environment_rotation.x, -ubo.environment_rotation.y)));
    }

    FragmentOutput output;
    output.colour = float4(colour * ubo.environment_exposure, 1.0);

    return output;
}
