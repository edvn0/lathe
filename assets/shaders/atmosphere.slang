module atmosphere;

// A planet's atmosphere as a post-process after lighting (rendering/planet_system.cxx, docs/globe.md): for every pixel
// the view ray is marched through a Rayleigh + Mie shell around the planet up to the scene depth, the scene behind is
// dimmed by the transmittance and the light scattered towards the camera is added. Where nothing was drawn (depth 0)
// the background is space: black with stars. Lengths are in metres, intersections are done in units of the planet's
// radius so the floats stay well conditioned at real scale.

import bindless;

// Mirrors AtmosphereParams in planet_system.cxx.
struct AtmospherePush
{
    uint colour;
    uint depth;
    uint result;
    uint flags;
    float4x4 inverse_view_projection;
    float3 planet_centre;
    float planet_radius;
    float3 sun_direction;
    float atmosphere_radius;
    float sun_intensity;
    float intensity;
    float time;
    float padding;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] AtmospherePush pc;

static const float pi = 3.14159265358979323846;
// Sea-level scattering coefficients (1/m) and scale heights (m) for Earth.
static const float3 rayleigh_scattering = float3(5.802e-6, 13.558e-6, 33.1e-6);
static const float rayleigh_height = 8000.0;
static const float mie_scattering = 3.996e-6;
static const float mie_extinction = 4.44e-6;
static const float mie_height = 1200.0;
static const float mie_g = 0.8;
static const uint view_steps = 16;
static const uint light_steps = 6;

// Distances along a unit-direction ray to a sphere about the origin, or (1, -1) when it misses.
float2 ray_sphere(float3 origin, float3 direction, float radius)
{
    const float b = dot(origin, direction);
    const float c = dot(origin, origin) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0) {
        return float2(1.0, -1.0);
    }
    const float root = sqrt(discriminant);
    return float2(-b - root, -b + root);
}

float3 unproject(float2 ndc, float ndc_z)
{
    const float4 point = mul(pc.inverse_view_projection, float4(ndc, ndc_z, 1.0));
    return point.xyz / point.w;
}

float hash(float2 p)
{
    return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
}

// A fixed star field on the sky sphere.
float stars(float3 direction)
{
    const float2 angles = float2(atan2(direction.z, direction.x), asin(clamp(direction.y, -1.0, 1.0)));
    const float2 cell = floor(angles * float2(400.0, 400.0));
    const float value = hash(cell);
    if (value < 0.9965) {
        return 0.0;
    }
    const float2 within = frac(angles * 400.0) - 0.5;
    const float brightness = (value - 0.9965) / 0.0035;
    return brightness * brightness * exp(-dot(within, within) * 40.0) * 0.6;
}

// Optical depth (Rayleigh, Mie) from `point` towards the sun to the top of the atmosphere; huge when the planet is in
// the way. `point` is in units of the planet's radius from its centre.
float2 sun_optical_depth(float3 point, float3 sun, float top, float radius)
{
    const float2 planet = ray_sphere(point, sun, 1.0);
    if (planet.x > 0.0 && planet.x <= planet.y) {
        return float2(1e9);
    }
    const float2 shell = ray_sphere(point, sun, top);
    const float span = max(shell.y, 0.0);
    const float step = span / float(light_steps);
    float2 depth = float2(0.0);
    for (uint i = 0; i < light_steps; ++i) {
        const float3 sample_point = point + sun * (step * (float(i) + 0.5));
        const float height = (length(sample_point) - 1.0) * radius;
        depth += float2(exp(-height / rayleigh_height), exp(-height / mie_height));
    }
    return depth * step * radius;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 thread_id: SV_DispatchThreadID)
{
    uint width, height;
    sampled_2d[pc.colour].GetDimensions(width, height);
    if (thread_id.x >= width || thread_id.y >= height) {
        return;
    }

    const float2 uv = (float2(thread_id.xy) + 0.5) / float2(width, height);
    const float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const float stored_depth = sampled_2d_depth[pc.depth].Load(int3(thread_id.xy, 0));
    float4 scene = sampled_2d[pc.colour].Load(int3(thread_id.xy, 0));

    // The depth buffer is reversed (1 near, 0 far): the near plane is ndc z 0, the far plane 1.
    const float3 near_point = unproject(ndc, 0.0);
    const bool sky = stored_depth <= 0.0;
    const float3 far_point = unproject(ndc, sky ? 1.0 : 1.0 - stored_depth);
    const float3 direction = normalize(far_point - near_point);
    const float scene_distance = sky ? 1e30 : length(far_point - near_point);

    const float radius = pc.planet_radius;
    const float top = pc.atmosphere_radius / radius;
    const float3 origin = (near_point - pc.planet_centre) / radius;

    if (sky) {
        // Space behind everything: black with stars, unless the planet itself is in the way.
        const float2 planet_hit = ray_sphere(origin, direction, 1.0);
        const bool blocked = planet_hit.x > 0.0 && planet_hit.x <= planet_hit.y;
        scene.rgb = blocked ? float3(0.0) : float3(stars(direction));
    }

    const float2 shell = ray_sphere(origin, direction, top);
    const float start = max(shell.x, 0.0);
    const float end = min(shell.y, scene_distance / radius);

    float3 transmittance = float3(1.0);
    float3 inscattered = float3(0.0);

    if (shell.x <= shell.y && end > start) {
        const float step = (end - start) / float(view_steps);
        const float step_metres = step * radius;
        const float mu = dot(direction, pc.sun_direction);
        const float rayleigh_phase = 3.0 / (16.0 * pi) * (1.0 + mu * mu);
        const float g2 = mie_g * mie_g;
        const float mie_phase = 3.0 / (8.0 * pi) * ((1.0 - g2) * (1.0 + mu * mu)) /
                                ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * mie_g * mu, 1e-4), 1.5));

        float2 view_depth = float2(0.0);
        float3 rayleigh_sum = float3(0.0);
        float mie_sum = 0.0;

        for (uint i = 0; i < view_steps; ++i) {
            const float3 point = origin + direction * (start + step * (float(i) + 0.5));
            const float altitude = (length(point) - 1.0) * radius;
            const float2 density = float2(exp(-altitude / rayleigh_height), exp(-altitude / mie_height)) * step_metres;
            view_depth += density;

            const float2 light_depth = sun_optical_depth(point, pc.sun_direction, top, radius);
            const float3 tau = rayleigh_scattering * (view_depth.x + light_depth.x) +
                               mie_extinction * (view_depth.y + light_depth.y);
            const float3 attenuation = exp(-tau);
            rayleigh_sum += density.x * attenuation;
            mie_sum += density.y * attenuation.g;
        }

        transmittance = exp(-(rayleigh_scattering * view_depth.x + mie_extinction * view_depth.y));
        inscattered = (rayleigh_sum * rayleigh_scattering * rayleigh_phase + mie_sum * mie_scattering * mie_phase) *
                      pc.sun_intensity * pc.intensity;
    }

    storage_2d[pc.result][thread_id.xy] = float4(scene.rgb * transmittance + inscattered, scene.a);
}
