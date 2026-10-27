module water;

// Water surfaces (Material.flags & material_flag_water), shaded in the blended forward pass. The scene behind the
// surface comes from the copy of the scene colour and depth taken after the opaque and masked draws
// (PC.scene_colour_index / scene_depth_index). See docs/water.md for the model and its limits.

import bindless;
import scene_types;
import environment;
import pbr;

static const float water_pi = 3.14159265358979323846;
static const float water_f0 = 0.02;
// The wave field below is mirrored on the CPU by include/rendering/water_waves.hxx (the gameplay height query):
// every constant from here to water_wave_field must match it.
static const uint water_wave_count = 8;
static const float2 water_wave_directions[water_wave_count] = {
    float2(0.92257512, 0.38581753),
    float2(0.94740099, 0.32004908),
    float2(0.41631630, 0.90921986),
    float2(0.38410088, 0.92329115),
    float2(0.63347083, 0.77376652),
    float2(0.81119156, 0.58478045),
    float2(0.99214888, 0.12506233),
    float2(0.98732668, -0.15870111),
};
static const float water_wave_octave_steps[water_wave_count] = {
    1.3700000, 1.3834766, 1.3878125, 1.3864063, 1.4070313, 1.3751563, 1.4726563, 1.4515625,
};
static const float water_vertex_footprint_minimum = 0.05;
static const float water_vertex_footprint_per_metre = 0.03;
// Scales Material.water_waves.z so a strength of 1 gives gentle slopes on the default pattern.
static const float water_slope_scale = 0.2;

// The foam is lit by the ordinary forward shading as a rough white surface (see main_fs).
public static const float3 water_foam_albedo = float3(0.9, 0.92, 0.94);
public static const float water_foam_roughness = 0.85;

public struct WaterShade
{
    // The water surface (refraction, reflection, sun highlight), fogged.
    public float3 colour;
    // Coverage: fades in from the shoreline.
    public float alpha;
    // How much of the surface is foam, to mix over `colour`.
    public float foam;
};

// The mean of exp(sin(x) - 1) over a period (I0(1) / e), and how far its peak rises above that mean.
static const float water_wave_mean = 0.465759;
static const float water_wave_peak = 1.0 - water_wave_mean;

public struct WaveSample
{
    // 0..1, about 0.5 on average; crests reach towards 1.
    public float height;
    // The height about its mean, scaled so a lone crest reaches 1: what the surface is displaced by, in units of the
    // material's wave height. Waves faded by the footprint add nothing here instead of their mean, so the surface does
    // not sink where they fade.
    public float offset;
    // d(height)/d(xz), per metre.
    public float2 gradient;
    // d(offset)/d(xz), per metre (the slope of the displaced surface, per metre of wave height).
    public float2 offset_gradient;
    // The Laplacian of `offset`: strongly negative on sharp crests, where the surface is squeezed together.
    public float curvature;
};

// A sum of directional waves that scroll at their deep-water speed, each one a sharpened sine (exp(sin - 1)) so crests
// are narrow and troughs wide. Directions scatter around one wind heading and frequencies rise by about 1.4 per octave.
// `phases` adds to each wave's phase (x..w of [0] and [1]) and to the warp's four terms ([2]): what a far-away anchor's
// own coordinates contribute (see WaterFrame); zero for flat water, whose coordinates are the world's XZ.
public WaveSample water_wave_field(float2 position, float time, float frequency, float speed, float footprint,
                                   float4 phases[3])
{
    WaveSample wave;
    wave.height = 0.0;
    wave.offset = 0.0;
    wave.gradient = float2(0.0);
    wave.curvature = 0.0;

    // A slow, gentle domain warp bends the otherwise ruler-straight crests, so crossing wave trains do not form a grid.
    position += 0.8 * float2(sin(position.y * 0.21 + time * 0.13 + phases[2].x),
                             cos(position.x * 0.17 - time * 0.11 + phases[2].y)) +
                0.35 * float2(sin(position.y * 0.83 - time * 0.19 + phases[2].z),
                              cos(position.x * 0.71 + time * 0.23 + phases[2].w));

    float amplitude = 1.0;
    float total = 0.0;
    float octave_frequency = frequency;

    [unroll]
    for (uint i = 0; i < water_wave_count; ++i) {
        // Waves share a wind direction, scattered within about +-50 degrees of it: trains travelling in every direction
        // interfere into concentric rings and grids. The directions are a table (not a sin() hash, whose last digits
        // differ between GPUs and the CPU) so water_waves.hxx can mirror the field exactly.
        const float2 direction = water_wave_directions[i];
        const float k = 2.0 * water_pi * octave_frequency;
        const float omega = sqrt(9.81 * k) * speed;
        const float phase = dot(direction, position) * k - omega * time + float(i) * 1.7 + phases[i / 4][i % 4];

        // Waves shorter than a few pixels alias into blocks; fade them out with the pixel footprint on the surface.
        const float resolved = saturate((1.0 / (octave_frequency * max(footprint, 1e-4)) - 2.0) * 0.25);

        const float sine = sin(phase);
        const float cosine = cos(phase);
        const float shaped = exp(sine - 1.0);
        wave.height += amplitude * resolved * shaped;
        wave.offset += amplitude * resolved * (shaped - water_wave_mean);
        wave.gradient += amplitude * resolved * shaped * cosine * k * direction;
        wave.curvature += amplitude * resolved * shaped * (cosine * cosine - sine) * k * k;

        total += amplitude;
        amplitude *= 0.72;
        octave_frequency *= water_wave_octave_steps[i];
    }

    wave.height /= total;
    wave.gradient /= total;
    wave.offset /= total * water_wave_peak;
    wave.offset_gradient = wave.gradient / water_wave_peak;
    wave.curvature /= total * water_wave_peak;
    return wave;
}

// Where a water surface's waves are laid out: its up direction, two tangent axes, and the 2D coordinate the wave field
// is evaluated at. Flat water uses the world's XZ. A sphere (a planet's ocean) uses the tangent plane at an anchor near
// the camera; the anchor's own (huge) coordinates enter only as phase offsets reduced on the CPU, so the floats here
// stay small. Moving the anchor shifts the pattern slightly (docs/globe.md).
public struct WaterFrame
{
    public float3 up;
    public float3 east;
    public float3 north;
    public float2 position;
};

public WaterFrame water_frame(Material material, float3 world_position)
{
    WaterFrame frame;
    if (material.water_sphere_centre.w > 0.5) {
        frame.up = normalize(world_position - material.water_sphere_centre.xyz);
        frame.east = normalize(material.water_east.xyz - frame.up * dot(material.water_east.xyz, frame.up));
        frame.north = cross(frame.east, frame.up);
        if (dot(frame.north, material.water_north.xyz) < 0.0) {
            frame.north = -frame.north;
        }
        const float3 offset = world_position - material.water_anchor.xyz;
        frame.position = float2(dot(offset, material.water_east.xyz), dot(offset, material.water_north.xyz));
    } else {
        frame.up = float3(0.0, 1.0, 0.0);
        frame.east = float3(1.0, 0.0, 0.0);
        frame.north = float3(0.0, 0.0, 1.0);
        frame.position = world_position.xz;
    }
    return frame;
}

// How far (m) the vertex stage lifts a water surface along `frame.up`: the same wave field the fragment shader shades,
// with the waves the mesh is too coarse for faded out (the mesh spacing grows with distance, so the footprint does).
public float water_displacement(Material material, WaterFrame frame, float time, float camera_distance)
{
    const float footprint = max(water_vertex_footprint_minimum, camera_distance * water_vertex_footprint_per_metre);
    const WaveSample wave = water_wave_field(frame.position, time, material.water_waves.x, material.water_waves.y,
                                             footprint, material.water_phases);
    return wave.offset * material.water_shape.x;
}

float water_hash(float2 p)
{
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

float water_value_noise(float2 p)
{
    const float2 cell = floor(p);
    const float2 f = frac(p);
    const float2 u = f * f * (3.0 - 2.0 * f);

    return lerp(lerp(water_hash(cell), water_hash(cell + float2(1.0, 0.0)), u.x),
                lerp(water_hash(cell + float2(0.0, 1.0)), water_hash(cell + float2(1.0, 1.0)), u.x), u.y);
}

// Breaking-up pattern for foam: two scrolling octaves of value noise.
float water_foam_noise(float2 p, float time)
{
    return 0.65 * water_value_noise(p * 2.3 + float2(time * 0.21, time * 0.13)) +
           0.35 * water_value_noise(p * 5.1 - float2(time * 0.17, -time * 0.24));
}

// A cheap caustic network: a domain-warped grid of sin(x) * sin(y), bright where it crosses zero.
float water_caustic_pattern(float2 p, float time)
{
    float2 q = p * 1.6;

    [unroll]
    for (uint i = 0; i < 3; ++i) {
        const float fi = float(i);
        q = float2(q.x + 0.55 * sin(q.y * 1.3 + time * 0.9 + fi), q.y + 0.55 * cos(q.x * 1.1 - time * 0.7 + fi * 1.7));
    }

    const float s = sin(q.x * 2.1) * sin(q.y * 2.1);
    return pow(1.0 - saturate(abs(s)), 6.0);
}

// View-space depth (distance along the view axis) of a depth-buffer value; the depth is reversed (1 near, 0 far) and
// 0 is the sky.
float water_view_depth(Ptr<UBO> ubo, float stored_depth)
{
    if (stored_depth <= 0.0) {
        return 1e7;
    }

    const float4 view = mul(ubo.inverse_projection, float4(0.0, 0.0, 1.0 - stored_depth, 1.0));
    return view.z / view.w;
}

float water_scene_depth(PC pc, float2 uv)
{
    const int2 size = int2(pc.screen_size);
    const int2 pixel = clamp(int2(uv * pc.screen_size), int2(0), size - 1);
    return water_view_depth(pc.ubo, sampled_2d_depth[NonUniformResourceIndex(pc.scene_depth_index)].Load(int3(pixel, 0)));
}

float3 water_scene_colour(PC pc, float2 uv)
{
    return sampled_2d[NonUniformResourceIndex(pc.scene_colour_index)]
            .SampleLevel(samplers[NonUniformResourceIndex(pc.ao_sampler_index)], uv, 0.0)
            .rgb;
}

// The environment in a direction, as the opaque shading's specular IBL sees it.
float3 water_environment(Ptr<UBO> ubo, float3 direction, float roughness)
{
    if ((ubo.environment_flags & env_ibl_valid) == 0) {
        return ubo.fog_colour * ubo.ambient_intensity;
    }

    const float2 unrotate = float2(ubo.environment_rotation.x, -ubo.environment_rotation.y);
    return sampled_cube[ubo.prefilter_cube_texture]
                   .SampleLevel(samplers[ubo.environment_sampler], rotate_y(direction, unrotate),
                                roughness * ubo.environment_rotation.z)
                   .rgb *
           ubo.environment_intensity.y * ubo.environment_exposure;
}

float3 water_ambient(Ptr<UBO> ubo)
{
    if ((ubo.environment_flags & env_ibl_valid) == 0) {
        return float3(ubo.ambient_intensity);
    }

    const float2 unrotate = float2(ubo.environment_rotation.x, -ubo.environment_rotation.y);
    return sh9_eval(ubo.environment_sh, rotate_y(float3(0.0, 1.0, 0.0), unrotate)) * ubo.environment_intensity.x *
           ubo.environment_exposure;
}

// Screen-space reflection: marches the reflected ray in world space, projecting each step against the depth copy.
// Returns the reflected colour and how much to trust it (0 = use the environment).
float4 water_screen_space_reflection(PC pc, float3 origin, float3 direction, float jitter)
{
    const var ubo = pc.ubo;
    const uint step_count = 32;
    const float max_distance = 80.0;

    // Starting each pixel a random fraction of a step along breaks the banding of equal-length steps into fine noise.
    float travelled = 0.15 * jitter;
    float step = 0.15;
    float previous = 0.0;
    bool hit = false;

    for (uint i = 0; i < step_count && !hit; ++i) {
        previous = travelled;
        travelled += step;
        step *= 1.15;

        const float4 clip = mul(ubo.view_projection, float4(origin + direction * travelled, 1.0));
        const float safe_w = max(clip.w, 1e-3);
        const float2 uv = float2(0.5 + 0.5 * clip.x / safe_w, 0.5 - 0.5 * clip.y / safe_w);

        if (travelled > max_distance || clip.w <= 1e-3 || any(uv < 0.0) || any(uv > 1.0)) {
            i = step_count;
        } else {
            // Behind the depth buffer, but not by more than a thickness that grows with the step, or it passed behind
            // an object instead of hitting it.
            const float behind = clip.w - water_scene_depth(pc, uv);
            hit = behind > 0.0 && behind < step * 2.0 + 0.3;
        }
    }

    if (!hit) {
        return float4(0.0);
    }

    // Bisect between the last step in front of the scene and the first one behind it.
    float near_t = previous;
    float far_t = travelled;
    for (uint refine = 0; refine < 5; ++refine) {
        const float mid_t = 0.5 * (near_t + far_t);
        const float4 mid_clip = mul(ubo.view_projection, float4(origin + direction * mid_t, 1.0));
        const float2 mid_uv = float2(0.5 + 0.5 * mid_clip.x / mid_clip.w, 0.5 - 0.5 * mid_clip.y / mid_clip.w);
        const bool past = mid_clip.w > water_scene_depth(pc, mid_uv);
        far_t = past ? mid_t : far_t;
        near_t = past ? near_t : mid_t;
    }

    const float4 hit_clip = mul(ubo.view_projection, float4(origin + direction * far_t, 1.0));
    const float2 hit_uv = saturate(float2(0.5 + 0.5 * hit_clip.x / hit_clip.w, 0.5 - 0.5 * hit_clip.y / hit_clip.w));

    const float2 edge = min(hit_uv, 1.0 - hit_uv);
    const float edge_fade = saturate(min(edge.x, edge.y) * 10.0);
    const float distance_fade = 1.0 - saturate((far_t - max_distance * 0.6) / (max_distance * 0.4));

    return float4(water_scene_colour(pc, hit_uv), edge_fade * distance_fade);
}

public WaterShade shade_water(PC pc, Material material, float3 world_position, float3 geometric_normal, float2 pixel,
                          float view_z, float shadow)
{
    const var ubo = pc.ubo;
    const float time = ubo.time;

    const float3 absorption = material.water_absorption.xyz;
    const float shore_fade = max(material.water_absorption.w, 1e-3);
    const float wave_frequency = material.water_waves.x;
    const float wave_speed = material.water_waves.y;
    const float normal_strength = material.water_waves.z * water_slope_scale;
    const float wave_height = material.water_shape.x;
    const float refraction_strength = material.water_waves.w;
    const float roughness = clamp(material.roughness_factor, 0.02, 1.0);

    // Normal: the wave slopes tilt the surface normal; only upward-facing surfaces get waves.
    const WaterFrame frame = water_frame(material, world_position);
    const WaveSample wave = water_wave_field(frame.position, time, wave_frequency, wave_speed,
                                             length(fwidth(frame.position)), material.water_phases);
    // Only surfaces facing up (along the frame) get waves.
    const float up = saturate(dot(geometric_normal, frame.up));
    // The geometry is displaced by wave_height * offset (water_displacement), so its slope is part of the normal; the
    // normal strength adds detail on top, including the short waves the mesh cannot follow.
    const float2 slope = wave.gradient * normal_strength + wave.offset_gradient * wave_height;
    const float3 normal = normalize(geometric_normal - (frame.east * slope.x + frame.north * slope.y) * up);

    const float3 to_camera = ubo.camera_position - world_position;
    const float camera_distance = length(to_camera);
    const float3 view_direction = to_camera / max(camera_distance, 1e-4);
    const float3 ray = -view_direction;
    const float ndotv = saturate(dot(normal, view_direction));

    const float2 screen_uv = pixel / pc.screen_size;
    const float surface_depth = max(view_z, 1e-3);
    // Metres along the view ray per unit of view-space depth.
    const float ray_per_depth = camera_distance / surface_depth;

    // Refraction: offset by the wave slope, less with distance; an offset that lands on something in front of the
    // water (a pillar standing in it) is rejected for the unshifted sample.
    const float3 tilt = normal - geometric_normal;
    const float2 distortion = float2(dot(tilt, frame.east), dot(tilt, frame.north)) * refraction_strength *
                              clamp(6.0 / surface_depth, 0.15, 1.0);
    float2 refracted_uv = clamp(screen_uv + distortion, float2(0.0), float2(1.0));
    float scene_depth = water_scene_depth(pc, refracted_uv);
    if (scene_depth < surface_depth) {
        refracted_uv = screen_uv;
        scene_depth = water_scene_depth(pc, refracted_uv);
    }

    const float sky_behind = scene_depth >= 1e6 ? 1.0 : 0.0;
    const float depth_behind = max(scene_depth - surface_depth, 0.0);
    const float3 floor_position = world_position + ray * (depth_behind * ray_per_depth);
    // How deep the water is above what is seen through it, and how far the light travels through it.
    const float water_column = sky_behind > 0.0 ? 1e4
                                                : max(dot(world_position - floor_position, frame.up),
                                                      depth_behind * ray_per_depth * abs(dot(ray, frame.up)));
    const float water_path = sky_behind > 0.0 ? 1e4 : depth_behind * ray_per_depth;

    const float3 ambient = water_ambient(ubo);
    const float3 light_direction = normalize(ubo.light_direction);
    const float3 sun = ubo.light_colour * ubo.light_intensity;
    const float sun_height = saturate(dot(light_direction, frame.up));

    float3 behind = water_scene_colour(pc, refracted_uv);

    // Caustics on what lies under the water, as seen through it: strongest a little below the surface, gone with depth.
    if (material.water_caustics > 0.0 && sky_behind == 0.0) {
        const float2 sun_along = float2(dot(light_direction, frame.east), dot(light_direction, frame.north));
        const float2 floor_coordinate = frame.position + float2(dot(floor_position - world_position, frame.east),
                                                                dot(floor_position - world_position, frame.north));
        const float caustic = water_caustic_pattern(floor_coordinate + sun_along * water_column, time * wave_speed);
        const float depth_fade = saturate(water_column * 4.0) * exp(-water_column * 0.35);
        behind *= 1.0 + caustic * material.water_caustics * depth_fade * shadow * sun_height * 2.5;
    }

    // Beer-Lambert absorption along the path, filled in by light scattered back out of the water body.
    const float3 transmittance = exp(-absorption * water_path);
    const float3 scatter = float3(material.colour[0], material.colour[1], material.colour[2]) *
                           (ambient + sun * sun_height * lerp(0.4, 1.0, shadow) * (1.0 / water_pi));
    const float3 refraction = behind * transmittance + scatter * (1.0 - transmittance);

    // Reflection: screen space where the march finds the scene, the environment where it does not.
    // Reflected rays are kept above the horizon (the frame's tangent plane).
    float3 reflected = reflect(ray, normal);
    reflected -= frame.up * min(2.0 * dot(reflected, frame.up), 0.0);
    const float3 environment = water_environment(ubo, reflected, roughness);
    // The march follows a flatter normal than the shading does: the full wave normal sends neighbouring pixels' rays to
    // different parts of the scene and the reflection breaks into dashes.
    float3 march_direction = reflect(ray, normalize(lerp(geometric_normal, normal, 0.3)));
    march_direction -= frame.up * min(2.0 * dot(march_direction, frame.up), 0.0);
    const float jitter = frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
    const float4 screen_reflection = water_screen_space_reflection(pc, world_position, march_direction, jitter);
    const float3 reflection = lerp(environment, screen_reflection.rgb, screen_reflection.a);

    const float fresnel = water_f0 + (1.0 - water_f0) * pow(1.0 - ndotv, 5.0);
    float3 colour = lerp(refraction, reflection, fresnel);

    // The sun's highlight on the waves.
    colour += shade_punctual(normal, view_direction, light_direction, sun * shadow, float3(0.0), 0.0, roughness);

    // Foam: a narrow, broken band along the shoreline and wherever something breaks the surface, and on the highest
    // crests once the waves are rough enough (none on a calm pool).
    const float noise = water_foam_noise(frame.position, time * wave_speed);
    const float shore = (1.0 - saturate(water_column / 0.2)) * (1.0 - sky_behind);
    // Whitecaps where the displaced surface is high and squeezed together (its curvature stands in for the Jacobian of
    // a choppy displacement), scaled by the material's whitecap amount.
    const float compression = saturate(-wave.curvature * wave_height * 3.0);
    const float crest_foam = smoothstep(0.15, 0.55, wave.offset) * smoothstep(0.1, 0.45, compression) *
                             material.water_shape.y;
    const float foam = saturate((shore * shore * (0.3 + 1.2 * noise) + crest_foam * (0.4 + noise)) *
                                material.water_foam) *
                       smoothstep(0.3, 0.6, noise + shore * 0.4 + crest_foam * 0.3);

    // The same distance fog as the opaque surfaces.
    const float fog_transmittance = exp(-camera_distance * ubo.fog_extinction);
    float3 fog_colour = ubo.fog_colour;
    if ((ubo.environment_flags & env_fog_from_environment) != 0) {
        fog_colour *= environment_fog_colour(ubo, ray);
    }
    colour = colour * fog_transmittance + fog_colour * ubo.fog_inscattering * (1.0 - fog_transmittance);

    // Fade in from the shoreline over shore_fade metres of water.
    WaterShade shade;
    shade.colour = colour;
    shade.alpha = sky_behind > 0.0 ? 1.0 : saturate(water_column / shore_fade);
    shade.foam = foam;
    return shade;
}
