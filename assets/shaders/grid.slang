module grid;

import bindless;
import scene_types;

// The editor's ground grid: an infinite plane at y = 0 with the world axes drawn into it, all computed per pixel from
// the view ray (no geometry). Drawn as a fullscreen triangle after the opaque geometry and the sky; it writes its own
// depth so geometry in front of the grid hides it.

struct GridPushConstants
{
    Ptr<UBO> ubo;
};

[vk::push_constant] GridPushConstants pc;

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
    float depth : SV_Depth;
};

static const float3 axis_x_colour = float3(0.8, 0.1, 0.16);
static const float3 axis_y_colour = float3(0.25, 0.65, 0.15);
static const float3 axis_z_colour = float3(0.15, 0.35, 0.9);
static const float3 grid_colour = float3(0.6);

// One level of grid lines, `scale` world units apart: 1 on a line, 0 between lines. A level whose lines are closer
// than a few pixels fades out instead of aliasing into a haze.
float grid_level(float2 world_xz, float scale)
{
    const float2 cell = world_xz / scale;
    const float2 width = fwidth(cell);
    const float2 distance_to_line = abs(frac(cell - 0.5) - 0.5) / max(width, 1e-6);
    const float line_coverage = 1.0 - min(min(distance_to_line.x, distance_to_line.y), 1.0);

    return line_coverage * (1.0 - smoothstep(0.1, 0.5, max(width.x, width.y)));
}

// An axis line of about one pixel's width: `distance` from the axis and `pixel` the world size of a pixel there.
float axis_coverage(float axis_distance, float pixel)
{
    return 1.0 - smoothstep(0.5, 1.5, axis_distance / max(pixel, 1e-6));
}

// The framebuffer depth of a world point (the viewport maps NDC z to 1 - z).
float window_depth(Ptr<UBO> ubo, float3 world_position)
{
    const float4 clip = mul(ubo.view_projection, float4(world_position, 1.0));

    return saturate(1.0 - clip.z / clip.w);
}

[shader("fragment")]
FragmentOutput main_fs(VertexOutput input)
{
    const var ubo = pc.ubo;

    const float4 view_point = mul(ubo.inverse_projection, float4(input.ndc, 1.0, 1.0));
    const float3 view_ray = view_point.xyz / view_point.w;
    const float3 ray = normalize(mul(transpose((float3x3)ubo.view), view_ray));
    const float3 origin = ubo.camera_position;

    // World size of one pixel per unit of distance along the ray.
    const float pixel_angle = abs(ddy(input.ndc.y)) * abs(ubo.inverse_projection[1][1]);

    // The ground plane. Everything is computed unconditionally (derivatives need uniform control flow) and masked.
    const float plane_t = -origin.y / (abs(ray.y) > 1e-6 ? ray.y : 1e-6);
    const bool plane_hit = plane_t > 0.0;
    const float3 plane_point = origin + ray * plane_t;

    const float far_distance = max(abs(origin.y), 5.0) * 200.0;
    const float plane_fade = plane_hit ? 1.0 - smoothstep(0.25 * far_distance, far_distance, plane_t) : 0.0;

    const float minor = grid_level(plane_point.xz, 1.0);
    const float major = grid_level(plane_point.xz, 10.0);
    const float huge = grid_level(plane_point.xz, 100.0);

    const float3 plane_width = fwidth(plane_point);
    const float x_axis = axis_coverage(abs(plane_point.z), plane_width.z);
    const float z_axis = axis_coverage(abs(plane_point.x), plane_width.x);

    float3 colour = grid_colour;
    float alpha = saturate(max(max(minor * 0.12, major * 0.25), huge * 0.4));
    colour = lerp(colour, axis_x_colour, x_axis);
    alpha = max(alpha, x_axis * 0.9);
    colour = lerp(colour, axis_z_colour, z_axis);
    alpha = max(alpha, z_axis * 0.9);
    alpha *= plane_fade;

    // The vertical (Y) axis: the closest approach of the view ray to the line through the origin along +Y.
    const float a = ray.y;
    const float b = dot(ray, origin);
    const float c = origin.y;
    const float s = (c - a * b) / max(1.0 - a * a, 1e-6);
    const float axis_t = s * a - b;
    const float3 axis_point = float3(0.0, s, 0.0);
    const float axis_distance = length(origin + ray * axis_t - axis_point);
    const float axis_fade = axis_t > 0.0 ? 1.0 - smoothstep(0.25 * far_distance, far_distance, axis_t) : 0.0;
    const float y_axis = axis_coverage(axis_distance, axis_t * pixel_angle) * axis_fade * 0.9;

    const float combined = alpha + y_axis * (1.0 - alpha);

    if (combined < 0.004) {
        discard;
    }

    FragmentOutput output;
    output.colour = float4((colour * alpha + axis_y_colour * y_axis * (1.0 - alpha)) / combined, combined);
    output.depth = y_axis > alpha ? window_depth(ubo, axis_point) : window_depth(ubo, plane_point);

    return output;
}
