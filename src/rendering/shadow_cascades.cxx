#include "rendering/shadow_cascades.hxx"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace {

    // Bounding sphere of the view-frustum slice [near_distance, far_distance],
    // expressed in view space (centre offset along the forward axis, radius).
    // Using a sphere rather than the exact frustum box makes the cascade
    // rotation-invariant, which is what lets us snap the light-space origin to
    // texel increments without shimmer.
    struct FrustumSliceSphere {
        float centre_distance = 0.0F;
        float radius = 0.0F;
    };

    [[nodiscard]] auto fit_slice_sphere(float near_distance, float far_distance, float vertical_fov_radians,
                                        float aspect_ratio) noexcept -> FrustumSliceSphere {
        auto const tan_v = std::tan(vertical_fov_radians * 0.5F);
        auto const tan_h = tan_v * aspect_ratio;
        auto const k2 = (tan_h * tan_h) + (tan_v * tan_v);

        auto const a = near_distance;
        auto const b = far_distance;

        auto centre = (a + b) * (1.0F + k2) * 0.5F;
        float radius = 0.0F;

        if (centre >= b) {
            centre = b;
            radius = b * std::sqrt(k2);
        } else {
            auto const dc = a - centre;
            radius = std::sqrt((dc * dc) + (a * a * k2));
        }

        return FrustumSliceSphere{.centre_distance = centre, .radius = radius};
    }

    [[nodiscard]] auto normalize_plane(glm::vec4 const &plane) noexcept -> glm::vec4 {
        auto const length = glm::length(glm::vec3{plane});

        return length > 0.0F ? plane / length : plane;
    }

    [[nodiscard]] auto practical_split_scheme(float near_distance, float far_distance, float lambda,
                                              std::uint32_t cascade_count) noexcept
            -> std::array<float, shadow_cascade_count> {
        std::array<float, shadow_cascade_count> bounds{};

        for (std::uint32_t i = 1; i <= cascade_count; ++i) {
            auto const fraction = static_cast<float>(i) / static_cast<float>(cascade_count);

            auto const logarithmic = near_distance * std::pow(far_distance / near_distance, fraction);
            auto const uniform = near_distance + ((far_distance - near_distance) * fraction);

            bounds[i - 1] = (lambda * logarithmic) + ((1.0F - lambda) * uniform);
        }

        return bounds;
    }

} // namespace

auto fit_shadow_cascades(ShadowCascadeFitInput const &input) noexcept -> ShadowCascades {
    ShadowCascades result{};

    auto const near_distance = std::max(input.camera_near, input.settings.shadow_near);
    auto const far_distance = std::min(input.camera_far, input.settings.shadow_distance);

    auto const split_bounds =
            practical_split_scheme(near_distance, far_distance, input.settings.split_lambda, shadow_cascade_count);

    auto const inverse_view = glm::inverse(input.camera_view);
    auto const camera_position = glm::vec3(inverse_view[3]);
    auto const camera_forward = glm::normalize(glm::vec3(inverse_view[2]));

    auto const light_direction = glm::normalize(input.light_direction);
    auto const light_up =
            std::abs(light_direction.y) > 0.99F ? glm::vec3{0.0F, 0.0F, 1.0F} : glm::vec3{0.0F, 1.0F, 0.0F};
    auto const light_view = glm::lookAtLH(glm::vec3{0.0F}, -light_direction, light_up);

    float slice_near = near_distance;

    for (std::uint32_t cascade = 0; cascade < shadow_cascade_count; ++cascade) {
        auto const slice_far = split_bounds[cascade];

        auto const sphere = fit_slice_sphere(slice_near, slice_far, input.vertical_fov_radians, input.aspect_ratio);

        auto const centre_world = camera_position + (camera_forward * sphere.centre_distance);
        auto const centre_light = glm::vec3(light_view * glm::vec4(centre_world, 1.0F));

        auto const resolution = static_cast<float>(shadow_cascade_resolutions[cascade]);
        auto const texel_size = (2.0F * sphere.radius) / resolution;

        auto const snap = [texel_size](float value) noexcept { return std::floor(value / texel_size) * texel_size; };

        auto const snapped_x = snap(centre_light.x);
        auto const snapped_y = snap(centre_light.y);
        auto const snapped_z = snap(centre_light.z);

        auto const z_near = snapped_z - sphere.radius - input.settings.caster_extrusion;
        auto const z_far = snapped_z + sphere.radius;

        // Near/far swapped so the reverse-Z convention used everywhere else in
        // the renderer (nearest-to-light = 1.0) is baked into the matrix
        // itself, rather than relying on an inverted viewport as the main
        // pass does.
        auto const projection = glm::orthoLH_ZO(snapped_x - sphere.radius, snapped_x + sphere.radius,
                                                snapped_y - sphere.radius, snapped_y + sphere.radius, z_far, z_near);

        result.view_projection[cascade] = projection * light_view;
        result.split_far[cascade] = slice_far;
        result.texel_world[cascade] = texel_size;
        result.depth_scale[cascade] = 1.0F / (z_far - z_near);

        slice_near = slice_far;
    }

    return result;
}

auto extract_frustum_planes(glm::mat4 const &view_projection) noexcept -> std::array<glm::vec4, 6> {
    // GLM matrices transform column vectors (clip = M * v), so clip.x is the
    // dot product of v with row 0 of M (not column 0) -- glm::mat4's
    // operator[] indexes columns, so rows are gathered manually below.
    auto const row = [&](int index) noexcept {
        return glm::vec4{view_projection[0][index], view_projection[1][index], view_projection[2][index],
                         view_projection[3][index]};
    };

    auto const row0 = row(0);
    auto const row1 = row(1);
    auto const row2 = row(2);
    auto const row3 = row(3);

    // Inside-frustum inequalities: -w <= x <= w, -w <= y <= w, 0 <= z <= w
    // (zero-to-one depth). Each plane below is the corresponding
    // rearranged inequality, giving an inward-facing normal directly.
    return std::array<glm::vec4, 6>{
            normalize_plane(row3 + row0), // left:   x >= -w
            normalize_plane(row3 - row0), // right:  x <= w
            normalize_plane(row3 + row1), // bottom: y >= -w
            normalize_plane(row3 - row1), // top:    y <= w
            normalize_plane(row2), // near:   z >= 0
            normalize_plane(row3 - row2), // far:    z <= w
    };
}
