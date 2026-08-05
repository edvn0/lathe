#pragma once

#include <array>
#include <cstdint>

#include <glm/glm.hpp>

// Parallel-split shadow map cascade fitting. Pure math -- no Vulkan/GLFW
// dependency -- so it can be exercised in isolation from the renderer.

inline constexpr std::uint32_t shadow_cascade_count = 4;
inline constexpr std::uint32_t shadow_cascade_resolution = 2048;
inline constexpr std::uint32_t shadow_atlas_width = shadow_cascade_resolution * shadow_cascade_count;
inline constexpr std::uint32_t shadow_atlas_height = shadow_cascade_resolution;

struct ShadowCascadeSettings {
    float shadow_distance = 150.0F; // metres covered by the cascades, clamped to camera far
    float shadow_near = 0.5F; // clamps the camera's near clip -- 0.1 would waste cascade 0
    float split_lambda = 0.85F; // PSSM lambda: 0 = uniform splits, 1 = logarithmic
    float caster_extrusion = 100.0F; // pulls the near plane back along the light to catch casters
                                     // outside the cascade's bounding sphere
};

struct ShadowCascadeFitInput {
    glm::mat4 camera_view{1.0F};
    float camera_near = 0.1F;
    float camera_far = 10000.0F;
    float vertical_fov_radians = 1.0471976F; // 60 degrees
    float aspect_ratio = 1.7777778F;
    glm::vec3 light_direction{0.4F, 0.8F, 0.25F}; // normalized, points from surface to light
    ShadowCascadeSettings settings{};
};

struct ShadowCascades {
    std::array<glm::mat4, shadow_cascade_count> view_projection{};
    std::array<float, shadow_cascade_count> split_far{}; // view-space far distance per cascade
    std::array<float, shadow_cascade_count> texel_world{}; // world-space size of one shadow texel
    std::array<float, shadow_cascade_count> depth_scale{}; // 1 / (z_far - z_near) per cascade
};

[[nodiscard]] auto fit_shadow_cascades(ShadowCascadeFitInput const &input) noexcept -> ShadowCascades;

// Extracts the 6 world-space view-frustum planes from a combined
// view_projection matrix (Gribb-Hartmann), for the GPU frustum-culling
// compute pass. Each plane is (nx, ny, nz, d) with the normal pointing
// inward and normalized, i.e. dot(normal, point) + d >= 0 for points inside
// the frustum. Order: left, right, bottom, top, near, far. Assumes a
// zero-to-one depth range (matches every *_ZO/*_LH_ZO projection in this
// codebase); handedness of the projection does not matter for this
// extraction.
[[nodiscard]] auto extract_frustum_planes(glm::mat4 const &view_projection) noexcept -> std::array<glm::vec4, 6>;
