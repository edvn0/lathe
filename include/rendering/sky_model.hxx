#pragma once

#include <array>

#include <glm/glm.hpp>

// CPU side of the procedural sky: the analytic Preetham et al. (1999) daylight model, "A Practical Analytic Model for
// Daylight". The shader (assets/shaders/environment.slang, preetham_sky) evaluates the same Perez distribution from the
// SkyState this builds, so keep the two in lockstep. See docs/ibl-and-skybox.md.
//
// Directions are world space with Y up. The sun is given by its direction (toward the sun), and everything is linear
// sRGB radiance in scene units after sky_calibration.

// The model's output is in kcd/m^2. This brings it to scene units so that, with the default sun (30 deg azimuth, 55 deg
// elevation, turbidity 2.5), an upward-facing white Lambert surface gets the same ambient as the flat 0.15 term the
// renderer used before IBL. Scenes switched to the sky therefore keep their exposure.
inline constexpr float sky_calibration = 0.0175F;

// Preetham is calibrated for turbidity 2-6 and stays well-behaved to 10.
inline constexpr float sky_min_turbidity = 2.0F;
inline constexpr float sky_max_turbidity = 10.0F;

struct SkyModelParams {
    // Sun elevation above the horizon. Clamped to [0, pi/2] for the model; below the horizon the caller fades the sky.
    float elevation_radians = 0.0F;
    float turbidity = 2.5F;
};

struct SkyState {
    // Rows 0-2: Perez A, B, C, D for Y (luminance), x and y (chromaticity). Row 3: E for Y, x, y (w unused).
    std::array<glm::vec4, 4> perez{};

    // xyz: the zenith (Y, x, y), each already divided by the Perez factor F(0, theta_sun) of its channel, so the shader
    // only multiplies by F(theta, gamma).
    glm::vec4 zenith{};
};

[[nodiscard]]
auto make_sky_state(SkyModelParams params) -> SkyState;

// Radiance toward `direction` from the upper hemisphere (`direction.y` is clamped to a small positive value, since the
// model is undefined below the horizon). Linear sRGB, scaled by sky_calibration.
[[nodiscard]]
auto evaluate_sky_rgb(SkyState const &state, glm::vec3 direction, glm::vec3 sun_direction) -> glm::vec3;

// The sky including the ground: below the horizon the colour is `ground_albedo` times the horizon radiance, blended
// over [-2 deg, 0] so there is no seam. This is what the procedural capture and the skybox show.
[[nodiscard]]
auto evaluate_environment_sky_rgb(SkyState const &state, glm::vec3 direction, glm::vec3 sun_direction,
                                  glm::vec3 ground_albedo) -> glm::vec3;

// Sun light colour after the atmosphere: Rayleigh + aerosol optical depth with Kasten-Young air mass, normalised to
// (1, 1, 1) at the zenith. It reddens as the sun drops. `elevation_radians` is clamped to [0.5 deg, 90 deg].
[[nodiscard]]
auto sun_transmittance(float elevation_radians, float turbidity) -> glm::vec3;
