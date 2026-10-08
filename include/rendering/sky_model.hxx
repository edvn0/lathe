#pragma once

#include <array>

#include <glm/glm.hpp>

inline constexpr float sky_calibration = 0.0175F;

inline constexpr float sky_min_turbidity = 2.0F;
inline constexpr float sky_max_turbidity = 10.0F;

struct SkyModelParams {
    float elevation_radians = 0.0F;
    float turbidity = 2.5F;
};

struct SkyState {
    std::array<glm::vec4, 4> perez{};

    glm::vec4 zenith{};
};

[[nodiscard]]
auto make_sky_state(SkyModelParams params) -> SkyState;

[[nodiscard]]
auto evaluate_sky_rgb(SkyState const &state, glm::vec3 direction, glm::vec3 sun_direction) -> glm::vec3;

[[nodiscard]]
auto evaluate_environment_sky_rgb(SkyState const &state, glm::vec3 direction, glm::vec3 sun_direction,
                                  glm::vec3 ground_albedo) -> glm::vec3;

[[nodiscard]]
auto sun_transmittance(float elevation_radians, float turbidity) -> glm::vec3;
