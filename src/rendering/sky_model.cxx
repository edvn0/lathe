#include "rendering/sky_model.hxx"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace {

    constexpr float half_pi = std::numbers::pi_v<float> / 2.0F;

    [[nodiscard]]
    auto perez_factor(float a, float b, float c, float d, float e, float cos_theta, float gamma) noexcept -> float {
        auto const cos_gamma = std::cos(gamma);

        return (1.0F + (a * std::exp(b / cos_theta))) * (1.0F + (c * std::exp(d * gamma)) + (e * cos_gamma * cos_gamma));
    }

    [[nodiscard]]
    auto xyy_to_linear_srgb(float x, float y, float luminance) noexcept -> glm::vec3 {
        auto const scale = luminance / std::max(y, 1e-4F);

        auto const big_x = x * scale;
        auto const big_z = (1.0F - x - y) * scale;

        return glm::max(glm::vec3{(3.2406F * big_x) - (1.5372F * luminance) - (0.4986F * big_z),
                                  (-0.9689F * big_x) + (1.8758F * luminance) + (0.0415F * big_z),
                                  (0.0557F * big_x) - (0.2040F * luminance) + (1.0570F * big_z)},
                        glm::vec3{0.0F});
    }

}

auto make_sky_state(SkyModelParams params) -> SkyState {
    auto const turbidity = std::clamp(params.turbidity, sky_min_turbidity, sky_max_turbidity);
    auto const theta_sun = half_pi - std::clamp(params.elevation_radians, 0.0F, half_pi);

    auto const t = turbidity;

    std::array<std::array<float, 5>, 3> const perez{{
            {(0.1787F * t) - 1.4630F, (-0.3554F * t) + 0.4275F, (-0.0227F * t) + 5.3251F, (0.1206F * t) - 2.5771F,
             (-0.0670F * t) + 0.3703F},
            {(-0.0193F * t) - 0.2592F, (-0.0665F * t) + 0.0008F, (-0.0004F * t) + 0.2125F, (-0.0641F * t) - 0.8989F,
             (-0.0033F * t) + 0.0452F},
            {(-0.0167F * t) - 0.2608F, (-0.0950F * t) + 0.0092F, (-0.0079F * t) + 0.2102F, (-0.0441F * t) - 1.6537F,
             (-0.0109F * t) + 0.0529F},
    }};

    auto const chi = ((4.0F / 9.0F) - (t / 120.0F)) * (std::numbers::pi_v<float> - (2.0F * theta_sun));
    auto const zenith_luminance = (((4.0453F * t) - 4.9710F) * std::tan(chi)) - (0.2155F * t) + 2.4192F;

    auto const s = theta_sun;
    auto const s2 = s * s;
    auto const s3 = s2 * s;

    auto const zenith_x = (t * t * ((0.00166F * s3) - (0.00375F * s2) + (0.00209F * s))) +
                          (t * ((-0.02903F * s3) + (0.06377F * s2) - (0.03202F * s) + 0.00394F)) +
                          ((0.11693F * s3) - (0.21196F * s2) + (0.06052F * s) + 0.25886F);

    auto const zenith_y = (t * t * ((0.00275F * s3) - (0.00610F * s2) + (0.00317F * s))) +
                          (t * ((-0.04214F * s3) + (0.08970F * s2) - (0.04153F * s) + 0.00516F)) +
                          ((0.15346F * s3) - (0.26756F * s2) + (0.06669F * s) + 0.26688F);

    SkyState state;

    std::array<float, 3> const zenith{zenith_luminance, zenith_x, zenith_y};

    for (std::size_t channel = 0; channel < 3; ++channel) {
        auto const &c = perez[channel];

        state.perez[channel] = glm::vec4{c[0], c[1], c[2], c[3]};
        state.perez[3][static_cast<int>(channel)] = c[4];

        auto const denominator = perez_factor(c[0], c[1], c[2], c[3], c[4], 1.0F, theta_sun);

        state.zenith[static_cast<int>(channel)] = zenith[channel] / denominator;
    }

    state.zenith.w = 0.0F;

    return state;
}

auto evaluate_sky_rgb(SkyState const &state, glm::vec3 direction, glm::vec3 sun_direction) -> glm::vec3 {
    auto const cos_theta = std::max(direction.y, 1e-2F);

    auto const gamma = std::acos(std::clamp(glm::dot(glm::normalize(direction), glm::normalize(sun_direction)), -1.0F, 1.0F));

    std::array<float, 3> values{};

    for (int channel = 0; channel < 3; ++channel) {
        auto const &abcd = state.perez[static_cast<std::size_t>(channel)];

        values[static_cast<std::size_t>(channel)] =
                state.zenith[channel] *
                perez_factor(abcd.x, abcd.y, abcd.z, abcd.w, state.perez[3][channel], cos_theta, gamma);
    }

    return xyy_to_linear_srgb(values[1], values[2], std::max(values[0], 0.0F)) * sky_calibration;
}

auto evaluate_environment_sky_rgb(SkyState const &state, glm::vec3 direction, glm::vec3 sun_direction,
                                  glm::vec3 ground_albedo) -> glm::vec3 {
    constexpr float blend_height = 0.0349066F;

    auto const sky = evaluate_sky_rgb(state, direction, sun_direction);

    if (direction.y >= 0.0F) {
        return sky;
    }

    auto const horizon = evaluate_sky_rgb(state, glm::vec3{1.0F, 0.0F, 0.0F}, sun_direction);

    auto const ground = ground_albedo * horizon;

    auto const blend = std::clamp(-direction.y / blend_height, 0.0F, 1.0F);
    auto const eased = blend * blend * (3.0F - (2.0F * blend));

    return glm::mix(horizon, ground, eased);
}

auto sun_transmittance(float elevation_radians, float turbidity) -> glm::vec3 {
    constexpr std::array<float, 3> wavelength_micrometres{0.650F, 0.570F, 0.475F};

    auto const clamped_turbidity = std::clamp(turbidity, sky_min_turbidity, sky_max_turbidity);
    auto const beta = (0.04608F * clamped_turbidity) - 0.04586F;

    auto const air_mass = [](float elevation) {
        auto const degrees = glm::degrees(elevation);

        return 1.0F / (std::sin(elevation) + (0.50572F * std::pow(degrees + 6.07995F, -1.6364F)));
    };

    auto const optical_depth = [&](std::size_t channel) {
        auto const lambda = wavelength_micrometres[channel];

        return (0.008735F * std::pow(lambda, -4.08F)) + (beta * std::pow(lambda, -1.3F));
    };

    auto const elevation = std::clamp(elevation_radians, glm::radians(0.5F), half_pi);

    auto const mass = air_mass(elevation);
    auto const zenith_mass = air_mass(half_pi);

    glm::vec3 result{1.0F};

    for (std::size_t channel = 0; channel < 3; ++channel) {
        auto const depth = optical_depth(channel);

        result[static_cast<int>(channel)] = std::exp(-depth * (mass - zenith_mass));
    }

    return result;
}
