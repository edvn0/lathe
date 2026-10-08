#include <doctest/doctest.h>

#include <cmath>
#include <numbers>

#include <glm/glm.hpp>

#include "rendering/sky_model.hxx"

namespace {
    constexpr float pi = std::numbers::pi_v<float>;

    auto sun_direction(float azimuth_degrees, float elevation_degrees) -> glm::vec3 {
        auto const a = glm::radians(azimuth_degrees);
        auto const e = glm::radians(elevation_degrees);

        return glm::vec3{std::cos(e) * std::cos(a), std::sin(e), std::cos(e) * std::sin(a)};
    }

    auto luminance(glm::vec3 rgb) -> float { return glm::dot(rgb, glm::vec3{0.2126F, 0.7152F, 0.0722F}); }
}

TEST_CASE("the Preetham state is finite with positive radiance over the whole domain") {
    for (float turbidity = 2.0F; turbidity <= 10.0F; turbidity += 1.0F) {
        for (float elevation = 0.0F; elevation <= 90.0F; elevation += 7.5F) {
            auto const state = make_sky_state({.elevation_radians = glm::radians(elevation), .turbidity = turbidity});
            auto const sun = sun_direction(30.0F, elevation);

            for (float polar = 2.0F; polar < 90.0F; polar += 11.0F) {
                for (float azimuth = 0.0F; azimuth < 360.0F; azimuth += 45.0F) {
                    auto const p = glm::radians(polar);
                    auto const a = glm::radians(azimuth);

                    glm::vec3 const direction{std::sin(p) * std::cos(a), std::cos(p), std::sin(p) * std::sin(a)};

                    auto const radiance = evaluate_sky_rgb(state, direction, sun);

                    REQUIRE(std::isfinite(radiance.x));
                    REQUIRE(std::isfinite(radiance.y));
                    REQUIRE(std::isfinite(radiance.z));
                    CHECK(radiance.x >= 0.0F);
                    CHECK(radiance.y >= 0.0F);
                    CHECK(radiance.z >= 0.0F);
                    CHECK(luminance(radiance) > 0.0F);
                }
            }
        }
    }
}

TEST_CASE("a clear daytime sky is blue overhead and brighter toward the sun") {
    auto const sun = sun_direction(30.0F, 55.0F);
    auto const state = make_sky_state({.elevation_radians = glm::radians(55.0F), .turbidity = 2.5F});

    auto const zenith = evaluate_sky_rgb(state, glm::vec3{0, 1, 0}, sun);
    CHECK(zenith.z > zenith.x);

    auto const towards_sun = evaluate_sky_rgb(state, sun, sun);
    auto const away = evaluate_sky_rgb(state, -glm::vec3{sun.x, 0.0F, sun.z} + glm::vec3{0.0F, 0.5F, 0.0F}, sun);

    CHECK(luminance(towards_sun) > luminance(zenith));
    CHECK(luminance(towards_sun) > luminance(away));
}

TEST_CASE("the zenith is normalised: sun overhead returns the model's zenith luminance") {
    constexpr float turbidity = 3.0F;

    auto const state = make_sky_state({.elevation_radians = pi / 2.0F, .turbidity = turbidity});
    auto const radiance = evaluate_sky_rgb(state, glm::vec3{0, 1, 0}, glm::vec3{0, 1, 0});

    auto const chi = ((4.0F / 9.0F) - (turbidity / 120.0F)) * pi;
    auto const expected = ((((4.0453F * turbidity) - 4.9710F) * std::tan(chi)) - (0.2155F * turbidity) + 2.4192F) * sky_calibration;

    CHECK(luminance(radiance) == doctest::Approx(expected).epsilon(0.02));
}

TEST_CASE("the ground blends in without a seam at the horizon") {
    auto const sun = sun_direction(30.0F, 40.0F);
    auto const state = make_sky_state({.elevation_radians = glm::radians(40.0F), .turbidity = 2.5F});
    glm::vec3 const albedo{0.3F};

    auto const above = evaluate_environment_sky_rgb(state, glm::normalize(glm::vec3{1.0F, 1e-4F, 0.0F}), sun, albedo);
    auto const below = evaluate_environment_sky_rgb(state, glm::normalize(glm::vec3{1.0F, -1e-4F, 0.0F}), sun, albedo);

    CHECK(glm::length(above - below) < 0.01F * luminance(above));

    auto const ground = evaluate_environment_sky_rgb(state, glm::vec3{0, -1, 0}, sun, albedo);
    auto const horizon = evaluate_sky_rgb(state, glm::vec3{1, 0, 0}, sun);

    CHECK(ground.x == doctest::Approx(0.3F * horizon.x).epsilon(1e-3));
    CHECK(ground.z == doctest::Approx(0.3F * horizon.z).epsilon(1e-3));
}

TEST_CASE("the default sun's upward ambient matches the flat 0.15 term it replaces") {
    auto const sun = sun_direction(30.0F, 55.0F);
    auto const state = make_sky_state({.elevation_radians = glm::radians(55.0F), .turbidity = 2.5F});

    constexpr int steps = 96;

    double total = 0.0;

    for (int i = 0; i < steps; ++i) {
        auto const theta = (static_cast<float>(i) + 0.5F) / steps * (pi / 2.0F);

        for (int j = 0; j < steps; ++j) {
            auto const phi = (static_cast<float>(j) + 0.5F) / steps * (2.0F * pi);

            glm::vec3 const direction{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};

            total += luminance(evaluate_sky_rgb(state, direction, sun)) * std::cos(theta) * std::sin(theta);
        }
    }

    auto const ambient = total * (pi / 2.0 / steps) * (2.0 * pi / steps) / pi;

    CHECK(ambient == doctest::Approx(0.15).epsilon(0.1));
}

TEST_CASE("sun transmittance is white at the zenith and reddens toward the horizon") {
    auto const zenith = sun_transmittance(pi / 2.0F, 2.5F);

    CHECK(zenith.x == doctest::Approx(1.0F).epsilon(1e-4));
    CHECK(zenith.z == doctest::Approx(1.0F).epsilon(1e-4));

    float previous_ratio = 0.0F;

    for (float elevation = 80.0F; elevation >= 2.0F; elevation -= 6.0F) {
        auto const t = sun_transmittance(glm::radians(elevation), 2.5F);

        CHECK(t.x <= 1.0F);
        CHECK(t.z <= t.x);
        CHECK(t.x > 0.0F);

        auto const ratio = t.x / t.z;

        CHECK(ratio >= previous_ratio);

        previous_ratio = ratio;
    }

    CHECK(previous_ratio > 1.5F);
}
