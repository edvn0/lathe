#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

#include <glm/glm.hpp>

#include "rendering/cube_map.hxx"
#include "rendering/spherical_harmonics.hxx"

namespace {
    template<typename Radiance>
    auto project_cube(std::uint32_t size, Radiance &&radiance) -> Sh9 {
        Sh9 sh;

        for (std::uint32_t face = 0; face < cube_face_count; ++face) {
            for (std::uint32_t y = 0; y < size; ++y) {
                for (std::uint32_t x = 0; x < size; ++x) {
                    auto const uv = glm::vec2{(static_cast<float>(x) + 0.5F) / static_cast<float>(size),
                                              (static_cast<float>(y) + 0.5F) / static_cast<float>(size)};

                    auto const direction = cube_texel_direction(face, uv);

                    sh9_accumulate(sh, direction, radiance(direction), cube_texel_solid_angle(x, y, size));
                }
            }
        }

        return sh;
    }
} // namespace

TEST_CASE("SH basis is orthonormal over the sphere") {
    constexpr std::uint32_t size = 128;

    std::array<std::array<double, sh9_coefficient_count>, sh9_coefficient_count> gram{};

    for (std::uint32_t face = 0; face < cube_face_count; ++face) {
        for (std::uint32_t y = 0; y < size; ++y) {
            for (std::uint32_t x = 0; x < size; ++x) {
                auto const uv = glm::vec2{(static_cast<float>(x) + 0.5F) / size, (static_cast<float>(y) + 0.5F) / size};

                auto const basis = sh9_basis(cube_texel_direction(face, uv));
                auto const weight = cube_texel_solid_angle(x, y, size);

                for (std::uint32_t row = 0; row < sh9_coefficient_count; ++row) {
                    for (std::uint32_t column = 0; column < sh9_coefficient_count; ++column) {
                        gram[row][column] += static_cast<double>(basis[row]) * basis[column] * weight;
                    }
                }
            }
        }
    }

    for (std::uint32_t row = 0; row < sh9_coefficient_count; ++row) {
        for (std::uint32_t column = 0; column < sh9_coefficient_count; ++column) {
            CHECK(gram[row][column] == doctest::Approx(row == column ? 1.0 : 0.0).epsilon(1e-3).scale(1.0));
        }
    }
}

TEST_CASE("constant radiance gives irradiance over pi equal to the radiance everywhere") {
    auto const sh = sh9_cosine_convolve_over_pi(project_cube(64, [](glm::vec3) { return glm::vec3{1.0F, 0.5F, 2.0F}; }));

    std::mt19937 engine{7};
    std::normal_distribution<float> normal{0.0F, 1.0F};

    for (int index = 0; index < 256; ++index) {
        auto const direction = glm::normalize(glm::vec3{normal(engine), normal(engine), normal(engine)});

        auto const value = sh9_eval(sh, direction);

        CHECK(value.x == doctest::Approx(1.0F).epsilon(1e-3));
        CHECK(value.y == doctest::Approx(0.5F).epsilon(1e-3));
        CHECK(value.z == doctest::Approx(2.0F).epsilon(1e-3));
    }
}

TEST_CASE("a directional radiance lobe follows the clamped cosine response within L2 ringing") {
    // A small bright patch around +Y (5 degree half angle) behaves as a delta for an L2 projection.
    constexpr float cap = 0.996194F; // cos(5 degrees)

    auto const sh = project_cube(256, [](glm::vec3 d) { return d.y > cap ? glm::vec3{1.0F} : glm::vec3{0.0F}; });

    auto const irradiance = sh9_cosine_convolve_over_pi(sh);

    // Straight toward the patch the exact response is solid_angle * cos(0) / pi; the L2 truncation overshoots slightly.
    auto const patch_solid_angle = 2.0 * std::numbers::pi * (1.0 - cap);
    auto const exact_up = static_cast<float>(patch_solid_angle / std::numbers::pi);

    auto const up = sh9_eval(irradiance, glm::vec3{0, 1, 0}).x;
    CHECK(up > 0.9F * exact_up);
    CHECK(up < 1.6F * exact_up);

    // Opposite side: the exact response is zero, and L2 ringing can leave only a small residual.
    auto const down = sh9_eval(irradiance, glm::vec3{0, -1, 0}).x;
    CHECK(std::abs(down) < 0.25F * exact_up);
}

TEST_CASE("yaw rotation of the lookup equals projecting a rotated cube") {
    auto const radiance = [](glm::vec3 d) {
        return glm::vec3{std::max(d.x, 0.0F) + 0.2F, std::max(d.z, 0.0F), std::max(-d.y, 0.0F) + 0.1F};
    };

    constexpr float angle = 0.7F;
    glm::vec2 const cos_sin{std::cos(angle), std::sin(angle)};

    auto const original = sh9_cosine_convolve_over_pi(project_cube(64, radiance));

    // The environment rotated by +angle about Y: L'(d) = L(R(-angle) d).
    auto const rotated = sh9_cosine_convolve_over_pi(
            project_cube(64, [&](glm::vec3 d) { return radiance(rotate_y(d, glm::vec2{cos_sin.x, -cos_sin.y})); }));

    std::mt19937 engine{99};
    std::normal_distribution<float> normal{0.0F, 1.0F};

    for (int index = 0; index < 64; ++index) {
        auto const n = glm::normalize(glm::vec3{normal(engine), normal(engine), normal(engine)});

        // Lookups rotate the normal by -angle, as the shaders do for a scene rotation of +angle.
        auto const via_lookup = sh9_eval(original, rotate_y(n, glm::vec2{cos_sin.x, -cos_sin.y}));
        auto const via_cube = sh9_eval(rotated, n);

        CHECK(glm::length(via_lookup - via_cube) < 5e-3F);
    }
}
