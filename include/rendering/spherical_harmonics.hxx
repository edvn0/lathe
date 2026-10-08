#pragma once

#include <array>
#include <cstdint>

#include <glm/glm.hpp>

#include "rendering/cube_map.hxx"

inline constexpr std::uint32_t sh9_coefficient_count = 9;

struct Sh9 {
    std::array<glm::vec3, sh9_coefficient_count> coefficients{};
};

[[nodiscard]]
inline auto sh9_basis(glm::vec3 d) noexcept -> std::array<float, sh9_coefficient_count> {
    return {
            0.282095F,
            0.488603F * d.y,
            0.488603F * d.z,
            0.488603F * d.x,
            1.092548F * d.x * d.y,
            1.092548F * d.y * d.z,
            0.315392F * ((3.0F * d.z * d.z) - 1.0F),
            1.092548F * d.x * d.z,
            0.546274F * ((d.x * d.x) - (d.y * d.y)),
    };
}

inline auto sh9_accumulate(Sh9 &sh, glm::vec3 direction, glm::vec3 radiance, double solid_angle) noexcept -> void {
    auto const basis = sh9_basis(direction);

    for (std::uint32_t index = 0; index < sh9_coefficient_count; ++index) {
        sh.coefficients[index] += radiance * static_cast<float>(basis[index] * solid_angle);
    }
}

[[nodiscard]]
inline auto sh9_cosine_convolve_over_pi(Sh9 const &radiance) noexcept -> Sh9 {
    constexpr std::array<float, sh9_coefficient_count> band_factor{
            1.0F, 2.0F / 3.0F, 2.0F / 3.0F, 2.0F / 3.0F, 0.25F, 0.25F, 0.25F, 0.25F, 0.25F,
    };

    Sh9 result;

    for (std::uint32_t index = 0; index < sh9_coefficient_count; ++index) {
        result.coefficients[index] = radiance.coefficients[index] * band_factor[index];
    }

    return result;
}

[[nodiscard]]
inline auto sh9_eval(Sh9 const &sh, glm::vec3 normal) noexcept -> glm::vec3 {
    auto const basis = sh9_basis(normal);

    glm::vec3 result{0.0F};

    for (std::uint32_t index = 0; index < sh9_coefficient_count; ++index) {
        result += sh.coefficients[index] * basis[index];
    }

    return result;
}
