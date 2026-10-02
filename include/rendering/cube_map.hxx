#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>

#include <glm/glm.hpp>

// Cube map and equirect conventions shared by the CPU and by assets/shaders/environment.slang, which mirrors this
// file. Keep the two in lockstep. See docs/ibl-and-skybox.md.
//
// Faces follow Vulkan's order and per-face (s, t) table: +X, -X, +Y, -Y, +Z, -Z, with s running right and t running
// DOWN the face's rows (t = 0 is the first row in memory).
//
// Equirect: phi = atan2(d.x, d.z), u = 0.5 + phi / 2pi, v = acos(d.y) / pi. The image's centre column faces +Z and its
// top row is the zenith (+Y).

inline constexpr std::uint32_t cube_face_count = 6;

struct CubeTexel {
    std::uint32_t face = 0;

    // Position within the face in [0, 1]^2; (0, 0) is the top-left corner.
    glm::vec2 uv{0.0F};
};

// Vulkan's cube map face selection table: the major axis, then (sc, tc, ma) as written in the specification.
[[nodiscard]]
inline auto direction_to_cube_texel(glm::vec3 direction) noexcept -> CubeTexel {
    auto const abs_direction = glm::abs(direction);

    float sc = 0.0F;
    float tc = 0.0F;
    float ma = 0.0F;
    std::uint32_t face = 0;

    if (abs_direction.x >= abs_direction.y && abs_direction.x >= abs_direction.z) {
        ma = abs_direction.x;
        if (direction.x >= 0.0F) {
            face = 0;
            sc = -direction.z;
        } else {
            face = 1;
            sc = direction.z;
        }
        tc = -direction.y;
    } else if (abs_direction.y >= abs_direction.z) {
        ma = abs_direction.y;
        sc = direction.x;
        if (direction.y >= 0.0F) {
            face = 2;
            tc = direction.z;
        } else {
            face = 3;
            tc = -direction.z;
        }
    } else {
        ma = abs_direction.z;
        tc = -direction.y;
        if (direction.z >= 0.0F) {
            face = 4;
            sc = direction.x;
        } else {
            face = 5;
            sc = -direction.x;
        }
    }

    return CubeTexel{
            .face = face,
            .uv = glm::vec2{0.5F * ((sc / ma) + 1.0F), 0.5F * ((tc / ma) + 1.0F)},
    };
}

// Inverse of direction_to_cube_texel(); the result is normalised.
[[nodiscard]]
inline auto cube_texel_direction(std::uint32_t face, glm::vec2 uv) noexcept -> glm::vec3 {
    auto const s = (2.0F * uv.x) - 1.0F;
    auto const t = (2.0F * uv.y) - 1.0F;

    glm::vec3 direction{0.0F};

    switch (face) {
        case 0:
            direction = glm::vec3{1.0F, -t, -s};
            break;
        case 1:
            direction = glm::vec3{-1.0F, -t, s};
            break;
        case 2:
            direction = glm::vec3{s, 1.0F, t};
            break;
        case 3:
            direction = glm::vec3{s, -1.0F, -t};
            break;
        case 4:
            direction = glm::vec3{s, -t, 1.0F};
            break;
        default:
            direction = glm::vec3{-s, -t, -1.0F};
            break;
    }

    return glm::normalize(direction);
}

// Solid angle of texel (x, y) on a face of `size` x `size` texels, by the exact corner formula; the six faces sum to 4pi.
[[nodiscard]]
inline auto cube_texel_solid_angle(std::uint32_t x, std::uint32_t y, std::uint32_t size) noexcept -> double {
    auto const area_integral = [](double px, double py) { return std::atan2(px * py, std::sqrt((px * px) + (py * py) + 1.0)); };

    auto const inverse_size = 2.0 / static_cast<double>(size);

    auto const x0 = (static_cast<double>(x) * inverse_size) - 1.0;
    auto const y0 = (static_cast<double>(y) * inverse_size) - 1.0;
    auto const x1 = x0 + inverse_size;
    auto const y1 = y0 + inverse_size;

    return area_integral(x1, y1) - area_integral(x0, y1) - area_integral(x1, y0) + area_integral(x0, y0);
}

[[nodiscard]]
inline auto direction_to_equirect_uv(glm::vec3 direction) noexcept -> glm::vec2 {
    auto const phi = std::atan2(direction.x, direction.z);

    auto const cos_theta = std::clamp(direction.y / glm::length(direction), -1.0F, 1.0F);

    return glm::vec2{0.5F + (phi / (2.0F * std::numbers::pi_v<float>)), std::acos(cos_theta) / std::numbers::pi_v<float>};
}

[[nodiscard]]
inline auto equirect_uv_direction(glm::vec2 uv) noexcept -> glm::vec3 {
    auto const phi = (uv.x - 0.5F) * 2.0F * std::numbers::pi_v<float>;
    auto const theta = uv.y * std::numbers::pi_v<float>;

    return glm::vec3{std::sin(theta) * std::sin(phi), std::cos(theta), std::sin(theta) * std::cos(phi)};
}

// Yaw by theta about +Y, given (cos theta, sin theta). Environment lookups rotate the direction by -theta, i.e. call
// this with (cos, -sin).
[[nodiscard]]
inline auto rotate_y(glm::vec3 v, glm::vec2 cos_sin) noexcept -> glm::vec3 {
    return glm::vec3{(cos_sin.x * v.x) + (cos_sin.y * v.z), v.y, (-cos_sin.y * v.x) + (cos_sin.x * v.z)};
}
