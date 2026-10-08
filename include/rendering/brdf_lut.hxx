#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>

#include <glm/glm.hpp>

[[nodiscard]]
inline auto brdf_hammersley(std::uint32_t index, std::uint32_t count) noexcept -> glm::vec2 {
    auto bits = index;
    bits = (bits << 16U) | (bits >> 16U);
    bits = ((bits & 0x55555555U) << 1U) | ((bits & 0xAAAAAAAAU) >> 1U);
    bits = ((bits & 0x33333333U) << 2U) | ((bits & 0xCCCCCCCCU) >> 2U);
    bits = ((bits & 0x0F0F0F0FU) << 4U) | ((bits & 0xF0F0F0F0U) >> 4U);
    bits = ((bits & 0x00FF00FFU) << 8U) | ((bits & 0xFF00FF00U) >> 8U);

    return glm::vec2{static_cast<float>(index) / static_cast<float>(count), static_cast<float>(bits) * 2.3283064365386963e-10F};
}

[[nodiscard]]
inline auto importance_sample_ggx_reference(glm::vec2 xi, float alpha) noexcept -> glm::vec3 {
    auto const phi = 2.0F * std::numbers::pi_v<float> * xi.x;
    auto const cos_theta = std::sqrt((1.0F - xi.y) / (1.0F + (((alpha * alpha) - 1.0F) * xi.y)));
    auto const sin_theta = std::sqrt(std::max(0.0F, 1.0F - (cos_theta * cos_theta)));

    return glm::vec3{sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};
}

[[nodiscard]]
inline auto integrate_brdf(float ndotv, float roughness, std::uint32_t sample_count) noexcept -> glm::vec2 {
    auto const alpha = roughness * roughness;
    glm::vec3 const v{std::sqrt(1.0F - (ndotv * ndotv)), 0.0F, ndotv};

    float a = 0.0F;
    float b = 0.0F;

    for (std::uint32_t index = 0; index < sample_count; ++index) {
        auto const h = importance_sample_ggx_reference(brdf_hammersley(index, sample_count), alpha);
        auto const l = (2.0F * glm::dot(v, h) * h) - v;

        auto const ndotl = std::clamp(l.z, 0.0F, 1.0F);
        auto const ndoth = std::clamp(h.z, 0.0F, 1.0F);
        auto const vdoth = std::clamp(glm::dot(v, h), 0.0F, 1.0F);

        if (ndotl > 0.0F) {
            auto const a2 = alpha * alpha;
            auto const lambda_v = ndotl * std::sqrt((ndotv * ndotv * (1.0F - a2)) + a2);
            auto const lambda_l = ndotv * std::sqrt((ndotl * ndotl * (1.0F - a2)) + a2);
            auto const visibility = 0.5F / std::max(lambda_v + lambda_l, 1e-6F);

            auto const weighted = visibility * 4.0F * ndotl * vdoth / std::max(ndoth, 1e-6F);
            auto const fresnel = std::pow(1.0F - vdoth, 5.0F);

            a += (1.0F - fresnel) * weighted;
            b += fresnel * weighted;
        }
    }

    return glm::vec2{a, b} / static_cast<float>(sample_count);
}
