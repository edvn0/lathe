#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>

// The CPU copy of the water wave field in assets/shaders/water.slang (water_wave_field and water_displacement), for
// gameplay that needs to know where the surface is (rendering/water_system.hxx). The two must stay identical: every
// constant and step here mirrors the shader, and test/water_waves_test.cxx checks the field's own consistency.
namespace water_waves {

    inline constexpr std::uint32_t wave_count = 8;

    inline constexpr std::array<glm::vec2, wave_count> directions{{
            {0.92257512F, 0.38581753F},
            {0.94740099F, 0.32004908F},
            {0.41631630F, 0.90921986F},
            {0.38410088F, 0.92329115F},
            {0.63347083F, 0.77376652F},
            {0.81119156F, 0.58478045F},
            {0.99214888F, 0.12506233F},
            {0.98732668F, -0.15870111F},
    }};

    inline constexpr std::array<float, wave_count> octave_steps{1.3700000F, 1.3834766F, 1.3878125F, 1.3864063F,
                                                                1.4070313F, 1.3751563F, 1.4726563F, 1.4515625F};

    // The mean of exp(sin(x) - 1) over a period (I0(1) / e), and how far its peak rises above it.
    inline constexpr float wave_mean = 0.465759F;
    inline constexpr float wave_peak = 1.0F - wave_mean;

    // The footprint the vertex stage displaces with: waves shorter than a few footprints are faded out. It grows with
    // the distance to the camera, never below the minimum (which is what the CPU query uses).
    inline constexpr float vertex_footprint_minimum = 0.05F;
    inline constexpr float vertex_footprint_per_metre = 0.03F;

    struct Sample {
        float height = 0.0F;
        float offset = 0.0F;
        glm::vec2 gradient{0.0F};
        glm::vec2 offset_gradient{0.0F};
        float curvature = 0.0F;
    };

    // What an anchor's coordinates add to each phase: the 8 waves, then the warp's 4 terms (water.slang's `phases`).
    using Phases = std::array<float, wave_count + 4>;

    // The phase offsets for an anchor at `anchor` (metres in the wave plane, as large as a planet), reduced in double
    // precision so the shader only ever sees small numbers.
    [[nodiscard]] inline auto anchor_phases(glm::dvec2 anchor, double frequency) -> Phases {
        constexpr auto tau = 2.0 * std::numbers::pi;
        auto const reduce = [](double phase) { return static_cast<float>(phase - tau * std::floor(phase / tau)); };

        auto phases = Phases{};
        auto octave_frequency = frequency;
        for (std::uint32_t i = 0; i < wave_count; ++i) {
            auto const k = tau * octave_frequency;
            phases[i] = reduce(k * (static_cast<double>(directions[i].x) * anchor.x +
                                    static_cast<double>(directions[i].y) * anchor.y));
            octave_frequency *= static_cast<double>(octave_steps[i]);
        }
        phases[wave_count + 0] = reduce(anchor.y * 0.21);
        phases[wave_count + 1] = reduce(anchor.x * 0.17);
        phases[wave_count + 2] = reduce(anchor.y * 0.83);
        phases[wave_count + 3] = reduce(anchor.x * 0.71);
        return phases;
    }

    [[nodiscard]] inline auto field(glm::vec2 position, float time, float frequency, float speed, float footprint,
                                    Phases const &phases = {}) -> Sample {
        auto sample = Sample{};

        position += 0.8F * glm::vec2{std::sin(position.y * 0.21F + time * 0.13F + phases[wave_count + 0]),
                                     std::cos(position.x * 0.17F - time * 0.11F + phases[wave_count + 1])} +
                    0.35F * glm::vec2{std::sin(position.y * 0.83F - time * 0.19F + phases[wave_count + 2]),
                                      std::cos(position.x * 0.71F + time * 0.23F + phases[wave_count + 3])};

        auto amplitude = 1.0F;
        auto total = 0.0F;
        auto octave_frequency = frequency;

        for (std::uint32_t i = 0; i < wave_count; ++i) {
            auto const direction = directions[i];
            auto const k = 2.0F * std::numbers::pi_v<float> * octave_frequency;
            auto const omega = std::sqrt(9.81F * k) * speed;
            auto const phase =
                    glm::dot(direction, position) * k - omega * time + static_cast<float>(i) * 1.7F + phases[i];

            auto const resolved =
                    std::clamp((1.0F / (octave_frequency * std::max(footprint, 1e-4F)) - 2.0F) * 0.25F, 0.0F, 1.0F);

            auto const sine = std::sin(phase);
            auto const cosine = std::cos(phase);
            auto const shaped = std::exp(sine - 1.0F);
            sample.height += amplitude * resolved * shaped;
            sample.offset += amplitude * resolved * (shaped - wave_mean);
            sample.gradient += amplitude * resolved * shaped * cosine * k * direction;
            sample.curvature += amplitude * resolved * shaped * (cosine * cosine - sine) * k * k;

            total += amplitude;
            amplitude *= 0.72F;
            octave_frequency *= octave_steps[i];
        }

        sample.height /= total;
        sample.gradient /= total;
        sample.offset /= total * wave_peak;
        sample.offset_gradient = sample.gradient / wave_peak;
        sample.curvature /= total * wave_peak;
        return sample;
    }

}
