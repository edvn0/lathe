#pragma once

#include <array>
#include <cstdint>

// Deterministic, seedable 2D simplex noise (Gustavson's formulation). Samples are roughly in [-1, 1].
class SimplexNoise2D {
public:
    explicit SimplexNoise2D(std::uint32_t seed);

    [[nodiscard]] auto sample(float x, float y) const -> float;

    // Sums `octaves` layers of sample(), scaling frequency by `lacunarity` and amplitude by `persistence` each
    // layer, normalized to [-1, 1].
    [[nodiscard]] auto fbm(float x, float y, std::uint32_t octaves, float lacunarity = 2.0F,
                           float persistence = 0.5F) const -> float;

private:
    // Doubled permutation table so lookups never wrap.
    std::array<std::uint8_t, 512> permutation_{};
};
