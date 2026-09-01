#pragma once

#include <array>
#include <cstdint>

// Deterministic, seedable 2D simplex noise (Ken Perlin's improved simplex
// algorithm, as popularized by Stefan Gustavson). Single-octave samples land
// in roughly [-1, 1].
class SimplexNoise2D {
public:
    explicit SimplexNoise2D(std::uint32_t seed);

    [[nodiscard]] auto sample(float x, float y) const -> float;

    // Fractal Brownian motion: sums `octaves` layers of sample(), each at
    // `lacunarity` times the previous frequency and `persistence` times the
    // previous amplitude, normalized so the result stays within [-1, 1].
    [[nodiscard]] auto fbm(float x, float y, std::uint32_t octaves, float lacunarity = 2.0F,
                           float persistence = 0.5F) const -> float;

private:
    // Classic doubled permutation table (0..255 shuffled, then repeated) so
    // lookups never need to wrap indices with a modulo.
    std::array<std::uint8_t, 512> permutation_{};
};
