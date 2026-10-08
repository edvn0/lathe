#pragma once

#include <array>
#include <cstdint>

class SimplexNoise2D {
public:
    explicit SimplexNoise2D(std::uint32_t seed);

    [[nodiscard]] auto sample(float x, float y) const -> float;

    [[nodiscard]] auto fbm(float x, float y, std::uint32_t octaves, float lacunarity = 2.0F,
                           float persistence = 0.5F) const -> float;

private:
    std::array<std::uint8_t, 512> permutation_{};
};
