#include "terrain/noise.hxx"

#include <algorithm>
#include <numeric>
#include <random>

namespace {

    constexpr std::array<std::array<float, 2>, 8> gradients{{
            {1.0F, 0.0F},
            {-1.0F, 0.0F},
            {0.0F, 1.0F},
            {0.0F, -1.0F},
            {0.70710678F, 0.70710678F},
            {-0.70710678F, 0.70710678F},
            {0.70710678F, -0.70710678F},
            {-0.70710678F, -0.70710678F},
    }};

    constexpr float f2 = 0.3660254038F; // 0.5 * (sqrt(3) - 1)
    constexpr float g2 = 0.2113248654F; // (3 - sqrt(3)) / 6

    auto fastfloor(float x) -> int {
        auto const truncated = static_cast<int>(x);
        return x < static_cast<float>(truncated) ? truncated - 1 : truncated;
    }

    auto corner_contribution(float x, float y, std::unsigned_integral auto gradient_index) -> float {
        auto t = 0.5F - x * x - y * y;
        if (t < 0.0F) {
            return 0.0F;
        }
        t *= t;
        auto const &gradient = gradients[gradient_index];
        return t * t * (gradient[0] * x + gradient[1] * y);
    }

} // namespace

SimplexNoise2D::SimplexNoise2D(std::uint32_t seed) {
    std::array<std::uint8_t, 256> base{};
    std::iota(base.begin(), base.end(), std::uint8_t{0});

    std::mt19937 engine{seed};
    std::shuffle(base.begin(), base.end(), engine);

    for (std::size_t i = 0; i < base.size(); ++i) {
        permutation_[i] = base[i];
        permutation_[i + base.size()] = base[i];
    }
}

auto SimplexNoise2D::sample(float x, float y) const -> float {
    auto const skew = (x + y) * f2;
    auto const i = fastfloor(x + skew);
    auto const j = fastfloor(y + skew);

    auto const unskew = static_cast<float>(i + j) * g2;
    auto const x0 = x - (static_cast<float>(i) - unskew);
    auto const y0 = y - (static_cast<float>(j) - unskew);

    // Which simplex (triangle half of the unit square) (x0, y0) falls in
    // determines the middle corner's offset.
    int const i1 = x0 > y0 ? 1 : 0;
    int const j1 = x0 > y0 ? 0 : 1;

    auto const x1 = x0 - static_cast<float>(i1) + g2;
    auto const y1 = y0 - static_cast<float>(j1) + g2;
    auto const x2 = x0 - 1.0F + 2.0F * g2;
    auto const y2 = y0 - 1.0F + 2.0F * g2;

    // Kept as plain int (not uint8_t) so `+ 1` below can't wrap before the
    // permutation lookup; the table is sized to 512 exactly to absorb these
    // offsets without ever needing a modulo.
    auto const ii = i & 255;
    auto const jj = j & 255;

    auto const gi0 = permutation_[ii + permutation_[jj]] & 7U;
    auto const gi1 = permutation_[ii + i1 + permutation_[jj + j1]] & 7U;
    auto const gi2 = permutation_[ii + 1 + permutation_[jj + 1]] & 7U;

    auto const n0 = corner_contribution(x0, y0, gi0);
    auto const n1 = corner_contribution(x1, y1, gi1);
    auto const n2 = corner_contribution(x2, y2, gi2);

    // Empirical scaling factor (standard for this formulation) that brings
    // the sum into roughly [-1, 1].
    return 70.0F * (n0 + n1 + n2);
}

auto SimplexNoise2D::fbm(float x, float y, std::uint32_t octaves, float lacunarity, float persistence) const -> float {
    float amplitude = 1.0F;
    float frequency = 1.0F;
    float sum = 0.0F;
    float amplitude_sum = 0.0F;

    for (std::uint32_t octave = 0; octave < octaves; ++octave) {
        sum += sample(x * frequency, y * frequency) * amplitude;
        amplitude_sum += amplitude;

        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return amplitude_sum > 0.0F ? sum / amplitude_sum : 0.0F;
}
