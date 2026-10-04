#include "assets/procedural_textures.hxx"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <random>

#include "gpu/image.hxx"

namespace {

    constexpr std::uint32_t texel_bytes = 4;

    struct LevelImage {
        std::uint32_t width = 0;
        std::uint32_t height = 0;

        // Linear RGBA, alpha straight (not premultiplied).
        std::vector<float> rgba;

        [[nodiscard]] auto at(std::uint32_t x, std::uint32_t y) -> float * {
            return &rgba[(static_cast<std::size_t>(y) * width + x) * 4U];
        }

        [[nodiscard]] auto at(std::uint32_t x, std::uint32_t y) const -> float const * {
            return &rgba[(static_cast<std::size_t>(y) * width + x) * 4U];
        }
    };

    // One blade silhouette in card space: x across 0..1, height 0 (root) .. 1 (top of the card).
    struct CardBlade {
        float root_x = 0.5F;
        float height = 1.0F;
        float half_width = 0.02F;
        float lean = 0.0F;
        float curve = 0.0F;
        float shade = 1.0F;
    };

    // Matches the blade geometry's taper (make_grass_clump_mesh()): wide low down, narrowing to the tip.
    constexpr float card_taper_exponent = 0.72F;

    [[nodiscard]] auto card_blades() -> std::vector<CardBlade> {
        // About as many blades as the clump shows across one card's width: a view sees two or three cards at once,
        // so a denser card makes distant grass read thicker than the blades it replaces.
        constexpr std::uint32_t blade_count = 11;

        // Fixed seed: the texture is part of the engine's grass clump, not per instance.
        std::mt19937 random_engine{0x43415244U};
        std::uniform_real_distribution<float> unit{0.0F, 1.0F};
        auto const range = [&](float low, float high) { return std::lerp(low, high, unit(random_engine)); };

        std::vector<CardBlade> blades;
        blades.reserve(blade_count);

        for (std::uint32_t index = 0; index < blade_count; ++index) {
            // Stratified across the card so there are no bald patches.
            auto const root_x =
                    (static_cast<float>(index) + 0.5F + range(-0.4F, 0.4F)) / static_cast<float>(blade_count);

            // Shorter towards the sides, like the clump's rounded silhouette.
            auto const edge = std::abs(root_x - 0.5F) * 2.0F;
            auto const height = range(0.58F, 0.98F) * std::lerp(1.0F, 0.72F, edge * edge);

            auto const lean = range(-0.10F, 0.10F);

            blades.push_back(CardBlade{
                    .root_x = root_x,
                    .height = height,
                    .half_width = range(0.018F, 0.034F),
                    .lean = lean,
                    .curve = lean * range(0.2F, 0.8F),
                    .shade = range(0.86F, 1.0F),
            });
        }

        return blades;
    }

    // Level 0 with `supersample`^2 samples per texel: alpha is the covered share, RGB the shade of the front-most
    // blade averaged over the covered samples (white where nothing is).
    [[nodiscard]] auto rasterise_card(std::uint32_t size, std::vector<CardBlade> const &blades) -> LevelImage {
        constexpr std::uint32_t supersample = 4;

        LevelImage image{
                .width = size, .height = size, .rgba = std::vector<float>(static_cast<std::size_t>(size) * size * 4U)};

        auto const inv_size = 1.0F / static_cast<float>(size);

        for (std::uint32_t row = 0; row < size; ++row) {
            for (std::uint32_t column = 0; column < size; ++column) {
                auto covered = 0U;
                auto shade_sum = 0.0F;

                for (std::uint32_t sample_y = 0; sample_y < supersample; ++sample_y) {
                    for (std::uint32_t sample_x = 0; sample_x < supersample; ++sample_x) {
                        auto const x = (static_cast<float>(column) +
                                        (static_cast<float>(sample_x) + 0.5F) / static_cast<float>(supersample)) *
                                       inv_size;

                        // Row 0 is the top of the card.
                        auto const height = 1.0F - (static_cast<float>(row) + (static_cast<float>(sample_y) + 0.5F) /
                                                                                      static_cast<float>(supersample)) *
                                                           inv_size;

                        // Later blades are in front.
                        for (auto blade = blades.rbegin(); blade != blades.rend(); ++blade) {
                            auto const t = height / blade->height;
                            if (t < 0.0F || t > 1.0F) {
                                continue;
                            }

                            auto const centre = blade->root_x + blade->lean * t + blade->curve * t * t;
                            auto const distance = std::abs(x - centre);
                            if (distance > blade->half_width) {
                                continue;
                            }

                            if (distance <= blade->half_width * std::pow(1.0F - t, card_taper_exponent)) {
                                ++covered;
                                shade_sum += blade->shade;
                                break;
                            }
                        }
                    }
                }

                auto *texel = image.at(column, row);
                auto const shade = covered > 0 ? shade_sum / static_cast<float>(covered) : 1.0F;
                texel[0] = shade;
                texel[1] = shade;
                texel[2] = shade;
                texel[3] = static_cast<float>(covered) / static_cast<float>(supersample * supersample);
            }
        }

        return image;
    }

    // 2x2 box filter; colour weighted by alpha so the transparent white around the blades doesn't bleed in.
    [[nodiscard]] auto downsample(LevelImage const &source) -> LevelImage {
        LevelImage result{.width = std::max(source.width / 2U, 1U), .height = std::max(source.height / 2U, 1U)};
        result.rgba.resize(static_cast<std::size_t>(result.width) * result.height * 4U);

        for (std::uint32_t y = 0; y < result.height; ++y) {
            for (std::uint32_t x = 0; x < result.width; ++x) {
                float alpha_sum = 0.0F;
                std::array<float, 3> weighted{};
                std::array<float, 3> plain{};
                auto taps = 0U;

                for (std::uint32_t dy = 0; dy < 2; ++dy) {
                    for (std::uint32_t dx = 0; dx < 2; ++dx) {
                        auto const source_x = std::min(x * 2U + dx, source.width - 1U);
                        auto const source_y = std::min(y * 2U + dy, source.height - 1U);
                        auto const *texel = source.at(source_x, source_y);

                        alpha_sum += texel[3];
                        for (std::size_t channel = 0; channel < 3; ++channel) {
                            weighted[channel] += texel[channel] * texel[3];
                            plain[channel] += texel[channel];
                        }
                        ++taps;
                    }
                }

                auto *texel = result.at(x, y);
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    texel[channel] = alpha_sum > 0.0F ? weighted[channel] / alpha_sum
                                                      : plain[channel] / static_cast<float>(taps);
                }
                texel[3] = alpha_sum / static_cast<float>(taps);
            }
        }

        return result;
    }

    [[nodiscard]] auto coverage_of(LevelImage const &image, float alpha_scale, float cutoff) -> float {
        auto const texel_count = static_cast<std::size_t>(image.width) * image.height;
        std::size_t covered = 0;

        for (std::size_t texel = 0; texel < texel_count; ++texel) {
            if (std::min(image.rgba[texel * 4U + 3U] * alpha_scale, 1.0F) >= cutoff) {
                ++covered;
            }
        }

        return static_cast<float>(covered) / static_cast<float>(texel_count);
    }

    // The alpha scale that gives `image` `target` coverage at `cutoff` (Castano, "Computing Alpha Mipmaps").
    [[nodiscard]] auto coverage_preserving_scale(LevelImage const &image, float target, float cutoff) -> float {
        auto low = 0.0F;
        auto high = 16.0F;

        for (auto iteration = 0; iteration < 24; ++iteration) {
            auto const middle = (low + high) * 0.5F;
            if (coverage_of(image, middle, cutoff) < target) {
                low = middle;
            } else {
                high = middle;
            }
        }

        return high;
    }

    [[nodiscard]] auto to_unorm8(float value) -> std::byte {
        return static_cast<std::byte>(static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F)));
    }

} // namespace

auto GeneratedTexture::level_width(std::uint32_t level) const noexcept -> std::uint32_t {
    return std::max(width >> level, 1U);
}

auto GeneratedTexture::level_height(std::uint32_t level) const noexcept -> std::uint32_t {
    return std::max(height >> level, 1U);
}

auto GeneratedTexture::level_offset(std::uint32_t level) const noexcept -> std::size_t {
    return mip_chain_offset(width, height, texel_bytes, level);
}

auto GeneratedTexture::alpha_coverage(std::uint32_t level, float cutoff) const noexcept -> float {
    auto const texel_count = static_cast<std::size_t>(level_width(level)) * level_height(level);
    auto const offset = level_offset(level);
    std::size_t covered = 0;

    for (std::size_t texel = 0; texel < texel_count; ++texel) {
        auto const alpha = static_cast<float>(std::to_integer<std::uint8_t>(pixels[offset + texel * 4U + 3U])) / 255.0F;
        if (alpha >= cutoff) {
            ++covered;
        }
    }

    return static_cast<float>(covered) / static_cast<float>(texel_count);
}

auto make_grass_card_texture(std::uint32_t size, float alpha_cutoff) -> GeneratedTexture {
    size = std::bit_floor(std::max(size, 4U));

    GeneratedTexture texture{
            .width = size,
            .height = size,
            .mip_levels = static_cast<std::uint32_t>(std::bit_width(size)),
    };
    texture.pixels.resize(texture.level_offset(texture.mip_levels));

    auto level = rasterise_card(size, card_blades());
    auto const target_coverage = coverage_of(level, 1.0F, alpha_cutoff);

    for (std::uint32_t mip = 0; mip < texture.mip_levels; ++mip) {
        if (mip > 0) {
            level = downsample(level);
        }

        // The box filter is kept unscaled for the next level; only what's stored is rescaled.
        auto const alpha_scale = mip == 0 ? 1.0F : coverage_preserving_scale(level, target_coverage, alpha_cutoff);

        auto *out = texture.pixels.data() + texture.level_offset(mip);
        auto const texel_count = static_cast<std::size_t>(level.width) * level.height;

        for (std::size_t texel = 0; texel < texel_count; ++texel) {
            auto const *source = &level.rgba[texel * 4U];
            out[texel * 4U + 0U] = to_unorm8(source[0]);
            out[texel * 4U + 1U] = to_unorm8(source[1]);
            out[texel * 4U + 2U] = to_unorm8(source[2]);
            out[texel * 4U + 3U] = to_unorm8(source[3] * alpha_scale);
        }
    }

    return texture;
}
