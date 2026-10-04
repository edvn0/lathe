#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// An RGBA8 texture generated in-process, with its whole mip chain.
struct GeneratedTexture {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mip_levels = 0;

    // Every level from 0 down to 1x1, tightly packed (mip_chain_offset() in gpu/image.hxx), 4 bytes per texel.
    std::vector<std::byte> pixels;

    [[nodiscard]] auto level_width(std::uint32_t level) const noexcept -> std::uint32_t;
    [[nodiscard]] auto level_height(std::uint32_t level) const noexcept -> std::uint32_t;
    [[nodiscard]] auto level_offset(std::uint32_t level) const noexcept -> std::size_t;

    // The share of level `level`'s texels whose alpha is at least `cutoff`.
    [[nodiscard]] auto alpha_coverage(std::uint32_t level, float cutoff) const noexcept -> float;
};

// The far-LOD texture of the engine grass clump (make_grass_clump_mesh()): a row of blade silhouettes for its
// crossed cards, `size` x `size`. Rows run top (tips, v = 0) to bottom (roots, v = 1). RGB is near-white so the
// material's base colour sets the grass colour, as it does for the blade geometry. Each mip keeps level 0's share of
// texels at or above `alpha_cutoff`, so the blades don't thin out with distance the way a plain box-filtered alpha
// does.
[[nodiscard]] auto make_grass_card_texture(std::uint32_t size = 128, float alpha_cutoff = 0.5F) -> GeneratedTexture;
