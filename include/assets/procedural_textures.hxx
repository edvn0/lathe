#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct GeneratedTexture {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mip_levels = 0;

    std::vector<std::byte> pixels;

    [[nodiscard]] auto level_width(std::uint32_t level) const noexcept -> std::uint32_t;
    [[nodiscard]] auto level_height(std::uint32_t level) const noexcept -> std::uint32_t;
    [[nodiscard]] auto level_offset(std::uint32_t level) const noexcept -> std::size_t;

    [[nodiscard]] auto alpha_coverage(std::uint32_t level, float cutoff) const noexcept -> float;
};

[[nodiscard]] auto make_grass_card_texture(std::uint32_t size = 128, float alpha_cutoff = 0.5F) -> GeneratedTexture;
