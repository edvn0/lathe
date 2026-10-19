#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

// A 3D colour lookup table for grading. It maps display-referred colour (sRGB-encoded, 0 .. 1) to graded colour, so
// composition applies it after tone mapping. Entries are stored in .cube order: red varies fastest, then green, then
// blue.
struct ColourLut {
    std::uint32_t size = 0;

    std::vector<glm::vec3> entries;

    [[nodiscard]] auto at(std::uint32_t r, std::uint32_t g, std::uint32_t b) const noexcept -> glm::vec3 const & {
        return entries[r + size * (g + size * b)];
    }

    // Trilinear lookup; `colour` is clamped to 0 .. 1.
    [[nodiscard]] auto sample(glm::vec3 colour) const noexcept -> glm::vec3;

    // The GPU layout: a `size`^3 RGBA8 volume with red along x, green along y and blue along z, which is the order of
    // `entries`.
    [[nodiscard]] auto to_volume_rgba8() const -> std::vector<std::byte>;
};

enum class DefaultColourLut : std::uint8_t {
    cinematic,
    vintage,
    noir,
};

inline constexpr auto default_colour_lut_count = std::size_t{3};

inline constexpr auto default_colour_lut_size = std::uint32_t{32};

[[nodiscard]] constexpr auto default_colour_lut_name(DefaultColourLut lut) noexcept -> std::string_view {
    switch (lut) {
        case DefaultColourLut::cinematic:
            return "cinematic";
        case DefaultColourLut::vintage:
            return "vintage";
        case DefaultColourLut::noir:
            return "noir";
    }

    return "unknown";
}

// Maps every colour to itself.
[[nodiscard]] auto make_identity_colour_lut(std::uint32_t size) -> ColourLut;

// cinematic: teal shadows, orange highlights, a touch more contrast and saturation.
// vintage: lifted, faded blacks, muted colour and a warm cast.
// noir: near-monochrome with strong contrast and slightly cool shadows.
[[nodiscard]] auto make_default_colour_lut(DefaultColourLut lut, std::uint32_t size = default_colour_lut_size)
        -> ColourLut;

// Parses an Adobe/Resolve .cube file with a 3D table (LUT_3D_SIZE 2 .. 64) over the default 0 .. 1 domain. Returns
// nothing for 1D tables, other domains, missing entries or malformed numbers.
[[nodiscard]] auto parse_cube_lut(std::string_view text) -> std::optional<ColourLut>;
