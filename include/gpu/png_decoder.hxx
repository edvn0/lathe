#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

struct PngRgba8 {
    // width * height * 4 bytes, rows one after another.
    std::vector<std::byte> pixels;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct PngDecodeOptions {
    // Off forces the portable code even where SIMD is available (the tests compare both).
    bool allow_simd = true;
};

[[nodiscard]] auto has_png_signature(std::span<std::byte const> encoded) noexcept -> bool;

// Decodes a PNG straight to RGBA8, several times faster than stb_image: libdeflate for the zlib stream, an unfilter
// that is SIMD for three- and four-byte pixels, and no copy of the result. It covers what textures use (8 bits per
// channel, not interlaced; grey, grey with alpha, RGB, RGB with alpha, and palette, each with tRNS) and gives the same
// pixels as stb_image for them. Anything else, and any damaged file, returns nothing, and the caller falls back to
// stb_image to decode it or to report why it cannot.
[[nodiscard]] auto decode_png_fast(std::span<std::byte const> encoded, PngDecodeOptions const &options = {})
        -> std::optional<PngRgba8>;
