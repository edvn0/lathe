#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// A high dynamic range environment source decoded to half-float RGBA, ready to upload as R16G16B16A16_SFLOAT. See
// docs/ibl-and-skybox.md.
struct HdrImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    // 1 for an equirect panorama (width == 2 * height), 6 for a cubemap (square faces, Vulkan face order).
    std::uint32_t layers = 1;

    // RGBA half floats, layer-major, row-major within a layer. Finite and within [0, 65000].
    std::vector<std::uint16_t> pixels;
};

enum class HdrImageErrorType : std::uint8_t {
    unreadable,
    unsupported_format,
    invalid_data,
    too_large,
};

struct HdrImageError {
    HdrImageErrorType type = HdrImageErrorType::invalid_data;
    std::string message;
};

inline constexpr std::uint32_t hdr_equirect_max_width = 16384;
inline constexpr std::uint32_t hdr_equirect_max_height = 8192;
inline constexpr std::uint32_t hdr_cube_max_face_size = 2048;

// Largest value kept: finite, and comfortably below half-float max (65504) so filtering cannot overflow.
inline constexpr float hdr_max_radiance = 65000.0F;

// Radiance `.hdr` (RGBE), as an equirect.
[[nodiscard]]
auto decode_radiance_hdr(std::span<std::byte const> encoded) -> std::expected<HdrImage, HdrImageError>;

// KTX2 with an uncompressed float format: R16G16B16A16_SFLOAT, R16G16B16_SFLOAT, R32G32B32A32_SFLOAT,
// R32G32B32_SFLOAT, E5B9G9R9_UFLOAT_PACK32 or B10G11R11_UFLOAT_PACK32. One face is an equirect, six faces a
// cubemap. Only mip 0 is read. Block-compressed and Basis Universal files are refused.
[[nodiscard]]
auto decode_ktx2_float(std::span<std::byte const> encoded) -> std::expected<HdrImage, HdrImageError>;

// Dispatches on the extension: `.hdr`, `.exr` (equirect) or `.ktx2`.
[[nodiscard]]
auto load_hdr_image(std::string_view path) -> std::expected<HdrImage, HdrImageError>;

// Float to the stored half, with NaN/inf -> 0 and the clamp to [0, hdr_max_radiance].
[[nodiscard]]
auto sanitize_to_half(float value) noexcept -> std::uint16_t;
