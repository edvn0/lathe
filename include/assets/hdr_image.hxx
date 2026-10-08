#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

struct HdrImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    std::uint32_t layers = 1;

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

inline constexpr float hdr_max_radiance = 65000.0F;

[[nodiscard]]
auto decode_radiance_hdr(std::span<std::byte const> encoded) -> std::expected<HdrImage, HdrImageError>;

[[nodiscard]]
auto decode_ktx2_float(std::span<std::byte const> encoded) -> std::expected<HdrImage, HdrImageError>;

[[nodiscard]]
auto load_hdr_image(std::string_view path) -> std::expected<HdrImage, HdrImageError>;

[[nodiscard]]
auto sanitize_to_half(float value) noexcept -> std::uint16_t;
