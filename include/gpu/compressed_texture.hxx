#pragma once

#include <volk.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/fly_string.hxx"

enum class TextureRole : std::uint8_t {
    colour,
    generic,
    normal_map,
};

struct CompressedMipLevel {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t byte_offset = 0;
    std::uint32_t byte_length = 0;
};

struct CompressedTexture {
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<CompressedMipLevel> mips;
    std::vector<std::byte> data;
    FlyString debug_name;
};

[[nodiscard]] constexpr auto compressed_block_bytes(std::uint32_t format) noexcept -> std::uint32_t {
    switch (format) {
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        case VK_FORMAT_BC4_UNORM_BLOCK:
        case VK_FORMAT_BC4_SNORM_BLOCK:
            return 8;
        case VK_FORMAT_BC2_UNORM_BLOCK:
        case VK_FORMAT_BC2_SRGB_BLOCK:
        case VK_FORMAT_BC3_UNORM_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK:
        case VK_FORMAT_BC5_SNORM_BLOCK:
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK:
            return 16;
        default:
            return 0;
    }
}

inline constexpr std::uint32_t max_compressed_texture_extent = 16384;

[[nodiscard]]
inline auto validate_compressed_texture(CompressedTexture const &texture) noexcept -> std::optional<std::string_view> {
    auto const block_bytes = compressed_block_bytes(static_cast<std::uint32_t>(texture.format));

    if (block_bytes == 0) {
        return "unsupported texture format";
    }

    if (texture.width == 0 || texture.height == 0 || texture.width > max_compressed_texture_extent ||
        texture.height > max_compressed_texture_extent) {
        return "texture extent out of range";
    }

    auto const max_levels = static_cast<std::size_t>(std::bit_width(std::max(texture.width, texture.height)));

    if (texture.mips.empty() || texture.mips.size() > max_levels) {
        return "texture mip count out of range";
    }

    for (std::size_t level = 0; level < texture.mips.size(); ++level) {
        auto const &mip = texture.mips[level];
        auto const width = std::max<std::uint32_t>(texture.width >> level, 1);
        auto const height = std::max<std::uint32_t>(texture.height >> level, 1);

        if (mip.width != width || mip.height != height) {
            return "texture mip extent does not match its level";
        }

        auto const blocks = static_cast<std::uint64_t>((width + 3) / 4) * ((height + 3) / 4);

        if (mip.byte_length != blocks * block_bytes) {
            return "texture mip size does not match its format";
        }

        if (mip.byte_offset % block_bytes != 0 ||
            static_cast<std::uint64_t>(mip.byte_offset) + mip.byte_length > texture.data.size()) {
            return "texture mip lies outside its data";
        }
    }

    return std::nullopt;
}
