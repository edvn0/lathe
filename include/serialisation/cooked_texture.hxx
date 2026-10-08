#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "assets/texture_pipeline.hxx"
#include "gpu/compressed_texture.hxx"
#include "serialisation/lbf_error.hxx"

inline constexpr std::uint16_t cooked_texture_version = 1;
inline constexpr std::uint16_t cooked_texture_oldest_readable_version = 1;

[[nodiscard]]
auto encode_cooked_texture(CompressedTexture const &texture, TextureRole role) -> std::vector<std::byte>;

struct CookedTexture {
    CompressedTexture texture;
    TextureRole role = TextureRole::colour;
};

[[nodiscard]]
auto decode_cooked_texture(std::span<std::byte const> payload, std::uint16_t version = cooked_texture_version)
        -> std::expected<CookedTexture, LbfError>;
