#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "assets/hdr_image.hxx"
#include "serialisation/lbf_error.hxx"

// ENVM chunk payload: an HDR environment as half-float RGBA, so loading one is a zstd decode and a byte reshuffle with
// no .hdr/.exr/.ktx2 parsing. Layout (little endian):
//
//   u32 width, u32 height, u32 vk_format (R16G16B16A16_SFLOAT only), u32 layers (1 = equirect, 6 = cubemap)
//   pixels: every layer's half-float RGBA, byte-shuffled (all low bytes, then all high bytes) so zstd finds long runs
//
// Versioning: the encoder always writes cooked_environment_version; the decoder accepts every version listed in
// docs/lathe-binary-format.md. See docs/ibl-and-skybox.md.
inline constexpr std::uint16_t cooked_environment_version = 1;
inline constexpr std::uint16_t cooked_environment_oldest_readable_version = 1;

// VK_FORMAT_R16G16B16A16_SFLOAT, spelled out so this header needs no Vulkan include.
inline constexpr std::uint32_t cooked_environment_format = 97;

[[nodiscard]]
auto encode_cooked_environment(HdrImage const &image) -> std::vector<std::byte>;

// Refuses layers other than 1 or 6, equirects that aren't 2:1 or exceed 16384x8192, cube faces that aren't square powers
// of two above 2048, other formats, and payloads whose size does not match exactly.
[[nodiscard]]
auto decode_cooked_environment(std::span<std::byte const> payload,
                               std::uint16_t version = cooked_environment_version) -> std::expected<HdrImage, LbfError>;
