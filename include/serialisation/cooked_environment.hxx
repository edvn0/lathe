#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "assets/hdr_image.hxx"
#include "serialisation/lbf_error.hxx"

inline constexpr std::uint16_t cooked_environment_version = 1;
inline constexpr std::uint16_t cooked_environment_oldest_readable_version = 1;

inline constexpr std::uint32_t cooked_environment_format = 97;

[[nodiscard]]
auto encode_cooked_environment(HdrImage const &image) -> std::vector<std::byte>;

[[nodiscard]]
auto decode_cooked_environment(std::span<std::byte const> payload,
                               std::uint16_t version = cooked_environment_version) -> std::expected<HdrImage, LbfError>;
