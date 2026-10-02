#pragma once

#include <cstddef>
#include <span>
#include <string>

// Lowercase hex SHA-256 digest (64 characters) of `data`.
[[nodiscard]]
auto sha256_hex(std::span<std::byte const> data) -> std::string;
