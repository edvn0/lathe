#pragma once

#include <cstddef>
#include <span>
#include <string>

[[nodiscard]]
auto sha256_hex(std::span<std::byte const> data) -> std::string;
