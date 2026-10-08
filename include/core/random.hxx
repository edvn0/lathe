#pragma once

#include <cstdint>
#include <optional>
#include <random>

auto set_fixed_random_seed(std::optional<std::uint32_t> seed) noexcept -> void;

[[nodiscard]]
auto fixed_random_seed() noexcept -> std::optional<std::uint32_t>;

[[nodiscard]]
auto make_random_engine(std::uint32_t stream) -> std::mt19937;
