#pragma once

#include <cstdint>
#include <optional>
#include <random>

// Seed for procedural scene content. Unset, make_random_engine() uses std::random_device; the benchmark pins
// it so every build populates the same scene. Set once at startup; not synchronised.
auto set_fixed_random_seed(std::optional<std::uint32_t> seed) noexcept -> void;

[[nodiscard]]
auto fixed_random_seed() noexcept -> std::optional<std::uint32_t>;

// `stream` keeps engines made under the same seed from producing the same sequence.
[[nodiscard]]
auto make_random_engine(std::uint32_t stream) -> std::mt19937;
