#pragma once

#include <cstdint>
#include <optional>
#include <random>

// Process-wide seeding for procedural scene content (grass placement,
// scattered props). Unset -- the default -- make_random_engine() seeds from
// std::random_device like before; the benchmark pins it (--seed=) so a
// base and a head build populate the identical scene. Set once at startup,
// before the scene is populated; not synchronised.
auto set_fixed_random_seed(std::optional<std::uint32_t> seed) noexcept -> void;

[[nodiscard]]
auto fixed_random_seed() noexcept -> std::optional<std::uint32_t>;

// `stream` tells call sites apart, so two engines made under the same fixed
// seed don't produce the same sequence.
[[nodiscard]]
auto make_random_engine(std::uint32_t stream) -> std::mt19937;
