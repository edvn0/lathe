#pragma once

#include <concepts>
#include <cstdint>
#include <string>

namespace detail {
    [[nodiscard]]
    auto human_readable_bytes_impl(std::uint64_t size) -> std::string;
} // namespace detail

// "512 B", "128 KiB", "1.50 MiB". Binary (1024) units.
[[nodiscard]]
auto human_readable_bytes(std::integral auto size) -> std::string {
    return detail::human_readable_bytes_impl(static_cast<std::uint64_t>(size));
}
