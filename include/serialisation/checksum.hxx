#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

[[nodiscard]]
auto xxh64(std::span<std::byte const> bytes, std::uint64_t seed = 0) noexcept -> std::uint64_t;

[[nodiscard]]
auto xxh64(std::string_view text, std::uint64_t seed = 0) noexcept -> std::uint64_t;

class Xxh64Stream {
public:
    explicit Xxh64Stream(std::uint64_t seed = 0) noexcept;

    auto update(std::span<std::byte const> bytes) noexcept -> void;

    [[nodiscard]] auto digest() const noexcept -> std::uint64_t;

private:
    std::uint64_t seed_;
    std::array<std::uint64_t, 4> lanes_;
    std::uint64_t total_ = 0;
    std::array<std::byte, 32> buffer_{};
    std::size_t buffered_ = 0;
};
