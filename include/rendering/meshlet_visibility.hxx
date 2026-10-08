#pragma once

#include <cstdint>

inline constexpr std::uint64_t maximum_meshlet_visibility_bits = std::uint64_t{1} << 28U;

[[nodiscard]] constexpr auto meshlet_visibility_word_count(std::uint64_t bits) noexcept -> std::uint64_t {
    return (bits + 31U) / 32U;
}

class MeshletVisibilityLayout {
public:
    [[nodiscard]] constexpr auto reserve(std::uint32_t instance_count,
                                         std::uint32_t meshlet_count) noexcept -> std::uint64_t {
        auto const first_bit = total_bits_;
        total_bits_ += std::uint64_t{instance_count} * meshlet_count;
        return first_bit;
    }

    [[nodiscard]] constexpr auto total_bits() const noexcept -> std::uint64_t { return total_bits_; }
    [[nodiscard]] constexpr auto word_count() const noexcept -> std::uint64_t {
        return meshlet_visibility_word_count(total_bits_);
    }

    [[nodiscard]] constexpr auto fits() const noexcept -> bool {
        return total_bits_ <= maximum_meshlet_visibility_bits;
    }

private:
    std::uint64_t total_bits_ = 0;
};

[[nodiscard]] constexpr auto meshlet_visibility_offset(std::uint64_t batch_first_bit, std::uint32_t instance,
                                                       std::uint32_t meshlet_count) noexcept -> std::uint32_t {
    auto const first_bit = batch_first_bit + std::uint64_t{instance} * meshlet_count;
    return first_bit < maximum_meshlet_visibility_bits ? static_cast<std::uint32_t>(first_bit) : 0U;
}
