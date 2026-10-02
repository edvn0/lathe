#pragma once

#include <cstdint>

// Layout of the per-frame meshlet visibility bitset used by meshlet-level occlusion culling
// (docs/occlusion-culling.md, "Meshlet level"): one bit per meshlet of every instance of every opaque and mask meshlet
// batch. Instance i of a batch owns the bits [offset + i * meshlet_count, offset + (i + 1) * meshlet_count), where
// `offset` is what MeshletVisibilityLayout::reserve() handed out for the batch. GpuDraw::meshlet_visibility_offset
// carries the per-instance start to the task shader, which addresses word `bit >> 5`, bit `bit & 31`.

// 2^28 bits = 32 MiB per frame in flight. A frame needing more runs without meshlet occlusion; instance-level
// occlusion is unaffected.
inline constexpr std::uint64_t maximum_meshlet_visibility_bits = std::uint64_t{1} << 28U;

// 32-bit words that hold `bits` bits (the bitset is an array of u32 for InterlockedOr).
[[nodiscard]] constexpr auto meshlet_visibility_word_count(std::uint64_t bits) noexcept -> std::uint64_t {
    return (bits + 31U) / 32U;
}

// Hands out contiguous, disjoint bit ranges, one per batch. 64-bit throughout so an oversized frame is detected
// (fits() is false) rather than wrapping.
class MeshletVisibilityLayout {
public:
    // Reserves instance_count * meshlet_count bits and returns the batch's first bit.
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

    // Every offset handed out is below the cap, so GpuDraw::meshlet_visibility_offset (32 bits) holds it.
    [[nodiscard]] constexpr auto fits() const noexcept -> bool {
        return total_bits_ <= maximum_meshlet_visibility_bits;
    }

private:
    std::uint64_t total_bits_ = 0;
};

// The first bit of `instance`'s meshlets in a batch whose range starts at `batch_first_bit`. 0 once it is past the cap:
// such a frame doesn't use the bitset (fits() is false), so the value is never read.
[[nodiscard]] constexpr auto meshlet_visibility_offset(std::uint64_t batch_first_bit, std::uint32_t instance,
                                                       std::uint32_t meshlet_count) noexcept -> std::uint32_t {
    auto const first_bit = batch_first_bit + std::uint64_t{instance} * meshlet_count;
    return first_bit < maximum_meshlet_visibility_bits ? static_cast<std::uint32_t>(first_bit) : 0U;
}
