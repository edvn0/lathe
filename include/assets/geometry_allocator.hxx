#pragma once

#include <volk.h>

#include "assets/geometry.hxx"
#include "core/error_context.hxx"

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

enum class GeometryArenaErrorType : std::uint8_t {
    invalid_argument,
    unsupported_index_type,
    out_of_memory,
    size_overflow,
    device_error,
};

struct GeometryArenaError {
    GeometryArenaErrorType type = GeometryArenaErrorType::invalid_argument;

    std::optional<ErrorCause> cause;
};

template<>
struct std::formatter<GeometryArenaErrorType> : std::formatter<std::string_view> {
    constexpr auto format(GeometryArenaErrorType error, std::format_context &context) const {

        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case GeometryArenaErrorType::invalid_argument:
                    return "invalid_argument";

                case GeometryArenaErrorType::unsupported_index_type:
                    return "unsupported_index_type";

                case GeometryArenaErrorType::out_of_memory:
                    return "out_of_memory";

                case GeometryArenaErrorType::size_overflow:
                    return "size_overflow";

                case GeometryArenaErrorType::device_error:
                    return "device_error";
            }

            return "unknown_geometry_arena_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

// The offset policy behind GeometryArenaT. checkpoint()/rollback() undo an allocation if the following GPU
// write fails; for BumpAllocator that also reclaims the alignment padding.
template<typename A>
concept GeometryAllocatorPolicy = requires(A a, A const &const_a, VkDeviceSize size, VkDeviceSize alignment,
                                           GeometrySlice slice, typename A::Checkpoint checkpoint) {
    typename A::Checkpoint;

    { a.reset(size) } -> std::same_as<void>;
    { a.grow(size) } -> std::same_as<void>;
    { a.allocate(size, alignment) } -> std::same_as<std::expected<GeometrySlice, GeometryArenaError>>;
    { a.deallocate(slice) } -> std::same_as<void>;
    { a.checkpoint() } -> std::same_as<typename A::Checkpoint>;
    { a.rollback(checkpoint) } -> std::same_as<void>;
    { const_a.used_size() } -> std::same_as<VkDeviceSize>;
    { const_a.capacity() } -> std::same_as<VkDeviceSize>;
};

// grow(new_capacity) extends the address space in place (no-op unless larger); existing offsets stay valid, and a
// rollback() to a checkpoint from before the growth keeps the added space.

// Never frees; offsets only advance.
class BumpAllocator {
public:
    using Checkpoint = VkDeviceSize;

    auto reset(VkDeviceSize capacity) noexcept -> void {
        capacity_ = capacity;
        next_offset_ = 0;
    }

    auto grow(VkDeviceSize new_capacity) noexcept -> void { capacity_ = std::max(capacity_, new_capacity); }

    [[nodiscard]]
    auto allocate(VkDeviceSize allocation_size, VkDeviceSize alignment) noexcept
            -> std::expected<GeometrySlice, GeometryArenaError> {

        if (allocation_size == 0 || alignment == 0 || !std::has_single_bit(alignment)) {

            return std::unexpected{GeometryArenaError{
                    .type = GeometryArenaErrorType::invalid_argument,
                    .cause = std::nullopt,
            }};
        }

        alignment = std::max(alignment, VkDeviceSize{4});

        if (next_offset_ > std::numeric_limits<VkDeviceSize>::max() - (alignment - 1)) {

            return std::unexpected{GeometryArenaError{
                    .type = GeometryArenaErrorType::size_overflow,
                    .cause = std::nullopt,
            }};
        }

        auto const offset = (next_offset_ + alignment - 1) & ~(alignment - 1);

        if (offset > capacity_ || allocation_size > capacity_ - offset) {

            return std::unexpected{GeometryArenaError{
                    .type = GeometryArenaErrorType::out_of_memory,
                    .cause = std::nullopt,
            }};
        }

        if (offset > std::numeric_limits<VkDeviceSize>::max() - allocation_size) {

            return std::unexpected{GeometryArenaError{
                    .type = GeometryArenaErrorType::size_overflow,
                    .cause = std::nullopt,
            }};
        }

        next_offset_ = offset + allocation_size;

        return GeometrySlice{
                .offset = offset,
                .size = allocation_size,
                .reserved_size = allocation_size,
        };
    }

    auto deallocate(GeometrySlice const &) noexcept -> void {
        // Bump allocators never free.
    }

    [[nodiscard]]
    auto checkpoint() const noexcept -> Checkpoint {
        return next_offset_;
    }

    auto rollback(Checkpoint checkpoint) noexcept -> void { next_offset_ = checkpoint; }

    [[nodiscard]]
    auto used_size() const noexcept -> VkDeviceSize {
        return next_offset_;
    }

    [[nodiscard]]
    auto capacity() const noexcept -> VkDeviceSize {
        return capacity_;
    }

private:
    VkDeviceSize capacity_ = 0;
    VkDeviceSize next_offset_ = 0;
};

static_assert(GeometryAllocatorPolicy<BumpAllocator>);

// Address-ordered, coalescing free-list with alignment-aware best fit.
//
// Only tracks address space: callers must not deallocate ranges the GPU may still read.
class FreeListAllocator {
public:
    struct FreeRange {
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
    };

    // A full copy of the free list. Only used as checkpoint, allocate, then commit or rollback, and the list stays
    // short.
    struct Checkpoint {
        std::vector<FreeRange> free_ranges;
        VkDeviceSize used = 0;
        VkDeviceSize capacity = 0;
    };

    auto reset(VkDeviceSize capacity) -> void;

    auto grow(VkDeviceSize new_capacity) -> void;

    [[nodiscard]]
    auto allocate(VkDeviceSize allocation_size, VkDeviceSize alignment)
            -> std::expected<GeometrySlice, GeometryArenaError>;

    auto deallocate(GeometrySlice const &slice) -> void;

    [[nodiscard]]
    auto checkpoint() const -> Checkpoint;

    auto rollback(Checkpoint const &checkpoint) -> void;

    [[nodiscard]]
    auto used_size() const noexcept -> VkDeviceSize {
        return used_;
    }

    [[nodiscard]]
    auto capacity() const noexcept -> VkDeviceSize {
        return capacity_;
    }

private:
    // Marks [from, capacity_) free, merging with a free range that ends at `from`.
    auto add_free_tail(VkDeviceSize from) -> void;

    // Address-ordered, non-overlapping and coalesced.
    std::vector<FreeRange> free_ranges_{};
    VkDeviceSize capacity_ = 0;
    VkDeviceSize used_ = 0;
};

static_assert(GeometryAllocatorPolicy<FreeListAllocator>);
