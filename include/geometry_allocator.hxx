#pragma once

#include <volk.h>

#include "error_context.hxx"
#include "geometry.hxx"

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

// A GeometryArenaT<Allocator> defers every offset decision to an Allocator
// satisfying this concept. `checkpoint()`/`rollback()` exist so
// GeometryArenaT::allocate_vertices/allocate_indices can undo a successful
// byte allocation if the subsequent GPU write fails, without that undo being
// expressible as deallocate() -- for BumpAllocator specifically, rollback
// also reclaims the alignment padding that a deallocate() of just the
// requested slice would leave behind.
template<typename A>
concept GeometryAllocatorPolicy = requires(A a, A const &const_a, VkDeviceSize size, VkDeviceSize alignment,
                                            GeometrySlice slice, typename A::Checkpoint checkpoint) {
    typename A::Checkpoint;

    { a.reset(size) } -> std::same_as<void>;
    { a.allocate(size, alignment) } -> std::same_as<std::expected<GeometrySlice, GeometryArenaError>>;
    { a.deallocate(slice) } -> std::same_as<void>;
    { a.checkpoint() } -> std::same_as<typename A::Checkpoint>;
    { a.rollback(checkpoint) } -> std::same_as<void>;
    { const_a.used_size() } -> std::same_as<VkDeviceSize>;
    { const_a.capacity() } -> std::same_as<VkDeviceSize>;
};

// Bump allocator: never frees, offsets only ever advance. This is the exact
// behavior GeometryArena has always had, lifted verbatim out of what used to
// be GeometryArena::allocate_bytes. Kept as the default backing allocator
// because most geometry in this engine is loaded once and never streamed
// out; see FreeListAllocator for the streaming case.
class BumpAllocator {
public:
    using Checkpoint = VkDeviceSize;

    auto reset(VkDeviceSize capacity) noexcept -> void {
        capacity_ = capacity;
        next_offset_ = 0;
    }

    [[nodiscard]]
    auto allocate(VkDeviceSize allocation_size, VkDeviceSize alignment) noexcept
            -> std::expected<GeometrySlice, GeometryArenaError> {

        if (allocation_size == 0 || alignment == 0 || !std::has_single_bit(alignment)) {

            return std::unexpected{GeometryArenaError{
                    .type = GeometryArenaErrorType::invalid_argument,
            }};
        }

        alignment = std::max(alignment, VkDeviceSize{4});

        if (next_offset_ > std::numeric_limits<VkDeviceSize>::max() - (alignment - 1)) {

            return std::unexpected{GeometryArenaError{
                    .type = GeometryArenaErrorType::size_overflow,
            }};
        }

        auto const offset = (next_offset_ + alignment - 1) & ~(alignment - 1);

        if (offset > capacity_ || allocation_size > capacity_ - offset) {

            return std::unexpected{GeometryArenaError{
                    .type = GeometryArenaErrorType::out_of_memory,
            }};
        }

        if (offset > std::numeric_limits<VkDeviceSize>::max() - allocation_size) {

            return std::unexpected{GeometryArenaError{
                    .type = GeometryArenaErrorType::size_overflow,
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
        // Bump allocator: ranges are never reclaimed individually.
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

// Address-ordered, coalescing free-list allocator with alignment-aware
// best-fit search. Unlike BumpAllocator, deallocate() actually reclaims the
// range so it can be handed back out by a later allocate() -- the piece
// streaming (e.g. terrain chunk geometry) needs and BumpAllocator cannot
// provide. Not currently wired into GeometryArena's default alias; flip
// `using GeometryArena = GeometryArenaT<FreeListAllocator>;` in
// geometry_arena.hxx when something needs it.
//
// Callers remain responsible for not calling deallocate() on a range the GPU
// may still be reading -- this allocator only tracks address space, not
// in-flight-frame safety. See docs/engine_review_followups.md and
// TerrainSlotPool's tick_retirement() for the deferred-release discipline
// that must sit in front of deallocate() in practice.
class FreeListAllocator {
public:
    struct FreeRange {
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
    };

    // Whole-vector snapshot rather than an allocation log: checkpoint/
    // rollback here is only ever used the way GeometryArenaT uses it --
    // checkpoint, allocate, and either commit or immediately rollback with
    // no other allocate()/deallocate() calls in between -- so the simplest
    // correct implementation is a full copy of the free-range list. The
    // free-range count stays small (bounded by fragmentation) so this copy
    // is cheap relative to the GPU write it brackets.
    struct Checkpoint {
        std::vector<FreeRange> free_ranges;
        VkDeviceSize used = 0;
    };

    auto reset(VkDeviceSize capacity) -> void;

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
    // Address-ordered, non-overlapping, non-adjacent (coalesced) at all times.
    std::vector<FreeRange> free_ranges_{};
    VkDeviceSize capacity_ = 0;
    VkDeviceSize used_ = 0;
};

static_assert(GeometryAllocatorPolicy<FreeListAllocator>);
