#include "assets/geometry_allocator.hxx"

#include <algorithm>
#include <bit>
#include <limits>

namespace {

    [[nodiscard]]
    constexpr auto align_up(VkDeviceSize value, VkDeviceSize alignment) noexcept -> VkDeviceSize {
        return (value + alignment - 1) & ~(alignment - 1);
    }

} // namespace

auto FreeListAllocator::reset(VkDeviceSize capacity) -> void {
    free_ranges_.clear();

    if (capacity > 0) {
        free_ranges_.push_back(FreeRange{.offset = 0, .size = capacity});
    }

    capacity_ = capacity;
    used_ = 0;
}

auto FreeListAllocator::allocate(VkDeviceSize allocation_size, VkDeviceSize alignment)
        -> std::expected<GeometrySlice, GeometryArenaError> {

    if (allocation_size == 0 || alignment == 0 || !std::has_single_bit(alignment)) {

        return std::unexpected{GeometryArenaError{
                .type = GeometryArenaErrorType::invalid_argument,
                .cause = std::nullopt,
        }};
    }

    alignment = std::max(alignment, VkDeviceSize{4});

    // Best fit by usable range size, tie-broken by lowest offset (i.e. the
    // first candidate found, since free_ranges_ is address-ordered) --
    // minimizes leftover fragmentation relative to first-fit.
    auto best = free_ranges_.end();
    VkDeviceSize best_aligned_offset = 0;

    for (auto it = free_ranges_.begin(); it != free_ranges_.end(); ++it) {

        if (it->offset > std::numeric_limits<VkDeviceSize>::max() - (alignment - 1)) {
            continue;
        }

        auto const aligned_offset = align_up(it->offset, alignment);
        auto const end = it->offset + it->size;

        if (aligned_offset > end || allocation_size > end - aligned_offset) {
            continue;
        }

        if (best == free_ranges_.end() || it->size < best->size) {
            best = it;
            best_aligned_offset = aligned_offset;
        }
    }

    if (best == free_ranges_.end()) {

        return std::unexpected{GeometryArenaError{
                .type = GeometryArenaErrorType::out_of_memory,
                .cause = std::nullopt,
        }};
    }

    auto const range = *best;
    auto const index = std::distance(free_ranges_.begin(), best);
    free_ranges_.erase(best);

    auto insert_at = free_ranges_.begin() + index;

    if (auto const back_offset = best_aligned_offset + allocation_size; back_offset < range.offset + range.size) {

        insert_at = free_ranges_.insert(insert_at, FreeRange{
                                                            .offset = back_offset,
                                                            .size = (range.offset + range.size) - back_offset,
                                                    });
    }

    if (best_aligned_offset > range.offset) {

        free_ranges_.insert(insert_at, FreeRange{
                                                .offset = range.offset,
                                                .size = best_aligned_offset - range.offset,
                                        });
    }

    used_ += allocation_size;

    return GeometrySlice{
            .offset = best_aligned_offset,
            .size = allocation_size,
            .reserved_size = allocation_size,
    };
}

auto FreeListAllocator::deallocate(GeometrySlice const &slice) -> void {

    if (slice.size == 0) {
        return;
    }

    auto const insert_pos =
            std::ranges::lower_bound(free_ranges_, slice.offset, {}, [](FreeRange const &range) { return range.offset; });

    auto it = free_ranges_.insert(insert_pos, FreeRange{.offset = slice.offset, .size = slice.size});

    // Coalesce with the following range first -- merging it into *it does
    // not invalidate `it`, whereas merging the preceding range would.
    if (auto next = std::next(it); next != free_ranges_.end() && it->offset + it->size == next->offset) {
        it->size += next->size;
        free_ranges_.erase(next);
    }

    if (it != free_ranges_.begin()) {

        if (auto prev = std::prev(it); prev->offset + prev->size == it->offset) {
            prev->size += it->size;
            free_ranges_.erase(it);
        }
    }

    used_ -= slice.size;
}

auto FreeListAllocator::checkpoint() const -> Checkpoint {
    return Checkpoint{.free_ranges = free_ranges_, .used = used_};
}

auto FreeListAllocator::rollback(Checkpoint const &checkpoint) -> void {
    free_ranges_ = checkpoint.free_ranges;
    used_ = checkpoint.used;
}
