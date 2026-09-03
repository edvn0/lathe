#include <doctest/doctest.h>

#include "assets/geometry_allocator.hxx"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

//
// BumpAllocator/FreeListAllocator operate purely on offsets (VkDeviceSize),
// with no Vulkan device or buffer involved, so they're unit-testable in
// isolation. This suite exists to prove GeometryArenaT<BumpAllocator>'s
// behavior is byte-identical to the pre-template GeometryArena (BumpAllocator
// tests), and that FreeListAllocator is safe to substitute in later
// (round-trip / coalescing / alignment tests) -- see
// docs/engine_review_followups.md and the terrain streaming plan.
//

TEST_SUITE("unit") {
    TEST_CASE("BumpAllocator allocates with alignment and never reuses an offset") {
        BumpAllocator allocator;
        allocator.reset(1024);

        auto const first = allocator.allocate(10, 4);
        REQUIRE(first.has_value());
        CHECK(first->offset == 0);
        CHECK(first->size == 10);

        auto const second = allocator.allocate(10, 16);
        REQUIRE(second.has_value());
        CHECK(second->offset == 16); // aligned up from 10
        CHECK(allocator.used_size() == 26);
    }

    TEST_CASE("BumpAllocator rejects invalid alignment and zero-size allocations") {
        BumpAllocator allocator;
        allocator.reset(1024);

        CHECK(allocator.allocate(0, 4).error().type == GeometryArenaErrorType::invalid_argument);
        CHECK(allocator.allocate(10, 0).error().type == GeometryArenaErrorType::invalid_argument);
        CHECK(allocator.allocate(10, 3).error().type == GeometryArenaErrorType::invalid_argument); // not power of two
    }

    TEST_CASE("BumpAllocator reports out_of_memory at the capacity boundary") {
        BumpAllocator allocator;
        allocator.reset(16);

        auto const first = allocator.allocate(16, 4);
        REQUIRE(first.has_value());

        auto const second = allocator.allocate(1, 4);
        REQUIRE_FALSE(second.has_value());
        CHECK(second.error().type == GeometryArenaErrorType::out_of_memory);
    }

    TEST_CASE("BumpAllocator checkpoint/rollback undoes an allocation") {
        BumpAllocator allocator;
        allocator.reset(1024);

        auto const checkpoint = allocator.checkpoint();
        auto const allocation = allocator.allocate(100, 4);
        REQUIRE(allocation.has_value());
        CHECK(allocator.used_size() == 100);

        allocator.rollback(checkpoint);
        CHECK(allocator.used_size() == 0);

        // The reclaimed space -- including alignment padding -- is reusable.
        auto const reused = allocator.allocate(1024, 4);
        REQUIRE(reused.has_value());
        CHECK(reused->offset == 0);
    }

    TEST_CASE("BumpAllocator::deallocate is a no-op") {
        BumpAllocator allocator;
        allocator.reset(16);

        auto const allocation = allocator.allocate(16, 4);
        REQUIRE(allocation.has_value());

        allocator.deallocate(*allocation);
        CHECK(allocator.used_size() == 16); // still consumed -- bump never frees

        auto const second = allocator.allocate(1, 4);
        CHECK_FALSE(second.has_value());
    }

    TEST_CASE("FreeListAllocator round-trips a single allocation") {
        FreeListAllocator allocator;
        allocator.reset(1024);

        auto const allocation = allocator.allocate(100, 4);
        REQUIRE(allocation.has_value());
        CHECK(allocator.used_size() == 100);

        allocator.deallocate(*allocation);
        CHECK(allocator.used_size() == 0);

        // Freed space is immediately reusable, and coalesces back to the
        // full original capacity (verified indirectly: an allocation the
        // size of the whole arena now succeeds).
        auto const reused = allocator.allocate(1024, 4);
        REQUIRE(reused.has_value());
        CHECK(reused->offset == 0);
    }

    TEST_CASE("FreeListAllocator coalesces adjacent free ranges regardless of free order") {
        FreeListAllocator allocator;
        allocator.reset(300);

        auto const first = allocator.allocate(100, 4);
        auto const second = allocator.allocate(100, 4);
        auto const third = allocator.allocate(100, 4);
        REQUIRE(first.has_value());
        REQUIRE(second.has_value());
        REQUIRE(third.has_value());

        // Free the middle, then the ends -- coalescing must merge all three
        // back into one contiguous free range no matter the free order.
        allocator.deallocate(*second);
        allocator.deallocate(*first);
        allocator.deallocate(*third);

        CHECK(allocator.used_size() == 0);

        auto const whole = allocator.allocate(300, 4);
        REQUIRE(whole.has_value());
        CHECK(whole->offset == 0);
    }

    TEST_CASE("FreeListAllocator honors mixed alignments without overlap") {
        FreeListAllocator allocator;
        allocator.reset(4096);

        auto const a = allocator.allocate(20, 4);  // e.g. CompressedModelVertex-ish
        auto const b = allocator.allocate(3, 2);   // uint16 indices
        auto const c = allocator.allocate(104, 4); // uint32 indices

        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        REQUIRE(c.has_value());

        CHECK(a->offset % 4 == 0);
        CHECK(b->offset % 2 == 0);
        CHECK(c->offset % 4 == 0);

        std::vector<GeometrySlice> slices{*a, *b, *c};
        std::ranges::sort(slices, {}, [](GeometrySlice const &slice) { return slice.offset; });

        for (std::size_t i = 1; i < slices.size(); ++i) {
            CHECK(slices[i - 1].offset + slices[i - 1].size <= slices[i].offset);
        }
    }

    TEST_CASE("FreeListAllocator reports out_of_memory when no free range fits") {
        FreeListAllocator allocator;
        allocator.reset(16);

        auto const first = allocator.allocate(16, 4);
        REQUIRE(first.has_value());

        auto const second = allocator.allocate(1, 4);
        REQUIRE_FALSE(second.has_value());
        CHECK(second.error().type == GeometryArenaErrorType::out_of_memory);
    }

    TEST_CASE("FreeListAllocator checkpoint/rollback undoes allocations made since the checkpoint") {
        FreeListAllocator allocator;
        allocator.reset(1024);

        auto const first = allocator.allocate(100, 4);
        REQUIRE(first.has_value());

        auto const checkpoint = allocator.checkpoint();

        auto const second = allocator.allocate(200, 4);
        REQUIRE(second.has_value());
        CHECK(allocator.used_size() == 300);

        allocator.rollback(checkpoint);
        CHECK(allocator.used_size() == 100);

        // The rolled-back range is reusable again.
        auto const third = allocator.allocate(200, 4);
        REQUIRE(third.has_value());
        CHECK(third->offset == second->offset);
    }

    TEST_CASE("FreeListAllocator survives a randomized alloc/free soak with no overlap and full reclaim") {
        constexpr VkDeviceSize capacity = 1U << 16U;

        FreeListAllocator allocator;
        allocator.reset(capacity);

        std::mt19937 rng{1234U};
        std::uniform_int_distribution<int> size_dist{1, 256};
        std::uniform_int_distribution<int> alignment_dist{0, 3}; // 4, 8, 16, 32
        std::vector<GeometrySlice> live;

        for (int iteration = 0; iteration < 5000; ++iteration) {
            if (!live.empty() && (live.size() >= 64 || rng() % 2 == 0)) {

                std::uniform_int_distribution<std::size_t> pick{0, live.size() - 1};
                auto const index = pick(rng);
                allocator.deallocate(live[index]);
                live.erase(live.begin() + static_cast<std::ptrdiff_t>(index));
                continue;
            }

            auto const alignment = VkDeviceSize{4} << alignment_dist(rng);
            auto const allocation = allocator.allocate(static_cast<VkDeviceSize>(size_dist(rng)), alignment);

            if (!allocation) {
                CHECK(allocation.error().type == GeometryArenaErrorType::out_of_memory);
                continue;
            }

            CHECK(allocation->offset % alignment == 0);
            live.push_back(*allocation);
        }

        // No two live slices may overlap.
        std::ranges::sort(live, {}, [](GeometrySlice const &slice) { return slice.offset; });
        for (std::size_t i = 1; i < live.size(); ++i) {
            CHECK(live[i - 1].offset + live[i - 1].size <= live[i].offset);
        }

        VkDeviceSize expected_used = 0;
        for (auto const &slice: live) {
            expected_used += slice.size;
        }
        CHECK(allocator.used_size() == expected_used);

        for (auto const &slice: live) {
            allocator.deallocate(slice);
        }

        CHECK(allocator.used_size() == 0);

        auto const whole = allocator.allocate(capacity, 4);
        REQUIRE(whole.has_value());
        CHECK(whole->offset == 0);
    }
}
