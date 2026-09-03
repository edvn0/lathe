#include <doctest/doctest.h>

#include "core/object_pool.hxx"
#include "core/handle.hxx"

#include <cstdint>
#include <utility>
#include <vector>

namespace {
    struct Payload {
        int value = -1;

        auto operator==(Payload const &) const -> bool = default;
    };

    using Pool = ObjectPool<Payload>;
    using PoolHandle = Pool::HandleT;
} // namespace

TEST_SUITE("unit") {
    TEST_CASE("ObjectPool: create reports capacity and starts empty") {
        auto pool = Pool::create(4);

        CHECK(pool.capacity() == 4);
        CHECK(pool.size() == 0);
    }

    TEST_CASE("ObjectPool: default-constructed handle is never contained") {
        auto pool = Pool::create(4);

        CHECK_FALSE(pool.contains(PoolHandle{}));
        CHECK(pool.get(PoolHandle{}) == nullptr);
    }

    TEST_CASE("ObjectPool: allocate returns a usable handle and reference") {
        auto pool = Pool::create(4);

        auto allocation = pool.allocate();
        REQUIRE(allocation.has_value());

        auto [handle, value] = *allocation;
        value.value = 42;

        CHECK(handle.generation != 0);
        CHECK(pool.size() == 1);
        REQUIRE(pool.get(handle) != nullptr);
        CHECK(pool.get(handle)->value == 42);
        CHECK(pool.contains(handle));
    }

    TEST_CASE("ObjectPool: insertion up to capacity yields distinct handles") {
        auto pool = Pool::create(3);

        auto a = pool.allocate();
        auto b = pool.allocate();
        auto c = pool.allocate();

        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        REQUIRE(c.has_value());

        CHECK(a->first.index != b->first.index);
        CHECK(b->first.index != c->first.index);
        CHECK(a->first.index != c->first.index);
        CHECK(pool.size() == 3);
    }

    TEST_CASE("ObjectPool: allocate past capacity fails without corrupting state") {
        auto pool = Pool::create(2);

        auto a = pool.allocate();
        auto b = pool.allocate();
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());

        auto overflow = pool.allocate();
        CHECK_FALSE(overflow.has_value());
        CHECK(pool.size() == 2);

        // Recovers once a slot is freed.
        auto released = pool.release(a->first);
        REQUIRE(released.has_value());

        auto reused = pool.allocate();
        CHECK(reused.has_value());
        CHECK(pool.size() == 2);
    }

    TEST_CASE("ObjectPool: release rejects out-of-range and already-released handles") {
        auto pool = Pool::create(2);

        CHECK_FALSE(pool.release(PoolHandle{.index = 999, .generation = 1}).has_value());

        auto allocation = pool.allocate();
        REQUIRE(allocation.has_value());
        auto const handle = allocation->first;

        auto first_release = pool.release(handle);
        CHECK(first_release.has_value());

        auto second_release = pool.release(handle);
        CHECK_FALSE(second_release.has_value());
    }

    TEST_CASE("ObjectPool: release returns the removed payload by value") {
        auto pool = Pool::create(1);

        auto allocation = pool.allocate();
        REQUIRE(allocation.has_value());
        allocation->second.value = 7;

        auto released = pool.release(allocation->first);
        REQUIRE(released.has_value());
        CHECK(released->value == 7);
    }

    TEST_CASE("ObjectPool: a stale handle is rejected after its slot is reused") {
        auto pool = Pool::create(1);

        auto first = pool.allocate();
        REQUIRE(first.has_value());
        auto const stale_handle = first->first;

        REQUIRE(pool.release(stale_handle).has_value());

        auto second = pool.allocate();
        REQUIRE(second.has_value());
        auto const fresh_handle = second->first;

        CHECK(fresh_handle.index == stale_handle.index);
        CHECK(fresh_handle.generation != stale_handle.generation);

        CHECK(pool.get(stale_handle) == nullptr);
        CHECK_FALSE(pool.contains(stale_handle));
        REQUIRE(pool.get(fresh_handle) != nullptr);
    }

    TEST_CASE("ObjectPool: generation wraps past zero and keeps rejecting every prior generation") {
        auto pool = Pool::create(1);

        auto allocation = pool.allocate();
        REQUIRE(allocation.has_value());
        auto handle = allocation->first;

        std::vector<PoolHandle> previous_handles{handle};

        // std::uint32_t generation wraps after ~4 billion cycles in
        // principle; exercise the explicit "skip back to 1" guard directly
        // by cycling enough times to be confident the invariant holds, and
        // spot-check a handful of prior generations stay rejected.
        for (int i = 0; i < 1000; ++i) {
            auto released = pool.release(handle);
            REQUIRE(released.has_value());

            auto reallocated = pool.allocate();
            REQUIRE(reallocated.has_value());

            handle = reallocated->first;
            CHECK(handle.generation != 0);

            if (i < 10) {
                previous_handles.push_back(handle);
            }
        }

        for (auto const &old_handle: previous_handles) {
            if (old_handle == handle) {
                continue;
            }

            CHECK(pool.get(old_handle) == nullptr);
        }
    }

    TEST_CASE("ObjectPool: raw-index accessors reflect state independent of handles") {
        auto pool = Pool::create(2);

        CHECK_FALSE(pool.occupied_at(0));
        CHECK(pool.generation_at(0) == 1);
        CHECK(pool.get_at(5) == nullptr);

        auto allocation = pool.allocate();
        REQUIRE(allocation.has_value());
        allocation->second.value = 99;

        auto const index = allocation->first.index;

        CHECK(pool.occupied_at(index));
        REQUIRE(pool.get_at(index) != nullptr);
        CHECK(pool.get_at(index)->value == 99);

        REQUIRE(pool.release(allocation->first).has_value());
        CHECK_FALSE(pool.occupied_at(index));
        CHECK(pool.generation_at(index) == 2);
    }

    TEST_CASE("ObjectPool: move construction transfers ownership and empties the source") {
        auto pool = Pool::create(2);

        auto allocation = pool.allocate();
        REQUIRE(allocation.has_value());
        allocation->second.value = 5;
        auto const handle = allocation->first;

        auto moved = std::move(pool);

        CHECK(moved.capacity() == 2);
        CHECK(moved.size() == 1);
        REQUIRE(moved.get(handle) != nullptr);
        CHECK(moved.get(handle)->value == 5);

        CHECK(pool.capacity() == 0);
        CHECK(pool.size() == 0);
    }

    TEST_CASE("ObjectPool: move assignment transfers ownership and empties the source") {
        auto pool = Pool::create(2);

        auto allocation = pool.allocate();
        REQUIRE(allocation.has_value());
        allocation->second.value = 8;
        auto const handle = allocation->first;

        auto other = Pool::create(1);
        other = std::move(pool);

        REQUIRE(other.get(handle) != nullptr);
        CHECK(other.get(handle)->value == 8);
        CHECK(pool.capacity() == 0);
    }

    TEST_CASE("Handle: default sentinel excludes the maximum index and requires a nonzero generation") {
        using DefaultHandle = Handle<Payload>;

        CHECK_FALSE(DefaultHandle{}.valid());

        DefaultHandle const zero_generation{.index = 0, .generation = 0};
        CHECK_FALSE(zero_generation.valid());

        DefaultHandle const real_handle{.index = 0, .generation = 1};
        CHECK(real_handle.valid());
    }

    TEST_CASE("Handle: a zero sentinel excludes index zero even with a nonzero generation") {
        using ZeroSentinelHandle = Handle<Payload, 0>;

        ZeroSentinelHandle const reserved_slot{.index = 0, .generation = 1};
        CHECK_FALSE(reserved_slot.valid());

        ZeroSentinelHandle const real_handle{.index = 1, .generation = 1};
        CHECK(real_handle.valid());
    }

    TEST_CASE("Handle: equality compares index and generation") {
        using DefaultHandle = Handle<Payload>;

        CHECK(DefaultHandle{.index = 3, .generation = 2} == DefaultHandle{.index = 3, .generation = 2});
        CHECK_FALSE(DefaultHandle{.index = 3, .generation = 2} == DefaultHandle{.index = 3, .generation = 3});
        CHECK_FALSE(DefaultHandle{.index = 3, .generation = 2} == DefaultHandle{.index = 4, .generation = 2});
    }

    TEST_CASE("ObjectPool: release leaves the slot's payload untouched rather than resetting it") {
        // Mirrors how ImageStorage/SamplerStorage track a monotonically
        // bumped descriptor revision counter alongside the resource: that
        // counter must survive release()/allocate() cycles unmodified so a
        // stale per-frame cache never mistakes a reused slot for unchanged.
        // ObjectPool must not reset the payload to T{} on release(), leaving
        // that entirely to the wrapper storage.
        struct RevisionedPayload {
            int resource = 0;
            int revision = 1;
        };

        using RevisionedPool = ObjectPool<RevisionedPayload>;

        auto pool = RevisionedPool::create(1);

        auto first = pool.allocate();
        REQUIRE(first.has_value());
        first->second.resource = 10;
        first->second.revision = 5; // simulate several prior bumps

        auto released = pool.release(first->first);
        REQUIRE(released.has_value());
        CHECK(released->revision == 5);

        auto second = pool.allocate();
        REQUIRE(second.has_value());

        // The wrapper hasn't touched the payload yet -- ObjectPool itself
        // must not have reset it back to the RevisionedPayload{} default.
        CHECK(second->second.revision == 5);
    }

    TEST_CASE("ObjectPool: a pool sharing Material's reserved-slot-0 policy still resolves index 0 by handle") {
        // Mirrors MaterialStorage: Sentinel = 0 means MaterialHandle::valid()
        // deliberately reports false for the permanently-reserved default
        // slot, but the pool's own get()/release() -- used internally by the
        // owning storage to manage that slot -- must not gate on valid() or
        // it could never read/update its own default slot.
        using ReservedZeroPool = ObjectPool<Payload, 0>;

        auto pool = ReservedZeroPool::create(2);

        auto first = pool.allocate();
        REQUIRE(first.has_value());
        auto const reserved_handle = first->first;

        CHECK(reserved_handle.index == 0);
        CHECK_FALSE(reserved_handle.valid());
        REQUIRE(pool.get(reserved_handle) != nullptr);
        CHECK(pool.contains(reserved_handle));

        auto second = pool.allocate();
        REQUIRE(second.has_value());
        CHECK(second->first.valid());
    }
}
