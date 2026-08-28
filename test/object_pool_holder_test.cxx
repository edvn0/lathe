#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "object_pool.hxx"

namespace {

    struct LifetimeState {
        static constexpr std::size_t max_resources = 128;

        std::uint32_t total_destructions = 0;

        std::array<std::uint32_t, max_resources> destructions{};
        std::vector<std::uint32_t> destruction_order;

        [[nodiscard]]
        auto destruction_count(std::uint32_t id) const -> std::uint32_t {
            return destructions.at(id);
        }
    };

    /*
     * Models an actual RAII resource such as Image, Buffer, Pipeline, etc.
     *
     * A default-constructed pool slot owns nothing.
     *
     * Moving transfers ownership and leaves the source inert. This is important:
     * ObjectPool::release() moves the payload out of the slot before the returned
     * T is destroyed.
     */
    struct TrackedResource {
        std::shared_ptr<LifetimeState> state;

        std::uint32_t id = 0;
        std::uint32_t value = 0;

        bool owns_resource = false;

        TrackedResource() = default;

        TrackedResource(std::shared_ptr<LifetimeState> s, std::uint32_t i, std::uint32_t v = 0) :
            state(std::move(s)), id(i), value(v), owns_resource(true) {}

        ~TrackedResource() { destroy(); }

        TrackedResource(TrackedResource const &) = delete;
        auto operator=(TrackedResource const &) -> TrackedResource & = delete;

        TrackedResource(TrackedResource &&other) noexcept :
            state(std::move(other.state)), id(std::exchange(other.id, 0)), value(std::exchange(other.value, 0)),
            owns_resource(std::exchange(other.owns_resource, false)) {}

        auto operator=(TrackedResource &&other) noexcept -> TrackedResource & {
            if (this == &other) {
                return *this;
            }

            destroy();

            state = std::move(other.state);
            id = std::exchange(other.id, 0);
            value = std::exchange(other.value, 0);
            owns_resource = std::exchange(other.owns_resource, false);

            return *this;
        }

        auto destroy() noexcept -> void {
            if (!owns_resource) {
                return;
            }

            if (state != nullptr) {
                ++state->total_destructions;
                ++state->destructions.at(id);
                state->destruction_order.push_back(id);
            }

            owns_resource = false;
        }
    };

    using Pool = ObjectPool<TrackedResource>;
    using TestHolder = Pool::HolderT;
    using ResourceHandle = Pool::HandleT;

    [[nodiscard]]
    auto make_state() -> std::shared_ptr<LifetimeState> {
        return std::make_shared<LifetimeState>();
    }

    [[nodiscard]]
    auto acquire_resource(Pool &pool, std::shared_ptr<LifetimeState> const &state, std::uint32_t id,
                          std::uint32_t value = 0) -> TestHolder {
        auto maybe_holder = pool.acquire();

        if (!maybe_holder) {
            std::abort();
        }

        auto holder = std::move(*maybe_holder);

        *holder = TrackedResource{
                state,
                id,
                value,
        };

        return holder;
    }

} // namespace

TEST_SUITE("ObjectPool::Holder") {

    TEST_CASE("default constructed holder is empty") {
        TestHolder holder;

        CHECK_FALSE(holder);
        CHECK(holder.get() == nullptr);
    }

    TEST_CASE("acquire returns a holder when capacity is available") {
        auto pool = Pool::create(1);

        auto holder = pool.acquire();

        REQUIRE(holder.has_value());
    }

    TEST_CASE("acquired holder evaluates true") {
        auto pool = Pool::create(1);

        auto holder = pool.acquire();

        REQUIRE(holder);
        CHECK(static_cast<bool>(*holder));
    }

    TEST_CASE("acquire increases pool size") {
        auto pool = Pool::create(4);

        CHECK(pool.size() == 0);

        auto holder = pool.acquire();

        REQUIRE(holder);
        CHECK(pool.size() == 1);
    }

    TEST_CASE("holder exposes its underlying object") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 1, 42);

        REQUIRE(holder.get() != nullptr);
        CHECK(holder.get()->value == 42);
    }

    TEST_CASE("holder dereference operator accesses underlying object") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 1, 42);

        CHECK((*holder).value == 42);

        (*holder).value = 100;

        CHECK((*holder).value == 100);
    }

    TEST_CASE("holder arrow operator accesses underlying object") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 1, 55);

        CHECK(holder->value == 55);

        holder->value = 77;

        CHECK(holder->value == 77);
    }

    TEST_CASE("holder exposes a valid handle") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 1);

        CHECK(holder.handle().valid());
    }

    TEST_CASE("pool contains holder handle") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 1);

        CHECK(pool.contains(holder.handle()));
    }

    TEST_CASE("pool get and holder get reference the same object") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 1);

        CHECK(pool.get(holder.handle()) == holder.get());
    }

    TEST_CASE("holder destruction releases pool slot") {
        auto pool = Pool::create(1);
        auto state = make_state();

        {
            auto holder = acquire_resource(pool, state, 1);

            CHECK(pool.size() == 1);
        }

        CHECK(pool.size() == 0);
    }

    TEST_CASE("holder destruction destroys payload") {
        auto pool = Pool::create(1);
        auto state = make_state();

        {
            auto holder = acquire_resource(pool, state, 7);

            CHECK(state->destruction_count(7) == 0);
        }

        CHECK(state->destruction_count(7) == 1);
        CHECK(state->total_destructions == 1);
    }

    TEST_CASE("holder destroys payload exactly once") {
        auto pool = Pool::create(1);
        auto state = make_state();

        {
            auto holder = acquire_resource(pool, state, 7);
        }

        CHECK(state->destruction_count(7) == 1);

        // Pool destruction must not destroy the moved-from slot payload again.
    }

    TEST_CASE("reset releases pool slot") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 1);

        CHECK(pool.size() == 1);

        holder.reset();

        CHECK(pool.size() == 0);
    }

    TEST_CASE("reset destroys payload") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 12);

        holder.reset();

        CHECK(state->destruction_count(12) == 1);
    }

    TEST_CASE("reset makes holder empty") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 1);

        holder.reset();

        CHECK_FALSE(holder);
        CHECK(holder.get() == nullptr);
    }

    TEST_CASE("repeated reset is harmless") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 3);

        holder.reset();
        holder.reset();
        holder.reset();

        CHECK(state->destruction_count(3) == 1);
        CHECK(pool.size() == 0);
    }

    TEST_CASE("old handle becomes invalid after reset") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 1);
        auto const handle = holder.handle();

        REQUIRE(pool.contains(handle));

        holder.reset();

        CHECK_FALSE(pool.contains(handle));
        CHECK(pool.get(handle) == nullptr);
    }

    TEST_CASE("holder destructor after reset does not double destroy") {
        auto pool = Pool::create(1);
        auto state = make_state();

        {
            auto holder = acquire_resource(pool, state, 9);

            holder.reset();

            CHECK(state->destruction_count(9) == 1);
        }

        CHECK(state->destruction_count(9) == 1);
    }

    TEST_CASE("moving holder transfers ownership") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto first = acquire_resource(pool, state, 10);

        auto const handle = first.handle();

        Holder second = std::move(first);

        CHECK_FALSE(first);
        CHECK(second);

        CHECK(second.handle() == handle);
        CHECK(pool.contains(handle));

        CHECK(state->destruction_count(10) == 0);
    }

    TEST_CASE("move construction does not destroy payload") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto first = acquire_resource(pool, state, 11);

        {
            auto second = std::move(first);

            CHECK(state->destruction_count(11) == 0);
        }

        CHECK(state->destruction_count(11) == 1);
    }

    TEST_CASE("moved-from holder destructor does nothing") {
        auto pool = Pool::create(1);
        auto state = make_state();

        {
            auto first = acquire_resource(pool, state, 12);

            {
                auto second = std::move(first);

                CHECK_FALSE(first);
                CHECK(state->destruction_count(12) == 0);
            }

            CHECK(state->destruction_count(12) == 1);
        }

        CHECK(state->destruction_count(12) == 1);
    }

    TEST_CASE("move assignment destroys destination resource") {
        auto pool = Pool::create(2);
        auto state = make_state();

        auto destination = acquire_resource(pool, state, 20);
        auto source = acquire_resource(pool, state, 21);

        CHECK(pool.size() == 2);

        destination = std::move(source);

        CHECK(state->destruction_count(20) == 1);
        CHECK(state->destruction_count(21) == 0);

        CHECK(pool.size() == 1);
    }

    TEST_CASE("move assignment transfers source ownership") {
        auto pool = Pool::create(2);
        auto state = make_state();

        auto destination = acquire_resource(pool, state, 20);
        auto source = acquire_resource(pool, state, 21);

        auto const source_handle = source.handle();

        destination = std::move(source);

        CHECK_FALSE(source);
        CHECK(destination);

        CHECK(destination.handle() == source_handle);
        CHECK(pool.contains(source_handle));
    }

    TEST_CASE("move assigned resource is eventually destroyed exactly once") {
        auto pool = Pool::create(2);
        auto state = make_state();

        {
            auto destination = acquire_resource(pool, state, 20);
            auto source = acquire_resource(pool, state, 21);

            destination = std::move(source);

            CHECK(state->destruction_count(20) == 1);
            CHECK(state->destruction_count(21) == 0);
        }

        CHECK(state->destruction_count(20) == 1);
        CHECK(state->destruction_count(21) == 1);
    }

    TEST_CASE("self move assignment retains ownership") {
        auto pool = Pool::create(1);
        auto state = make_state();

        {
            auto holder = acquire_resource(pool, state, 30);

            auto const handle = holder.handle();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
            holder = std::move(holder);
#pragma GCC diagnostic pop

            CHECK(holder);
            CHECK(holder.handle() == handle);
            CHECK(pool.contains(handle));
            CHECK(state->destruction_count(30) == 0);
        }

        CHECK(state->destruction_count(30) == 1);
    }

    TEST_CASE("capacity exhaustion returns nullopt") {
        auto pool = Pool::create(2);
        auto state = make_state();

        auto first = acquire_resource(pool, state, 1);
        auto second = acquire_resource(pool, state, 2);

        CHECK(pool.size() == 2);

        auto third = pool.acquire();

        CHECK_FALSE(third.has_value());
    }

    TEST_CASE("destroying holder makes capacity available again") {
        auto pool = Pool::create(1);
        auto state = make_state();

        {
            auto first = acquire_resource(pool, state, 1);

            CHECK_FALSE(pool.acquire().has_value());
        }

        auto second = pool.acquire();

        CHECK(second.has_value());
    }

    TEST_CASE("reset makes capacity available again") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto first = acquire_resource(pool, state, 1);

        CHECK_FALSE(pool.acquire());

        first.reset();

        CHECK(pool.acquire().has_value());
    }

    TEST_CASE("released slot reuses same index") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto first = acquire_resource(pool, state, 1);
        auto const first_handle = first.handle();

        first.reset();

        auto second = acquire_resource(pool, state, 2);
        auto const second_handle = second.handle();

        CHECK(second_handle.index == first_handle.index);
    }

    TEST_CASE("released slot receives new generation") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto first = acquire_resource(pool, state, 1);
        auto const first_handle = first.handle();

        first.reset();

        auto second = acquire_resource(pool, state, 2);
        auto const second_handle = second.handle();

        CHECK(second_handle.index == first_handle.index);
        CHECK(second_handle.generation != first_handle.generation);
    }

    TEST_CASE("stale handle cannot access reused slot") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto first = acquire_resource(pool, state, 1);
        auto const stale_handle = first.handle();

        first.reset();

        auto second = acquire_resource(pool, state, 2);

        REQUIRE(second.handle().index == stale_handle.index);

        CHECK_FALSE(pool.contains(stale_handle));
        CHECK(pool.get(stale_handle) == nullptr);

        CHECK(pool.contains(second.handle()));
    }

    TEST_CASE("new holder accesses newly installed resource after reuse") {
        auto pool = Pool::create(1);
        auto state = make_state();

        {
            auto first = acquire_resource(pool, state, 1, 100);

            CHECK(first->value == 100);
        }

        auto second = acquire_resource(pool, state, 2, 200);

        CHECK(second->id == 2);
        CHECK(second->value == 200);
    }

    TEST_CASE("repeated slot reuse destroys every resource exactly once") {
        auto pool = Pool::create(1);
        auto state = make_state();

        for (std::uint32_t id = 1; id <= 10; ++id) {
            auto holder = acquire_resource(pool, state, id);

            CHECK(state->destruction_count(id) == 0);
        }

        CHECK(state->total_destructions == 10);

        for (std::uint32_t id = 1; id <= 10; ++id) {
            CHECK(state->destruction_count(id) == 1);
        }
    }

    TEST_CASE("several holders own independent resources") {
        auto pool = Pool::create(4);
        auto state = make_state();

        auto first = acquire_resource(pool, state, 1);
        auto second = acquire_resource(pool, state, 2);
        auto third = acquire_resource(pool, state, 3);

        CHECK(pool.size() == 3);

        CHECK(first->id == 1);
        CHECK(second->id == 2);
        CHECK(third->id == 3);

        CHECK(first.handle() != second.handle());
        CHECK(first.handle() != third.handle());
        CHECK(second.handle() != third.handle());
    }

    TEST_CASE("destroying one holder does not affect other holders") {
        auto pool = Pool::create(3);
        auto state = make_state();

        auto first = acquire_resource(pool, state, 1);
        auto second = acquire_resource(pool, state, 2);
        auto third = acquire_resource(pool, state, 3);

        auto const second_handle = second.handle();
        auto const third_handle = third.handle();

        first.reset();

        CHECK(pool.size() == 2);

        CHECK(state->destruction_count(1) == 1);
        CHECK(state->destruction_count(2) == 0);
        CHECK(state->destruction_count(3) == 0);

        CHECK(pool.contains(second_handle));
        CHECK(pool.contains(third_handle));

        CHECK(second->id == 2);
        CHECK(third->id == 3);
    }

    TEST_CASE("nested holders are destroyed in reverse scope order") {
        auto pool = Pool::create(3);
        auto state = make_state();

        {
            auto first = acquire_resource(pool, state, 1);

            {
                auto second = acquire_resource(pool, state, 2);

                {
                    auto third = acquire_resource(pool, state, 3);
                }
            }
        }

        REQUIRE(state->destruction_order.size() == 3);

        CHECK(state->destruction_order[0] == 3);
        CHECK(state->destruction_order[1] == 2);
        CHECK(state->destruction_order[2] == 1);
    }

    TEST_CASE("holders can be stored in vector") {
        auto pool = Pool::create(16);
        auto state = make_state();

        std::vector<TestHolder> holders;

        for (std::uint32_t id = 1; id <= 10; ++id) {
            holders.push_back(acquire_resource(pool, state, id));
        }

        CHECK(pool.size() == 10);
        CHECK(holders.size() == 10);

        for (std::uint32_t index = 0; index < holders.size(); ++index) {
            CHECK(holders[index]->id == index + 1);
        }

        CHECK(state->total_destructions == 0);
    }

    TEST_CASE("vector reallocations do not prematurely destroy resources") {
        auto pool = Pool::create(32);
        auto state = make_state();

        std::vector<TestHolder> holders;

        // Intentionally force multiple reallocations.
        for (std::uint32_t id = 1; id <= 20; ++id) {
            holders.push_back(acquire_resource(pool, state, id));

            CHECK(state->total_destructions == 0);
        }

        CHECK(pool.size() == 20);
    }

    TEST_CASE("clearing vector of holders destroys all resources") {
        auto pool = Pool::create(16);
        auto state = make_state();

        std::vector<TestHolder> holders;

        for (std::uint32_t id = 1; id <= 10; ++id) {
            holders.push_back(acquire_resource(pool, state, id));
        }

        REQUIRE(pool.size() == 10);

        holders.clear();

        CHECK(pool.size() == 0);
        CHECK(state->total_destructions == 10);

        for (std::uint32_t id = 1; id <= 10; ++id) {
            CHECK(state->destruction_count(id) == 1);
        }
    }

    TEST_CASE("holder can move through optional") {
        auto pool = Pool::create(1);
        auto state = make_state();

        std::optional<TestHolder> optional_holder{
                acquire_resource(pool, state, 50),
        };

        REQUIRE(optional_holder);
        CHECK((*optional_holder)->id == 50);
        CHECK(pool.size() == 1);

        auto holder = std::move(*optional_holder);

        CHECK(holder);
        CHECK(holder->id == 50);

        CHECK_FALSE(static_cast<bool>(*optional_holder));

        CHECK(state->destruction_count(50) == 0);
    }

    TEST_CASE("overwriting holder payload destroys previous payload") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 60);

        REQUIRE(holder->id == 60);

        *holder = TrackedResource{
                state,
                61,
        };

        CHECK(state->destruction_count(60) == 1);
        CHECK(state->destruction_count(61) == 0);

        CHECK(holder->id == 61);
        CHECK(pool.size() == 1);
    }

    TEST_CASE("replacement payload is destroyed when holder dies") {
        auto pool = Pool::create(1);
        auto state = make_state();

        {
            auto holder = acquire_resource(pool, state, 60);

            *holder = TrackedResource{
                    state,
                    61,
            };

            CHECK(state->destruction_count(60) == 1);
            CHECK(state->destruction_count(61) == 0);
        }

        CHECK(state->destruction_count(60) == 1);
        CHECK(state->destruction_count(61) == 1);
    }

    TEST_CASE("detach relinquishes holder ownership") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 70);

        auto const handle = holder.detach();

        CHECK_FALSE(holder);

        CHECK(pool.contains(handle));
        CHECK(pool.size() == 1);

        CHECK(state->destruction_count(70) == 0);
    }

    TEST_CASE("detach returns original handle") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 71);

        auto const expected = holder.handle();
        auto const actual = holder.detach();

        CHECK(actual == expected);
    }

    TEST_CASE("detached holder destructor does not destroy payload") {
        auto pool = Pool::create(1);
        auto state = make_state();

        ResourceHandle detached_handle;

        {
            auto holder = acquire_resource(pool, state, 72);

            detached_handle = holder.detach();

            CHECK(state->destruction_count(72) == 0);
        }

        CHECK(state->destruction_count(72) == 0);
        CHECK(pool.contains(detached_handle));
    }

    TEST_CASE("detached resource can be manually released and destroyed") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 73);

        auto const handle = holder.detach();

        REQUIRE(pool.contains(handle));

        {
            auto resource = pool.release(handle);

            REQUIRE(resource);
            CHECK(resource->id == 73);

            CHECK(state->destruction_count(73) == 0);
        }

        CHECK(state->destruction_count(73) == 1);
        CHECK(pool.size() == 0);
    }

    TEST_CASE("detached slot remains unavailable until explicitly released") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 74);

        auto const handle = holder.detach();

        CHECK_FALSE(pool.acquire());

        {
            auto resource = pool.release(handle);
            REQUIRE(resource);
        }

        CHECK(pool.acquire().has_value());
    }

    TEST_CASE("empty pool cannot acquire holder") {
        auto pool = Pool::create(0);

        CHECK(pool.capacity() == 0);
        CHECK(pool.size() == 0);

        auto holder = pool.acquire();

        CHECK_FALSE(holder);
    }

    TEST_CASE("holder reset followed by reuse keeps size correct") {
        auto pool = Pool::create(1);
        auto state = make_state();

        for (std::uint32_t iteration = 1; iteration <= 50; ++iteration) {
            {
                auto holder = acquire_resource(pool, state, iteration);

                CHECK(pool.size() == 1);
            }

            CHECK(pool.size() == 0);
        }

        CHECK(state->total_destructions == 50);
    }

    TEST_CASE("all capacity is recovered after many holders are destroyed") {
        constexpr std::uint32_t capacity = 32;

        auto pool = Pool::create(capacity);
        auto state = make_state();

        {
            std::vector<TestHolder> holders;

            holders.reserve(capacity);

            for (std::uint32_t id = 1; id <= capacity; ++id) {
                holders.push_back(acquire_resource(pool, state, id));
            }

            CHECK(pool.size() == capacity);
            CHECK_FALSE(pool.acquire());
        }

        CHECK(pool.size() == 0);

        std::vector<TestHolder> holders;

        holders.reserve(capacity);

        for (std::uint32_t id = 1; id <= capacity; ++id) {
            holders.push_back(acquire_resource(pool, state, id));
        }

        CHECK(pool.size() == capacity);
    }

    TEST_CASE("generation changes every time single slot is reused") {
        auto pool = Pool::create(1);
        auto state = make_state();

        std::uint32_t previous_generation = 0;

        for (std::uint32_t iteration = 1; iteration <= 50; ++iteration) {
            auto holder = acquire_resource(pool, state, iteration);

            auto const handle = holder.handle();

            CHECK(handle.index == 0);

            if (previous_generation != 0) {
                CHECK(handle.generation != previous_generation);
            }

            previous_generation = handle.generation;
        }
    }

    TEST_CASE("all stale generations remain invalid after repeated reuse") {
        auto pool = Pool::create(1);
        auto state = make_state();

        std::vector<ResourceHandle> old_handles;

        for (std::uint32_t iteration = 1; iteration <= 20; ++iteration) {
            auto holder = acquire_resource(pool, state, iteration);

            for (auto const old_handle: old_handles) {
                CHECK_FALSE(pool.contains(old_handle));
                CHECK(pool.get(old_handle) == nullptr);
            }

            old_handles.push_back(holder.handle());
        }
    }

    TEST_CASE("const holder can inspect resource") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 80, 1234);

        auto const &const_holder = holder;

        static_assert(std::is_same_v<decltype(const_holder.get()), TrackedResource const *>);

        CHECK(const_holder.get() != nullptr);
        CHECK(const_holder->id == 80);
        CHECK(const_holder->value == 1234);
        CHECK((*const_holder).value == 1234);
    }

    TEST_CASE("holder handle remains stable while owned") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 81);

        auto const original = holder.handle();

        holder->value = 1;
        CHECK(holder.handle() == original);

        holder->value = 2;
        CHECK(holder.handle() == original);

        holder->value = 3;
        CHECK(holder.handle() == original);
    }

    TEST_CASE("mutating payload does not affect generation") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 82);

        auto const generation = holder.handle().generation;

        for (std::uint32_t value = 0; value < 100; ++value) {
            holder->value = value;

            CHECK(holder.handle().generation == generation);
        }
    }

    TEST_CASE("resource destruction happens before slot is reused") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto first = acquire_resource(pool, state, 90);
        auto const index = first.handle().index;

        first.reset();

        CHECK(state->destruction_count(90) == 1);

        auto second = acquire_resource(pool, state, 91);

        CHECK(second.handle().index == index);
        CHECK(state->destruction_count(90) == 1);
        CHECK(state->destruction_count(91) == 0);
    }

    TEST_CASE("raw pool release moves payload without prematurely destroying it") {
        auto pool = Pool::create(1);
        auto state = make_state();

        auto holder = acquire_resource(pool, state, 100);
        auto const handle = holder.detach();

        auto released = pool.release(handle);

        REQUIRE(released);

        CHECK(released->id == 100);

        // Object has been removed from the pool but the returned value still
        // owns the underlying resource.
        CHECK(state->destruction_count(100) == 0);
        CHECK(pool.size() == 0);

        released.reset();

        CHECK(state->destruction_count(100) == 1);
    }

    TEST_CASE("pool slots themselves do not double destroy released resources") {
        auto state = make_state();

        {
            auto pool = Pool::create(1);

            {
                auto holder = acquire_resource(pool, state, 101);
            }

            CHECK(state->destruction_count(101) == 1);
        }

        // Destroying ObjectPool destroys the moved-from T stored inside the
        // slot, but that T no longer owns the resource.
        CHECK(state->destruction_count(101) == 1);
    }

    TEST_CASE("destruction order remains correct through holder moves") {
        auto pool = Pool::create(3);
        auto state = make_state();

        {
            auto first = acquire_resource(pool, state, 1);
            auto second = acquire_resource(pool, state, 2);
            auto third = acquire_resource(pool, state, 3);

            std::vector<TestHolder> holders;

            holders.push_back(std::move(first));
            holders.push_back(std::move(second));
            holders.push_back(std::move(third));

            CHECK(state->total_destructions == 0);

            holders.pop_back();
            CHECK(state->destruction_order == std::vector<std::uint32_t>{3});

            holders.pop_back();
            CHECK(state->destruction_order == std::vector<std::uint32_t>{3, 2});

            holders.pop_back();
            CHECK(state->destruction_order == std::vector<std::uint32_t>{3, 2, 1});
        }

        CHECK(state->total_destructions == 3);
    }
}
