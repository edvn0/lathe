#include <doctest/doctest.h>

#include "rendering/script_storage.hxx"

namespace {
    struct TestScript : IScript {};
} // namespace

TEST_SUITE("unit") {
    TEST_CASE("ScriptStorage: create with capacity < 2 fails with invalid_argument") {
        auto storage = ScriptStorage::create({.capacity = 1});

        REQUIRE_FALSE(storage.has_value());
        CHECK(storage.error().type == ScriptStorageErrorType::invalid_argument);
    }

    TEST_CASE("ScriptStorage: emplace returns a handle shared by every lookup") {
        auto storage = ScriptStorage::create({.capacity = 4});
        REQUIRE(storage.has_value());

        auto handle = storage->emplace<TestScript>();
        REQUIRE(handle.has_value());
        CHECK(handle->valid());

        auto *first = storage->get(*handle);
        auto *second = storage->get(*handle);

        REQUIRE(first != nullptr);
        REQUIRE(second != nullptr);
        CHECK(first == second);
    }

    TEST_CASE("ScriptStorage: two emplace calls resolve to distinct instances") {
        auto storage = ScriptStorage::create({.capacity = 4});
        REQUIRE(storage.has_value());

        auto handle_a = storage->emplace<TestScript>();
        auto handle_b = storage->emplace<TestScript>();

        REQUIRE(handle_a.has_value());
        REQUIRE(handle_b.has_value());

        auto *script_a = storage->get(*handle_a);
        auto *script_b = storage->get(*handle_b);

        REQUIRE(script_a != nullptr);
        REQUIRE(script_b != nullptr);
        CHECK(script_a != script_b);
    }

    TEST_CASE("ScriptStorage: a default-constructed handle resolves to nullptr") {
        auto storage = ScriptStorage::create({.capacity = 4});
        REQUIRE(storage.has_value());

        CHECK(storage->get(ScriptHandle{}) == nullptr);
    }

    TEST_CASE("ScriptStorage: emplace past capacity fails with capacity_exceeded") {
        auto storage = ScriptStorage::create({.capacity = 2});
        REQUIRE(storage.has_value());

        // Slot 0 is permanently reserved, leaving exactly one real slot.
        auto first = storage->emplace<TestScript>();
        REQUIRE(first.has_value());

        auto overflow = storage->emplace<TestScript>();
        REQUIRE_FALSE(overflow.has_value());
        CHECK(overflow.error().type == ScriptStorageErrorType::capacity_exceeded);
    }
}
