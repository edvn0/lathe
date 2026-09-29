#include <doctest/doctest.h>

#include <cstdint>
#include <utility>

#include "assets/model_storage.hxx"

namespace {

    [[nodiscard]] auto make_storage() -> ModelStorage {
        auto storage = ModelStorage::create(ModelStorageCreateInfo{.capacity = 8});
        REQUIRE(storage.has_value());
        return std::move(*storage);
    }

    [[nodiscard]] auto make_slot_data(std::uint32_t mesh_index) -> ModelSlotData {
        ModelSlotData data;
        data.draws.push_back(ModelDraw{.mesh = MeshHandle{.index = mesh_index, .generation = 1}});
        return data;
    }

} // namespace

TEST_CASE("A pending model borrows its fallback's meshes and holds a reference on it") {
    auto storage = make_storage();

    auto const fallback = storage.create_model(make_slot_data(3));
    REQUIRE(fallback.has_value());

    auto const pending = storage.create_pending_model(*fallback);
    REQUIRE(pending.has_value());

    auto const *pending_slot = storage.get(*pending);
    REQUIRE(pending_slot != nullptr);
    CHECK(pending_slot->borrowed_from == *fallback);
    CHECK(pending_slot->ref_count == 1);
    REQUIRE(pending_slot->draws.size() == 1);
    CHECK(pending_slot->draws.front().mesh == storage.get(*fallback)->draws.front().mesh);

    CHECK(storage.get(*fallback)->ref_count == 2);
}

TEST_CASE("Upgrading a pending model keeps its references and stops borrowing") {
    auto storage = make_storage();

    auto const fallback = storage.create_model(make_slot_data(3));
    REQUIRE(fallback.has_value());
    auto const pending = storage.create_pending_model(*fallback);
    REQUIRE(pending.has_value());

    // A second caller retained the pending handle before it installed.
    ++storage.get(*pending)->ref_count;

    auto const upgraded = storage.upgrade_pending_model(*pending, make_slot_data(5));
    REQUIRE(upgraded.has_value());
    CHECK(*upgraded == *pending);

    auto const *slot = storage.get(*pending);
    REQUIRE(slot != nullptr);
    CHECK_FALSE(slot->borrowed_from.valid());
    CHECK(slot->ref_count == 2);
    CHECK(slot->draws.front().mesh == (MeshHandle{.index = 5, .generation = 1}));
}

TEST_CASE("A copy of a pending slot's data owns its meshes") {
    auto storage = make_storage();

    auto const fallback = storage.create_model(make_slot_data(3));
    REQUIRE(fallback.has_value());
    auto const pending = storage.create_pending_model(*fallback);
    REQUIRE(pending.has_value());

    auto const copy = storage.create_model(*storage.get(*pending));
    REQUIRE(copy.has_value());
    CHECK_FALSE(storage.get(*copy)->borrowed_from.valid());
    CHECK(storage.get(*copy)->ref_count == 1);
}

TEST_CASE("A pending model needs a live fallback") {
    auto storage = make_storage();

    CHECK_FALSE(storage.create_pending_model(ModelHandle{}).has_value());
}
