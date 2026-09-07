#include <doctest/doctest.h>

#include "assets/asset_registry.hxx"

TEST_SUITE("unit") {
    TEST_CASE("NamedAssetTable: register then find/name_of round-trip") {
        NamedAssetTable<ModelHandle> table;
        ModelHandle const handle{.index = 1, .generation = 1};

        CHECK(table.register_asset("cube", handle));
        CHECK(table.find("cube") == handle);
        CHECK(table.name_of(handle) == "cube");
    }

    TEST_CASE("NamedAssetTable: unknown name resolves to a default (invalid) handle") {
        NamedAssetTable<ModelHandle> table;

        CHECK_FALSE(table.find("missing").valid());
    }

    TEST_CASE("NamedAssetTable: unregistered handle has no name") {
        NamedAssetTable<ModelHandle> table;
        ModelHandle const handle{.index = 1, .generation = 1};

        CHECK(table.name_of(handle).empty());
    }

    TEST_CASE("NamedAssetTable: a name collision is rejected, original registration stands") {
        NamedAssetTable<ModelHandle> table;
        ModelHandle const first{.index = 1, .generation = 1};
        ModelHandle const second{.index = 2, .generation = 1};

        CHECK(table.register_asset("cube", first));
        CHECK_FALSE(table.register_asset("cube", second));
        CHECK(table.find("cube") == first);
        CHECK(table.name_of(second).empty());
    }

    TEST_CASE("NamedAssetTable: unregister removes the entry") {
        NamedAssetTable<ModelHandle> table;
        ModelHandle const handle{.index = 1, .generation = 1};

        REQUIRE(table.register_asset("cube", handle));
        table.unregister(handle);

        CHECK_FALSE(table.find("cube").valid());
        CHECK(table.name_of(handle).empty());
        CHECK(table.entries().empty());
    }

    TEST_CASE("NamedAssetTable: unregistering an unknown handle is a no-op") {
        NamedAssetTable<ModelHandle> table;
        ModelHandle const handle{.index = 1, .generation = 1};

        REQUIRE(table.register_asset("cube", handle));
        table.unregister(ModelHandle{.index = 99, .generation = 1});

        CHECK(table.find("cube") == handle);
    }

    TEST_CASE("NamedAssetTable: entries() stays sorted by name regardless of insertion order") {
        NamedAssetTable<ModelHandle> table;

        REQUIRE(table.register_asset("tree", ModelHandle{.index = 1, .generation = 1}));
        REQUIRE(table.register_asset("cube", ModelHandle{.index = 2, .generation = 1}));
        REQUIRE(table.register_asset("house", ModelHandle{.index = 3, .generation = 1}));

        auto const entries = table.entries();
        REQUIRE(entries.size() == 3);
        CHECK(entries[0].name == "cube");
        CHECK(entries[1].name == "house");
        CHECK(entries[2].name == "tree");
    }

    TEST_CASE("AssetRegistry: each asset kind's names are independent") {
        AssetRegistry registry;
        ModelHandle const model{.index = 1, .generation = 1};
        MaterialHandle const material{.index = 1, .generation = 1};

        CHECK(registry.models().register_asset("grass", model));
        // Same name, different kind -- not a collision, distinct tables.
        CHECK(registry.materials().register_asset("grass", material));

        CHECK(registry.models().find("grass") == model);
        CHECK(registry.materials().find("grass") == material);
        CHECK_FALSE(registry.scripts().find("grass").valid());
    }
}
