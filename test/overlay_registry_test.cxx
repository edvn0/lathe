#include <doctest/doctest.h>

#include "rendering/overlay.hxx"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {
    auto make_desc(std::string name, OverlayStage stage = OverlayStage::scene, std::int32_t order = 0) -> OverlayDesc {
        return OverlayDesc{
                .name = std::move(name),
                .stage = stage,
                .order = order,
                .prepare = {},
                .record = [](OverlayRecordContext const &) {},
        };
    }

    auto names(std::span<OverlayRegistry::Entry> entries) -> std::vector<std::string> {
        std::vector<std::string> result;

        for (auto const &entry: entries) {
            result.push_back(entry.desc.name);
        }

        return result;
    }
}

TEST_SUITE("unit") {
    TEST_CASE("OverlayRegistry: rejects invalid descriptions") {
        OverlayRegistry registry;

        auto unnamed = registry.add(make_desc(""));
        REQUIRE_FALSE(unnamed.has_value());
        CHECK(unnamed.error() == OverlayRegistryError::empty_name);

        auto no_record = make_desc("no record");
        no_record.record = {};
        auto missing = registry.add(std::move(no_record));
        REQUIRE_FALSE(missing.has_value());
        CHECK(missing.error() == OverlayRegistryError::missing_record_callback);

        auto bad_stage = registry.add(make_desc("bad stage", OverlayStage::count));
        REQUIRE_FALSE(bad_stage.has_value());
        CHECK(bad_stage.error() == OverlayRegistryError::invalid_stage);

        CHECK(registry.size() == 0);
    }

    TEST_CASE("OverlayRegistry: stages are ordered by order, then registration") {
        OverlayRegistry registry;

        auto late = registry.add(make_desc("late", OverlayStage::scene, 10));
        auto ui = registry.add(make_desc("ui", OverlayStage::ui, -50));
        auto first_tie = registry.add(make_desc("first tie", OverlayStage::scene, 0));
        auto early = registry.add(make_desc("early", OverlayStage::scene, -100));
        auto second_tie = registry.add(make_desc("second tie", OverlayStage::scene, 0));

        REQUIRE((late && ui && first_tie && early && second_tie));

        CHECK(names(registry.stage(OverlayStage::scene)) ==
              std::vector<std::string>{"early", "first tie", "second tie", "late"});
        CHECK(names(registry.stage(OverlayStage::ui)) == std::vector<std::string>{"ui"});

        CHECK(names(registry.all()) == std::vector<std::string>{"early", "first tie", "second tie", "late", "ui"});
    }

    TEST_CASE("OverlayRegistry: destroying the registration unregisters") {
        OverlayRegistry registry;

        {
            auto registration = registry.add(make_desc("scoped"));
            REQUIRE(registration.has_value());
            CHECK(registration->valid());
            CHECK(registry.size() == 1);
        }

        CHECK(registry.size() == 0);
    }

    TEST_CASE("OverlayRegistry: moved-from registration no longer owns the overlay") {
        OverlayRegistry registry;

        auto registration = registry.add(make_desc("moved"));
        REQUIRE(registration.has_value());

        OverlayRegistration owner = std::move(*registration);
        CHECK_FALSE(registration->valid());
        CHECK(owner.valid());

        registration->reset();
        CHECK(registry.size() == 1);

        owner.reset();
        CHECK(registry.size() == 0);
    }

    TEST_CASE("OverlayRegistry: timing slots are unique, capped and reused") {
        OverlayRegistry registry;
        std::vector<OverlayRegistration> registrations;

        for (std::uint32_t i = 0; i < OverlayRegistry::max_overlays; ++i) {
            auto registration = registry.add(make_desc("overlay " + std::to_string(i)));
            REQUIRE(registration.has_value());
            registrations.push_back(std::move(*registration));
        }

        std::uint32_t seen_slots = 0;
        for (auto const &entry: registry.all()) {
            CHECK(entry.slot < OverlayRegistry::max_overlays);
            CHECK((seen_slots & (1U << entry.slot)) == 0);
            seen_slots |= 1U << entry.slot;
        }

        auto overflow = registry.add(make_desc("one too many"));
        REQUIRE_FALSE(overflow.has_value());
        CHECK(overflow.error() == OverlayRegistryError::capacity_exceeded);

        auto const freed_slot = registry.stage(OverlayStage::scene)[3].slot;
        registrations[3].reset();

        auto replacement = registry.add(make_desc("replacement"));
        REQUIRE(replacement.has_value());

        for (auto const &entry: registry.all()) {
            if (entry.desc.name == "replacement") {
                CHECK(entry.slot == freed_slot);
            }
        }
    }

    TEST_CASE("OverlayRegistry: changes during iteration apply when it ends") {
        OverlayRegistry registry;

        auto kept = registry.add(make_desc("kept"));
        auto removed = registry.add(make_desc("removed"));
        REQUIRE((kept && removed));

        std::optional<OverlayRegistration> added;

        {
            auto const guard = registry.iterate();
            auto const before = registry.stage(OverlayStage::scene);

            removed->reset();

            auto registration = registry.add(make_desc("added"));
            REQUIRE(registration.has_value());
            added = std::move(*registration);

            CHECK(names(registry.stage(OverlayStage::scene)) == std::vector<std::string>{"kept", "removed"});
            CHECK(before.size() == 2);
        }

        CHECK(names(registry.stage(OverlayStage::scene)) == std::vector<std::string>{"kept", "added"});
    }

    TEST_CASE("OverlayRegistry: an overlay added and removed within one iteration never appears") {
        OverlayRegistry registry;

        {
            auto const guard = registry.iterate();

            auto registration = registry.add(make_desc("transient"));
            REQUIRE(registration.has_value());
            registration->reset();
        }

        CHECK(registry.size() == 0);

        std::vector<OverlayRegistration> registrations;
        for (std::uint32_t i = 0; i < OverlayRegistry::max_overlays; ++i) {
            auto registration = registry.add(make_desc("overlay " + std::to_string(i)));
            REQUIRE(registration.has_value());
            registrations.push_back(std::move(*registration));
        }
    }
}
