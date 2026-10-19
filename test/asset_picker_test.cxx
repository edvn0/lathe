#include <doctest/doctest.h>

#include <cstddef>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "app/asset_picker.hxx"
#include "assets/asset_registry.hxx"
#include "ui_test_harness.hxx"

namespace {

    using Models = NamedAssetTable<ModelHandle>;

    auto handle_of(std::uint32_t index) -> ModelHandle { return ModelHandle{.index = index, .generation = 1}; }

    // Names a picker has to cope with, whatever the assets are called or how they came about.
    auto hostile_names() -> std::vector<std::string> {
        return {
                "",
                " ",
                "\t",
                "##",
                "##label",
                "###id",
                "%s %d %n",
                "line\nbreak",
                "wide \xE2\x9C\x93 \xF0\x9F\x98\x80 \xE6\x97\xA5\xE6\x9C\xAC",
                std::string(20'000, 'x'),
                std::string{"nul\0inside", 10},
                "cube",
        };
    }

    auto hostile_models() -> Models {
        auto models = Models{};
        auto index = std::uint32_t{1};
        for (auto const &name: hostile_names()) {
            models.register_asset(name, handle_of(index++));
        }
        // Entries with no valid handle, and a registered name shared by nothing else.
        models.register_asset("unloaded", ModelHandle{});
        return models;
    }

    auto draw(Models const &models, std::span<ModelHandle const> used, ModelHandle current) {
        return ui::draw_asset_choices(models.entries(), used, current);
    }

}

TEST_SUITE("ui") {
    TEST_CASE("asset picker survives names that are empty, odd or huge") {
        UiTestHarness harness;
        auto const models = hostile_models();

        for (auto frame = 0; frame < 4; ++frame) {
            harness.popup_frame([&] { CHECK_FALSE(draw(models, {}, ModelHandle{}).has_value()); });
        }

        CHECK(harness.errors.empty());
    }

    TEST_CASE("asset picker survives every entry being unnamed or invalid") {
        UiTestHarness harness;
        auto models = Models{};
        models.register_asset("", handle_of(1));
        models.register_asset("never loaded", ModelHandle{});

        for (auto frame = 0; frame < 3; ++frame) {
            harness.popup_frame([&] { CHECK_FALSE(draw(models, {}, handle_of(1)).has_value()); });
        }

        CHECK(harness.errors.empty());
    }

    TEST_CASE("asset picker survives an empty registry") {
        UiTestHarness harness;
        auto const models = Models{};

        for (auto frame = 0; frame < 3; ++frame) {
            harness.popup_frame([&] { CHECK_FALSE(draw(models, {}, ModelHandle{}).has_value()); });
        }

        CHECK(harness.errors.empty());
    }

    TEST_CASE("asset picker survives thousands of assets") {
        UiTestHarness harness;
        auto models = Models{};
        auto used = std::vector<ModelHandle>{};
        for (auto index = std::uint32_t{1}; index <= 5000; ++index) {
            models.register_asset("model_" + std::to_string(index), handle_of(index));
            if (index % 97 == 0) {
                used.push_back(handle_of(index));
            }
        }

        for (auto frame = 0; frame < 3; ++frame) {
            harness.popup_frame([&] { CHECK_FALSE(draw(models, used, handle_of(97)).has_value()); });
        }

        CHECK(harness.errors.empty());
    }

    TEST_CASE("asset picker lists only part of a long registry until it is filtered") {
        UiTestHarness harness;
        auto models = Models{};
        for (auto index = std::uint32_t{1}; index <= 500; ++index) {
            models.register_asset("model_" + std::to_string(index), handle_of(index));
        }

        auto lines = 0.0F;
        harness.popup_frame([&] {
            std::ignore = draw(models, {}, ModelHandle{});
            // Where the next widget would go, in lines of text: 500 listed models would be far past 60.
            lines = ImGui::GetCursorPosY() / ImGui::GetTextLineHeightWithSpacing();
        });

        CHECK(harness.errors.empty());
        CHECK(lines > 10.0F);
        CHECK(lines < 60.0F);
    }

    TEST_CASE("the same name may appear twice without clashing IDs") {
        UiTestHarness harness;
        // NamedAssetTable refuses duplicates, so a table is built by hand: the picker gets whatever it is given.
        auto const entries = std::vector<Models::Entry>{
                {.name = "twin", .handle = handle_of(1)},
                {.name = "twin", .handle = handle_of(2)},
        };

        for (auto frame = 0; frame < 3; ++frame) {
            harness.popup_frame([&] {
                std::ignore = ui::draw_asset_choices(std::span<Models::Entry const>{entries},
                                                     std::span<ModelHandle const>{}, ModelHandle{});
            });
        }

        CHECK(harness.errors.empty());
    }

    TEST_CASE("the harness reports a mismatch ImGui recovers from") {
        UiTestHarness harness;

        harness.frame([] { ImGui::PushID("never popped"); });

        CHECK_FALSE(harness.errors.empty());
    }
}
