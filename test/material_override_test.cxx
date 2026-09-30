#include <doctest/doctest.h>

#include "scene/components.hxx"

TEST_SUITE("unit") {
    TEST_CASE("MaterialOverride: a default override is empty and replaces nothing") {
        Components::MaterialOverride const material_override{};
        MaterialHandle const source{.index = 3, .generation = 1};

        CHECK(material_override.empty());
        CHECK_FALSE(material_override.replacement_for(source).valid());
    }

    TEST_CASE("MaterialOverride: the whole-model material replaces every source") {
        MaterialHandle const whole{.index = 5, .generation = 1};
        Components::MaterialOverride const material_override{.material = whole};

        CHECK_FALSE(material_override.empty());
        CHECK(material_override.replacement_for(MaterialHandle{.index = 3, .generation = 1}) == whole);
        CHECK(material_override.replacement_for(MaterialHandle{.index = 4, .generation = 2}) == whole);
    }

    TEST_CASE("MaterialOverride: a slot override wins over the whole-model material for its source only") {
        MaterialHandle const whole{.index = 5, .generation = 1};
        MaterialHandle const bark{.index = 3, .generation = 1};
        MaterialHandle const leaves{.index = 4, .generation = 1};
        MaterialHandle const autumn_leaves{.index = 6, .generation = 1};

        Components::MaterialOverride const material_override{
                .material = whole,
                .slots = {MaterialSlotOverride{.source = leaves, .material = autumn_leaves}},
        };

        CHECK(material_override.replacement_for(leaves) == autumn_leaves);
        CHECK(material_override.replacement_for(bark) == whole);
    }

    TEST_CASE("MaterialOverride: slot sources match on generation too") {
        MaterialHandle const leaves{.index = 4, .generation = 1};
        MaterialHandle const reused_slot{.index = 4, .generation = 2};
        MaterialHandle const autumn_leaves{.index = 6, .generation = 1};

        Components::MaterialOverride const material_override{
                .slots = {MaterialSlotOverride{.source = leaves, .material = autumn_leaves}},
        };

        CHECK_FALSE(material_override.empty());
        CHECK_FALSE(material_override.replacement_for(reused_slot).valid());
    }
}
