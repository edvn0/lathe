#include <doctest/doctest.h>

#include "scene/components.hxx"

TEST_SUITE("unit") {
    TEST_CASE("InstancedModel: every component starts with its own revision") {
        Components::InstancedModel const first{};
        Components::InstancedModel const second{.transforms = {glm::mat4{1.0F}}};

        CHECK(first.revision != 0);
        CHECK(second.revision != 0);
        CHECK(first.revision != second.revision);
    }

    TEST_CASE("InstancedModel: a copy shares the revision until one of them is touched") {
        Components::InstancedModel original{.transforms = {glm::mat4{1.0F}, glm::mat4{2.0F}}};
        auto copy = original;
        CHECK(copy.revision == original.revision);

        auto const before = original.revision;
        original.transforms.pop_back();
        original.touch();

        CHECK(original.revision != before);
        CHECK(copy.revision == before);
    }
}
