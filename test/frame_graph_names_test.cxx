#include <doctest/doctest.h>

#include <algorithm>
#include <string_view>
#include <vector>

#include "rendering/frame_graph/names.hxx"

using namespace frame_graph;

TEST_SUITE("unit") {
    TEST_CASE("every use has a distinct readable name") {
        auto seen = std::vector<std::string_view>{};
        for (auto value = 0U; value <= static_cast<unsigned>(Use::token_read); ++value) {
            auto const name = use_name(static_cast<Use>(value));
            CHECK(name != "unknown");
            CHECK(std::ranges::find(seen, name) == seen.end());
            seen.push_back(name);
        }
    }

    TEST_CASE("common image layouts are named and the rest fall back") {
        CHECK(layout_name(VK_IMAGE_LAYOUT_UNDEFINED) == "undefined");
        CHECK(layout_name(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) == "shader read-only");
        CHECK(layout_name(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) == "present");
        CHECK(layout_name(VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR) == "other");
    }
}
