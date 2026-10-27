#include <doctest/doctest.h>

#include <algorithm>
#include <vector>

#include "rendering/blend_order.hxx"

namespace {

    auto sorted(std::vector<blend_order::Entry> entries) -> std::vector<blend_order::Entry> {
        std::ranges::sort(entries, blend_order::draws_before);
        return entries;
    }

}

TEST_CASE("blended batches are drawn back to front") {
    auto const order = sorted({
            {.camera_distance = 3.0F, .bounds_radius = 1.0F, .stable_id = {1, 0, 0, 0}},
            {.camera_distance = 9.0F, .bounds_radius = 1.0F, .stable_id = {2, 0, 0, 0}},
            {.camera_distance = 5.0F, .bounds_radius = 1.0F, .stable_id = {3, 0, 0, 0}},
    });

    CHECK(order[0].stable_id[0] == 2);
    CHECK(order[1].stable_id[0] == 3);
    CHECK(order[2].stable_id[0] == 1);
}

TEST_CASE("concentric blended batches draw the enclosing one first, whatever order they arrive in") {
    blend_order::Entry const outer{.camera_distance = 6.0F, .bounds_radius = 1.2F, .stable_id = {7, 1, 0, 0}};
    blend_order::Entry const inner{.camera_distance = 6.0F, .bounds_radius = 0.6F, .stable_id = {4, 1, 0, 0}};
    blend_order::Entry const core{.camera_distance = 6.0F, .bounds_radius = 0.25F, .stable_id = {9, 1, 0, 0}};

    for (auto const &arrival: {std::vector{outer, inner, core}, std::vector{core, inner, outer},
                               std::vector{inner, core, outer}}) {
        auto const order = sorted(arrival);
        CHECK(order[0].bounds_radius == outer.bounds_radius);
        CHECK(order[1].bounds_radius == inner.bounds_radius);
        CHECK(order[2].bounds_radius == core.bounds_radius);
    }
}

TEST_CASE("full ties fall back to the stable id, and the order is strict") {
    blend_order::Entry const first{.camera_distance = 2.0F, .bounds_radius = 1.0F, .stable_id = {1, 5, 0, 0}};
    blend_order::Entry const second{.camera_distance = 2.0F, .bounds_radius = 1.0F, .stable_id = {1, 5, 1, 0}};

    CHECK(blend_order::draws_before(first, second));
    CHECK_FALSE(blend_order::draws_before(second, first));
    CHECK_FALSE(blend_order::draws_before(first, first));
}

TEST_CASE("water is drawn before every other blended batch") {
    auto const order = sorted({
            {.camera_distance = 50.0F, .bounds_radius = 1.0F, .stable_id = {1, 0, 0, 0}},
            {.camera_distance = 2.0F, .bounds_radius = 30.0F, .stable_id = {2, 0, 0, 0}, .water = true},
    });

    CHECK(order[0].water);
    CHECK_FALSE(order[1].water);
}

TEST_CASE("double-sided blended batches draw their back faces before their front faces") {
    auto const double_sided = blend_order::passes_for(true, true);
    REQUIRE(double_sided.count == 2);
    CHECK(double_sided.faces[0] == blend_order::Faces::back);
    CHECK(double_sided.faces[1] == blend_order::Faces::front);

    auto const single_sided = blend_order::passes_for(false, true);
    REQUIRE(single_sided.count == 1);
    CHECK(single_sided.faces[0] == blend_order::Faces::front);

    auto const unsplit = blend_order::passes_for(true, false);
    REQUIRE(unsplit.count == 1);
    CHECK(unsplit.faces[0] == blend_order::Faces::both);
}
