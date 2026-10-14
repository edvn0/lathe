#include <doctest/doctest.h>

#include <array>
#include <cmath>

#include <glm/gtc/constants.hpp>

#include "rendering/local_shadows.hxx"

using namespace local_shadows;

namespace {
    auto overlaps(Tile const &a, Tile const &b) -> bool {
        return a.x < b.x + b.size && b.x < a.x + a.size && a.y < b.y + b.size && b.y < a.y + a.size;
    }

    auto project(glm::mat4 const &view_projection, glm::vec3 point) -> glm::vec3 {
        auto const clip = view_projection * glm::vec4{point, 1.0F};
        return glm::vec3{clip} / clip.w;
    }
}

TEST_CASE("Atlas hands out aligned, non-overlapping tiles") {
    Atlas atlas;
    atlas.begin_frame();

    auto const *spot = atlas.request(1, 1, 1024);
    auto const *point = atlas.request(2, point_face_count, 512);
    REQUIRE(spot != nullptr);
    REQUIRE(point != nullptr);

    CHECK(spot->tile_count == 1);
    CHECK(point->tile_count == point_face_count);

    std::array<Tile, 1 + point_face_count> tiles{};
    tiles[0] = spot->tiles[0];
    for (std::uint32_t face = 0; face < point_face_count; ++face) {
        tiles[1 + face] = point->tiles[face];
    }

    for (auto const &tile: tiles) {
        CHECK(tile.x % tile.size == 0);
        CHECK(tile.y % tile.size == 0);
        CHECK(tile.x + tile.size <= atlas_size);
        CHECK(tile.y + tile.size <= atlas_size);
    }
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        for (std::size_t j = i + 1; j < tiles.size(); ++j) {
            CHECK_FALSE(overlaps(tiles[i], tiles[j]));
        }
    }

    CHECK(atlas.used_cells() == 16 + (point_face_count * 4));
}

TEST_CASE("Atlas keeps a light's tiles while it is requested") {
    Atlas atlas;
    atlas.begin_frame();
    auto const first = *atlas.request(7, 1, 512);

    atlas.begin_frame();
    auto const *again = atlas.request(7, 1, 512);
    REQUIRE(again != nullptr);
    CHECK(again->tiles[0].x == first.tiles[0].x);
    CHECK(again->tiles[0].y == first.tiles[0].y);
    CHECK(again->generation == first.generation);
    CHECK(again->last_used_frame == atlas.frame());

    atlas.begin_frame();
    auto const *resized = atlas.request(7, 1, 256);
    REQUIRE(resized != nullptr);
    CHECK(resized->resolution == 256);
    CHECK(resized->generation != first.generation);
    CHECK(atlas.allocation_count() == 1);
    CHECK(atlas.used_cells() == 1);
}

TEST_CASE("Atlas clamps the requested resolution") {
    Atlas atlas;
    atlas.begin_frame();

    CHECK(atlas.request(1, 1, 8192)->resolution == maximum_resolution);
    CHECK(atlas.request(2, 1, 1)->resolution == minimum_resolution);
    CHECK(atlas.request(3, 1, 700)->resolution == 512);
    CHECK(atlas.request(0, 1, 512) == nullptr);
    CHECK(atlas.request(4, 0, 512) == nullptr);
    CHECK(atlas.request(5, point_face_count + 1, 512) == nullptr);
}

TEST_CASE("Atlas evicts only lights that were not requested this frame") {
    Atlas atlas;
    atlas.begin_frame();

    // Sixteen 1024 tiles fill the whole atlas.
    for (std::uint64_t id = 1; id <= 16; ++id) {
        REQUIRE(atlas.request(id, 1, 1024) != nullptr);
    }
    CHECK(atlas.used_cells() == cells_per_side * cells_per_side);
    CHECK(atlas.request(100, 1, 256) == nullptr);

    // A new frame in which only some of them are wanted again: the newcomer displaces an unrequested light.
    atlas.begin_frame();
    for (std::uint64_t id = 1; id <= 15; ++id) {
        REQUIRE(atlas.request(id, 1, 1024) != nullptr);
    }
    REQUIRE(atlas.request(100, 1, 1024) != nullptr);
    CHECK(atlas.find(16) == nullptr);
    CHECK(atlas.find(100) != nullptr);

    // Everything is requested this frame, so nothing may be evicted.
    CHECK(atlas.request(101, 1, 256) == nullptr);
    CHECK(atlas.allocation_count() == 16);
}

TEST_CASE("Atlas evicts the least recently used unrequested light first") {
    Atlas atlas;
    atlas.begin_frame();
    for (std::uint64_t id = 1; id <= 16; ++id) {
        REQUIRE(atlas.request(id, 1, 1024) != nullptr);
    }

    atlas.begin_frame();
    for (std::uint64_t id = 2; id <= 16; ++id) {
        REQUIRE(atlas.request(id, 1, 1024) != nullptr);
    }

    atlas.begin_frame();
    for (std::uint64_t id = 3; id <= 16; ++id) {
        REQUIRE(atlas.request(id, 1, 1024) != nullptr);
    }

    // Light 1 was last used two frames ago, light 2 one frame ago.
    REQUIRE(atlas.request(50, 1, 1024) != nullptr);
    CHECK(atlas.find(1) == nullptr);
    CHECK(atlas.find(2) != nullptr);
}

TEST_CASE("Atlas rolls back a request it cannot satisfy") {
    Atlas atlas;
    atlas.begin_frame();
    for (std::uint64_t id = 1; id <= 14; ++id) {
        REQUIRE(atlas.request(id, 1, 1024) != nullptr);
    }
    auto const used_before = atlas.used_cells();

    // A point light at 1024 needs six tiles but only two remain, and everything is in use this frame.
    CHECK(atlas.request(99, point_face_count, 1024) == nullptr);
    CHECK(atlas.used_cells() == used_before);
    CHECK(atlas.find(99) == nullptr);
}

TEST_CASE("Atlas release and clear free their cells") {
    Atlas atlas;
    atlas.begin_frame();
    REQUIRE(atlas.request(1, point_face_count, 512) != nullptr);
    REQUIRE(atlas.request(2, 1, 256) != nullptr);

    atlas.release(1);
    CHECK(atlas.used_cells() == 1);
    CHECK(atlas.find(1) == nullptr);

    atlas.clear();
    CHECK(atlas.used_cells() == 0);
    CHECK(atlas.allocation_count() == 0);
}

TEST_CASE("resolution_for_screen_radius picks a tier by screen size") {
    CHECK(resolution_for_screen_radius(0.0F) == 256);
    CHECK(resolution_for_screen_radius(109.0F) == 256);
    CHECK(resolution_for_screen_radius(110.0F) == 512);
    CHECK(resolution_for_screen_radius(319.0F) == 512);
    CHECK(resolution_for_screen_radius(320.0F) == 1024);
    CHECK(resolution_for_screen_radius(5000.0F) == 1024);
}

TEST_CASE("spot_view_projection maps the cone into the unit depth range, nearer is larger") {
    auto const vp = spot_view_projection({0.0F, 5.0F, 0.0F}, {0.0F, -1.0F, 0.0F}, glm::radians(30.0F), 20.0F, 512);

    auto const near_point = project(vp, {0.0F, 3.0F, 0.0F});
    auto const far_point = project(vp, {0.0F, -10.0F, 0.0F});

    CHECK(near_point.x == doctest::Approx(0.0F).epsilon(1e-4));
    CHECK(near_point.y == doctest::Approx(0.0F).epsilon(1e-4));
    CHECK(near_point.z > 0.0F);
    CHECK(near_point.z < 1.0F);
    CHECK(far_point.z > 0.0F);
    CHECK(near_point.z > far_point.z);

    // Just inside the cone edge lands inside the tile, well outside it does not.
    auto const inside = project(vp, {std::tan(glm::radians(28.0F)) * 5.0F, 0.0F, 0.0F});
    auto const outside = project(vp, {std::tan(glm::radians(40.0F)) * 5.0F, 0.0F, 0.0F});
    CHECK(std::abs(inside.x) < 1.0F);
    CHECK(std::abs(outside.x) > 1.0F);

    // Behind the light has no valid projection.
    CHECK((vp * glm::vec4{0.0F, 20.0F, 0.0F, 1.0F}).w < 0.0F);
}

TEST_CASE("spot_view_projection copes with a spot pointing along the up axis") {
    auto const vp = spot_view_projection(glm::vec3{0.0F}, {0.0F, 1.0F, 0.0F}, glm::radians(25.0F), 10.0F, 512);
    auto const centre = project(vp, {0.0F, 4.0F, 0.0F});

    CHECK(centre.x == doctest::Approx(0.0F).epsilon(1e-4));
    CHECK(centre.y == doctest::Approx(0.0F).epsilon(1e-4));
}

TEST_CASE("point faces each look down their own axis") {
    constexpr std::array<glm::vec3, point_face_count> directions{{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                                                  {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};

    for (std::uint32_t face = 0; face < point_face_count; ++face) {
        auto const vp = point_face_view_projection({1.0F, 2.0F, 3.0F}, face, 30.0F, 512);
        auto const target = glm::vec3{1.0F, 2.0F, 3.0F} + (directions[face] * 6.0F);
        auto const centre = project(vp, target);

        CAPTURE(face);
        CHECK(centre.x == doctest::Approx(0.0F).epsilon(1e-4));
        CHECK(centre.y == doctest::Approx(0.0F).epsilon(1e-4));
        CHECK(centre.z > 0.0F);
        CHECK(centre.z < 1.0F);

        auto const behind = glm::vec3{1.0F, 2.0F, 3.0F} - (directions[face] * 6.0F);
        CHECK((vp * glm::vec4{behind, 1.0F}).w < 0.0F);
    }
}

TEST_CASE("point faces overlap by a guard border so edges filter over real depth") {
    auto const vp = point_face_view_projection(glm::vec3{0.0F}, 0, 30.0F, 512);

    // Exactly on the 45 degree seam is a hair inside the face.
    auto const seam = project(vp, {5.0F, 5.0F, 0.0F});
    CHECK(seam.y < 1.0F);
    CHECK(seam.y > 0.99F);
}

TEST_CASE("make_gpu_tile places the tile in atlas UV space") {
    auto const gpu = make_gpu_tile(glm::mat4{1.0F}, Tile{.x = 1024, .y = 2048, .size = 512}, 0.01F);

    CHECK(gpu.atlas_rect.x == doctest::Approx(0.25F));
    CHECK(gpu.atlas_rect.y == doctest::Approx(0.5F));
    CHECK(gpu.atlas_rect.z == doctest::Approx(0.125F));
    CHECK(gpu.atlas_rect.w == doctest::Approx(0.125F));
    CHECK(gpu.tile_texel == doctest::Approx(1.0F / 512.0F));
    CHECK(gpu.texel_world == doctest::Approx(0.01F));
}
