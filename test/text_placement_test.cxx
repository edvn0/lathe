#include <doctest/doctest.h>

#include <array>

#include <glm/geometric.hpp>

#include "app/text_placement.hxx"

namespace {

    auto check_vec(glm::vec3 const &actual, glm::vec3 const &expected) -> void {
        CHECK(actual.x == doctest::Approx(expected.x).epsilon(1e-4));
        CHECK(actual.y == doctest::Approx(expected.y).epsilon(1e-4));
        CHECK(actual.z == doctest::Approx(expected.z).epsilon(1e-4));
    }

    constexpr glm::vec3 world_up{0.0F, 1.0F, 0.0F};

}

TEST_CASE("text on a wall facing -z is unrotated and lifted off the wall") {
    auto const frame = text_placement::surface_frame({1.0F, 2.0F, 3.0F}, {0.0F, 0.0F, -1.0F}, world_up,
                                                     {0.0F, 0.0F, 1.0F}, 0.1F);

    check_vec(frame.position, {1.0F, 2.0F, 2.9F});
    check_vec(frame.rotation * glm::vec3{1.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
    check_vec(frame.rotation * glm::vec3{0.0F, 1.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
}

TEST_CASE("text keeps its up pointing up on any wall and its back to the surface") {
    for (auto const &normal: {glm::vec3{1.0F, 0.0F, 0.0F}, glm::vec3{-1.0F, 0.0F, 0.0F}, glm::normalize(glm::vec3{1.0F, 0.3F, 1.0F})}) {
        auto const frame = text_placement::surface_frame(glm::vec3{0.0F}, normal, world_up, {0.0F, 0.0F, 1.0F}, 0.0F);

        auto const text_up = frame.rotation * glm::vec3{0.0F, 1.0F, 0.0F};
        auto const text_forward = frame.rotation * glm::vec3{0.0F, 0.0F, 1.0F};

        CHECK(glm::dot(text_up, world_up) > 0.9F);
        // Text faces a viewer looking along its +z, so a viewer in front of the wall looks along -normal.
        check_vec(text_forward, -normal);
        CHECK(glm::dot(frame.rotation * glm::vec3{1.0F, 0.0F, 0.0F}, text_up) == doctest::Approx(0.0F).epsilon(1e-4));
    }
}

TEST_CASE("text on a floor takes its up from the fallback instead of spinning") {
    auto const frame =
            text_placement::surface_frame(glm::vec3{0.0F}, {0.0F, 1.0F, 0.0F}, world_up, {0.0F, 0.0F, 1.0F}, 0.0F);

    check_vec(frame.rotation * glm::vec3{0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F});
    check_vec(frame.rotation * glm::vec3{0.0F, 0.0F, 1.0F}, {0.0F, -1.0F, 0.0F});
}

TEST_CASE("a hint along the normal and no usable fallback still gives a valid frame") {
    auto const frame = text_placement::surface_frame(glm::vec3{0.0F}, {0.0F, 1.0F, 0.0F}, world_up, world_up, 0.0F);
    CHECK(glm::length(frame.rotation) == doctest::Approx(1.0F));
    check_vec(frame.rotation * glm::vec3{0.0F, 0.0F, 1.0F}, {0.0F, -1.0F, 0.0F});
}

TEST_CASE("a local transform under a parent lands back at the frame in the world") {
    auto const frame = text_placement::surface_frame({4.0F, 1.0F, -2.0F}, glm::normalize(glm::vec3{1.0F, 0.0F, 1.0F}),
                                                     world_up, {0.0F, 0.0F, 1.0F}, 0.05F);

    Components::Transform parent;
    parent.position = {10.0F, -3.0F, 2.0F};
    parent.rotation = glm::angleAxis(0.7F, glm::normalize(glm::vec3{0.2F, 1.0F, 0.1F}));
    parent.scale = {2.0F, 2.0F, 2.0F};

    auto const local = text_placement::local_transform(parent.matrix(), frame);
    auto const world = parent.matrix() * local.matrix();

    check_vec(glm::vec3{world[3]}, frame.position);
    check_vec(glm::mat3{world} * glm::vec3{0.0F, 0.0F, 1.0F}, frame.rotation * glm::vec3{0.0F, 0.0F, 1.0F});
    // A uniformly scaled parent is undone by the child's scale, so the text keeps its world size.
    CHECK(local.scale.x == doctest::Approx(0.5F));
}

namespace {

    // A camera at the origin looking along +z, with a 90 degree field of view on a 1000 x 1000 viewport.
    auto head_on() -> text_placement::ScreenProjector {
        return {.view = glm::mat4{1.0F}, .focal = {1.0F, 1.0F}, .origin = {0.0F, 0.0F}, .size = {1000.0F, 1000.0F}};
    }

    // A triangle two metres ahead whose corners land at known pixels: (500,500) is the screen centre.
    constexpr std::array<glm::vec3, 3> wall{{{-1.0F, 1.0F, 2.0F}, {1.0F, 1.0F, 2.0F}, {0.0F, -1.0F, 2.0F}}};

}

TEST_CASE("a point projects to the pixel the camera would draw it at, and one behind the camera does not") {
    auto const centre = head_on().project({0.0F, 0.0F, 3.0F});
    REQUIRE(centre.has_value());
    CHECK(centre->x == doctest::Approx(500.0F));
    CHECK(centre->y == doctest::Approx(500.0F));

    // x = z at 90 degrees is the right edge; y = z is the top.
    auto const corner = head_on().project({2.0F, 2.0F, 2.0F});
    REQUIRE(corner.has_value());
    CHECK(corner->x == doctest::Approx(1000.0F));
    CHECK(corner->y == doctest::Approx(0.0F));

    CHECK_FALSE(head_on().project({0.0F, 0.0F, -1.0F}).has_value());
}

TEST_CASE("the cursor snaps to a nearby vertex, edge midpoint or centre") {
    auto const projector = head_on();
    // Pixels: corners (250,250), (750,250), (500,750); midpoints (500,250), (625,500), (375,500); centre (500,417).
    auto const vertex = text_placement::snap_to_triangle(wall, projector, {255.0F, 255.0F}, 16.0F);
    REQUIRE(vertex.has_value());
    CHECK(vertex->kind == text_placement::SnapKind::vertex);
    check_vec(vertex->position, wall[0]);

    auto const edge = text_placement::snap_to_triangle(wall, projector, {505.0F, 244.0F}, 16.0F);
    REQUIRE(edge.has_value());
    CHECK(edge->kind == text_placement::SnapKind::edge_midpoint);
    check_vec(edge->position, {0.0F, 1.0F, 2.0F});

    auto const centre = text_placement::snap_to_triangle(wall, projector, {500.0F, 425.0F}, 16.0F);
    REQUIRE(centre.has_value());
    CHECK(centre->kind == text_placement::SnapKind::face_centre);
}

TEST_CASE("a cursor outside the radius of every target snaps to nothing") {
    CHECK_FALSE(text_placement::snap_to_triangle(wall, head_on(), {600.0F, 300.0F}, 16.0F).has_value());
    CHECK_FALSE(text_placement::snap_to_triangle(wall, head_on(), {255.0F, 255.0F}, 2.0F).has_value());
}

TEST_CASE("a finer target wins over a nearer coarse one when both are in reach") {
    // With a wide radius the centre is closer to a cursor at the middle, but the edge midpoint at (500,250) is in
    // reach too and the vertex at (250,250) is not.
    auto const snap = text_placement::snap_to_triangle(wall, head_on(), {500.0F, 380.0F}, 140.0F);
    REQUIRE(snap.has_value());
    CHECK(snap->kind == text_placement::SnapKind::edge_midpoint);
}
