#include <doctest/doctest.h>

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
