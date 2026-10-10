#include <doctest/doctest.h>

#include "physics/physics_world.hxx"
#include "scene/components.hxx"

#include <BS_thread_pool.hpp>
#include <entt/entt.hpp>

TEST_SUITE("unit") {
    TEST_CASE("PhysicsWorld reports the entity of a body hit by a ray or a sweep, including entity 0") {
        BS::priority_thread_pool pool{2};
        entt::registry registry;
        PhysicsWorldSettings const settings{};
        PhysicsWorld world{settings, pool, registry};

        // The first entity of a fresh registry has id 0.
        auto const floor = registry.create();
        REQUIRE(entt::to_integral(floor) == 0);

        auto const transform = registry.emplace<Components::Transform>(floor, Components::Transform{.position = {0, -0.5F, 0}});
        auto const body = registry.emplace<Components::RigidBody>(
                floor, Components::RigidBody{.half_extents = {10.0F, 0.5F, 10.0F}, .is_static = true});
        world.add_body(registry, floor, transform, body);

        auto const ray = world.raycast({0.0F, 5.0F, 0.0F}, {0.0F, -1.0F, 0.0F}, 10.0F);
        REQUIRE(ray.has_value());
        CHECK(ray->entity == floor);

        auto const segment = world.raycast({0.0F, 5.0F, 0.0F}, {0.0F, -5.0F, 0.0F});
        REQUIRE(segment.has_value());
        CHECK(segment->entity == floor);

        auto const sweep = world.sweep_capsule({0.0F, 3.0F, 0.0F}, {0.0F, -1.0F, 0.0F}, 0.3F, 1.8F, entt::null);
        REQUIRE(sweep.has_value());
        CHECK(sweep->entity == floor);

        // A second entity is told apart from it.
        auto const wall = registry.create();
        auto const wall_transform = registry.emplace<Components::Transform>(wall, Components::Transform{.position = {5.0F, 1.0F, 0.0F}});
        auto const wall_body = registry.emplace<Components::RigidBody>(
                wall, Components::RigidBody{.half_extents = {0.5F, 1.0F, 3.0F}, .is_static = true});
        world.add_body(registry, wall, wall_transform, wall_body);

        auto const across = world.raycast({0.0F, 1.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 10.0F);
        REQUIRE(across.has_value());
        CHECK(across->entity == wall);

        // Nothing hit reports no hit rather than entity 0.
        CHECK_FALSE(world.raycast({0.0F, 5.0F, 50.0F}, {0.0F, -1.0F, 0.0F}, 10.0F).has_value());
    }
}
