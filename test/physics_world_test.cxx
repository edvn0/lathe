#include <doctest/doctest.h>

#include "scene/components.hxx"
#include "physics/physics_world.hxx"

#include <BS_thread_pool.hpp>
#include <entt/entt.hpp>

#include <memory>
#include <vector>

// Repeated construct/step/destroy cycles, as play()/stop() do. Bullet's task scheduler is a process global,
// and setting a new one calls deactivate() on the previous one, so a PhysicsWorld that doesn't clear it
// leaves the next one dereferencing a dangling pointer.
TEST_SUITE("unit") {
    TEST_CASE("PhysicsWorld survives repeated stop/start cycles") {
        BS::priority_thread_pool pool{2};

        entt::registry registry;
        auto const entity = registry.create();
        registry.emplace<Components::Transform>(entity, Components::Transform{.position = {0.0F, 5.0F, 0.0F}});
        registry.emplace<Components::RigidBody>(entity, Components::RigidBody{});

        for (int run = 0; run < 4; ++run) {
            PhysicsWorldSettings const settings{};
            auto world = std::make_unique<PhysicsWorld>(settings, pool, registry);

            world->populate_from(registry);

            for (int step = 0; step < 5; ++step) {
                world->step(registry, 1.0F / 60.0F);
            }

            CHECK(registry.get<Components::Transform>(entity).position.y < 5.0F);

            world.reset(); // as Scene::on_scene_stop() does
        }
    }

    TEST_CASE("PhysicsWorld::add_body/remove_body round-trip does not corrupt the world") {
        BS::priority_thread_pool pool{2};
        entt::registry registry;

        PhysicsWorldSettings const settings{};
        PhysicsWorld world{settings, pool, registry};

        auto const entity = registry.create();
        auto const transform =
                registry.emplace<Components::Transform>(entity, Components::Transform{.position = {0.0F, 1.0F, 0.0F}});
        auto const body = registry.emplace<Components::RigidBody>(entity, Components::RigidBody{.is_static = true});

        world.add_body(registry, entity, transform, body);
        world.step(registry, 1.0F / 60.0F);
        world.remove_body(registry, entity);
        world.step(registry, 1.0F / 60.0F);

        CHECK(true); // not crashing is the assertion
    }

    namespace {

        auto settle(PhysicsWorld &world, entt::registry &registry) -> void {
            for (int step = 0; step < 240; ++step) {
                world.step(registry, 1.0F / 60.0F);
            }
        }

    } // namespace

    TEST_CASE("Terrain collider: bind places a dropped body at the expected height") {
        BS::priority_thread_pool pool{2};
        entt::registry registry;

        PhysicsWorldSettings const settings{};
        PhysicsWorld world{settings, pool, registry};

        // Flat heightfield centred in its symmetric AABB.
        TerrainColliderDesc const desc{
                .samples_x = 5, .samples_z = 5, .cell_size_x = 1.0F, .cell_size_z = 1.0F, .min_height = -1.0F, .max_height = 1.0F,
        };
        auto const handle = world.reserve_terrain_collider(desc);
        REQUIRE(handle.valid());

        std::vector<float> const flat_heights(static_cast<std::size_t>(desc.samples_x) * desc.samples_z, 0.0F);
        world.bind_terrain_collider(handle, glm::vec3{0.0F, 5.0F, 0.0F}, flat_heights);

        auto const entity = registry.create();
        auto const transform = registry.emplace<Components::Transform>(
                entity, Components::Transform{.position = {0.0F, 15.0F, 0.0F}});
        auto const body = registry.emplace<Components::RigidBody>(
                entity, Components::RigidBody{.half_extents = {0.5F, 0.5F, 0.5F}, .mass = 1.0F});

        world.add_body(registry, entity, transform, body);
        settle(world, registry);

        CHECK(registry.get<Components::Transform>(entity).position.y == doctest::Approx(5.5F).epsilon(0.1));
    }

    TEST_CASE("Terrain collider: rebinding the same handle moves a resting body to the new height") {
        BS::priority_thread_pool pool{2};
        entt::registry registry;

        PhysicsWorldSettings const settings{};
        PhysicsWorld world{settings, pool, registry};

        TerrainColliderDesc const desc{
                .samples_x = 5, .samples_z = 5, .cell_size_x = 1.0F, .cell_size_z = 1.0F, .min_height = -1.0F, .max_height = 1.0F,
        };
        auto const handle = world.reserve_terrain_collider(desc);
        REQUIRE(handle.valid());

        std::vector<float> const flat_heights(static_cast<std::size_t>(desc.samples_x) * desc.samples_z, 0.0F);
        world.bind_terrain_collider(handle, glm::vec3{0.0F, 5.0F, 0.0F}, flat_heights);

        auto const entity = registry.create();
        auto const transform = registry.emplace<Components::Transform>(
                entity, Components::Transform{.position = {0.0F, 15.0F, 0.0F}});
        auto const body = registry.emplace<Components::RigidBody>(
                entity, Components::RigidBody{.half_extents = {0.5F, 0.5F, 0.5F}, .mass = 1.0F});

        world.add_body(registry, entity, transform, body);
        settle(world, registry);
        REQUIRE(registry.get<Components::Transform>(entity).position.y == doctest::Approx(5.5F).epsilon(0.1));

        // Rebinding to a new centre must keep the shape working.
        world.bind_terrain_collider(handle, glm::vec3{0.0F, 10.0F, 0.0F}, flat_heights);
        world.set_velocity(registry, entity, glm::vec3{0.0F, 0.0F, 0.0F});
        settle(world, registry);

        CHECK(registry.get<Components::Transform>(entity).position.y == doctest::Approx(10.5F).epsilon(0.1));
    }

    TEST_CASE("Terrain collider slots survive repeated PhysicsWorld construct/destroy cycles, entity 0 unaffected") {
        BS::priority_thread_pool pool{2};
        entt::registry registry;

        auto const entity_zero = registry.create();
        REQUIRE(entity_zero == entt::entity{0});
        registry.emplace<Components::Transform>(entity_zero, Components::Transform{.position = {0.0F, 1.0F, 0.0F}});
        registry.emplace<Components::RigidBody>(entity_zero, Components::RigidBody{.is_static = true});

        for (int run = 0; run < 4; ++run) {
            PhysicsWorldSettings const settings{};
            auto world = std::make_unique<PhysicsWorld>(settings, pool, registry);

            world->populate_from(registry);
            REQUIRE(registry.all_of<Components::PhysicsBody>(entity_zero));

            TerrainColliderDesc const desc{.samples_x = 3, .samples_z = 3};
            auto const handle = world->reserve_terrain_collider(desc);
            std::vector<float> const heights(9, 0.0F);
            world->bind_terrain_collider(handle, glm::vec3{0.0F}, heights);

            world->step(registry, 1.0F / 60.0F);

            world.reset(); // as Scene::on_scene_stop() does

            // Entity 0's body is torn down normally despite the terrain collider's null user pointer.
            CHECK(registry.valid(entity_zero));
            CHECK_FALSE(registry.all_of<Components::PhysicsBody>(entity_zero));
        }
    }
}
