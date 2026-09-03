#include <doctest/doctest.h>

#include "scene/components.hxx"
#include "physics/physics_world.hxx"

#include <BS_thread_pool.hpp>
#include <entt/entt.hpp>

#include <memory>
#include <vector>

//
// PhysicsWorld needs no live GPU/window -- only a thread pool and Bullet --
// so its play/stop lifecycle is unit-testable in isolation. This suite
// specifically covers repeated construct/destroy cycles: Application::play()
// creates a fresh PhysicsWorld and Application::stop() destroys it, once per
// simulation run, exactly like the loop below.
//
// Bullet's task scheduler is a process-global (btSetTaskScheduler /
// btGetTaskScheduler in LinearMath/btThreads.h) that PhysicsWorld::Impl
// installs from its own ThreadPoolTaskScheduler member. A prior bug left
// that global dangling once the owning PhysicsWorld was destroyed: the next
// PhysicsWorld's constructor called btSetTaskScheduler() again, which
// dereferences the *previous* scheduler to call deactivate() on it first --
// a use-after-free that crashed on the second play(). This test reproduces
// that exact construct/step/destroy/reconstruct sequence so a regression
// fails here under plain CTest, and fails loudly under ASan (SANITIZE=1).
//
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

            world.reset(); // mirrors Scene::on_scene_stop()'s physics_world.reset()
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

        CHECK(true); // reaching here without crashing/UB is the assertion
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

        // Flat heightfield (all zero samples), centred in its own local
        // AABB since min/max are symmetric -- see
        // PhysicsWorld::bind_terrain_collider's doc comment.
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

        // Rebind the same handle to a new centre and re-settle -- proves
        // the height rewrite and the shape's AABB (derived once from
        // desc.min_height/max_height, never from the data) both still work
        // after the move.
        world.bind_terrain_collider(handle, glm::vec3{0.0F, 10.0F, 0.0F}, flat_heights);
        world.set_velocity(registry, entity, glm::vec3{0.0F, 0.0F, 0.0F});
        settle(world, registry);

        CHECK(registry.get<Components::Transform>(entity).position.y == doctest::Approx(10.5F).epsilon(0.1));
    }

    TEST_CASE("Terrain collider slots survive repeated PhysicsWorld construct/destroy cycles, entity 0 unaffected") {
        BS::priority_thread_pool pool{2};
        entt::registry registry;

        // entt::entity{0} bit-casts to a null pointer, same as a terrain
        // collider's (deliberately unset) user pointer -- see
        // PhysicsWorld::reserve_terrain_collider. ~Impl's generic teardown
        // loop relies on terrain colliders being removed from the world
        // (via a separate, earlier loop) before it runs, rather than on a
        // null-pointer check, precisely so entity 0's own body is torn down
        // normally instead of being mistaken for "no entity backing" --
        // this reproduces both bodies coexisting across repeated
        // construct/destroy cycles.
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

            world.reset(); // mirrors Scene::on_scene_stop()'s physics_world.reset()

            // entity 0's own body is torn down along with its PhysicsWorld,
            // same as any other entity -- the terrain collider's coexisting
            // null-user-pointer body must not have changed that.
            CHECK(registry.valid(entity_zero));
            CHECK_FALSE(registry.all_of<Components::PhysicsBody>(entity_zero));
        }
    }
}
