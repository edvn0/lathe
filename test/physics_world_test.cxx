#include <doctest/doctest.h>

#include "components.hxx"
#include "physics_world.hxx"

#include <BS_thread_pool.hpp>
#include <entt/entt.hpp>

#include <memory>

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
}
