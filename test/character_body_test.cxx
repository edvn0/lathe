#include <doctest/doctest.h>

#include "physics/character_body.hxx"
#include "physics/fixed_stepper.hxx"
#include "physics/physics_world.hxx"
#include "scene/components.hxx"

#include <BS_thread_pool.hpp>
#include <entt/entt.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

TEST_SUITE("unit") {
    namespace {
        constexpr float dt = 1.0F / 60.0F;
        constexpr float radius = 0.3F;
        constexpr float height = 1.8F;

        struct Plane {
            glm::vec3 normal{0.0F, 1.0F, 0.0F};
            glm::vec3 point{0.0F};
            float x_min = -1e9F;
            float x_max = 1e9F;
            float y_max = 1e9F;
        };

        class FakeWorld final : public CapsuleSweep {
        public:
            std::vector<Plane> planes;

            [[nodiscard]] auto sweep_capsule(glm::vec3 const &from, glm::vec3 const &to, float r, float,
                                             entt::entity) const -> std::optional<SweepHit> override {
                std::optional<SweepHit> best;
                for (auto const &plane: planes) {
                    glm::vec3 const c0 = from + glm::vec3{0.0F, r, 0.0F};
                    glm::vec3 const c1 = to + glm::vec3{0.0F, r, 0.0F};
                    float const d0 = glm::dot(plane.normal, c0 - plane.point) - r;
                    float const d1 = glm::dot(plane.normal, c1 - plane.point) - r;
                    if (d1 >= d0 || d1 >= 0.0F || d0 < -0.001F) {
                        continue;
                    }
                    float const fraction = std::clamp(d0 / (d0 - d1), 0.0F, 1.0F);
                    glm::vec3 const at = c0 + (c1 - c0) * fraction;
                    if (at.x < plane.x_min || at.x > plane.x_max || at.y - r > plane.y_max) {
                        continue;
                    }
                    if (!best || fraction < best->fraction) {
                        best = SweepHit{.fraction = fraction, .point = at, .normal = plane.normal};
                    }
                }
                return best;
            }
        };

        auto flat_world() -> FakeWorld {
            FakeWorld world;
            world.planes.push_back(Plane{});
            return world;
        }

        auto run(CharacterBody &body, FakeWorld const &world, CharacterInput const &input, int steps) -> void {
            for (int i = 0; i < steps; ++i) {
                body.step(world, input, dt);
            }
        }

        auto settled(FakeWorld const &world, MovementParams const &params = {}) -> CharacterBody {
            CharacterBody body{{0.0F, 0.5F, 0.0F}, radius, height, params};
            run(body, world, {}, 60);
            return body;
        }

        auto jump_apex(CharacterBody &body, FakeWorld const &world, int held_steps) -> float {
            float apex = body.position().y;
            for (int i = 0; i < 120; ++i) {
                CharacterInput const input{.jump_pressed = i == 0, .jump_held = i < held_steps};
                body.step(world, input, dt);
                apex = std::max(apex, body.position().y);
                if (i > 3 && body.grounded()) {
                    break;
                }
            }
            return apex;
        }
    }

    TEST_CASE("CharacterBody lands and stays grounded on flat ground") {
        auto const world = flat_world();
        auto const body = settled(world);
        CHECK(body.grounded());
        CHECK(std::abs(body.position().y) < 0.02F);
    }

    TEST_CASE("CharacterBody jump apex matches jump_height") {
        auto const world = flat_world();
        MovementParams params;
        params.jump_height = 1.5F;
        auto body = settled(world, params);
        float const base = body.position().y;
        float const apex = jump_apex(body, world, 1000);
        CHECK(apex - base == doctest::Approx(1.5F).epsilon(0.03));
    }

    TEST_CASE("Releasing jump early lowers the apex") {
        auto const world = flat_world();
        auto full = settled(world);
        auto cut = settled(world);
        float const full_apex = jump_apex(full, world, 1000);
        float const cut_apex = jump_apex(cut, world, 3);
        CHECK(cut_apex < full_apex * 0.7F);
        CHECK(cut_apex > cut.position().y + 0.05F);
    }

    TEST_CASE("Coyote time allows a late jump but not a later one") {
        auto const world = [] {
            FakeWorld w;
            w.planes.push_back(Plane{.x_max = 1.0F});
            return w;
        }();

        auto walk_off = [&](float extra_air) {
            CharacterBody body{{0.5F, 0.0F, 0.0F}, radius, height, MovementParams{.step_height = 0.1F}};
            run(body, world, {}, 30);
            CharacterInput const walk{.desired_velocity = {6.0F, 0.0F, 0.0F}};
            int guard = 0;
            while (body.grounded() && guard++ < 200) {
                body.step(world, walk, dt);
            }
            REQUIRE_FALSE(body.grounded());
            run(body, world, walk, static_cast<int>(extra_air / dt));
            body.step(world, CharacterInput{.desired_velocity = walk.desired_velocity, .jump_pressed = true,
                                            .jump_held = true},
                      dt);
            return body.velocity().y;
        };

        CHECK(walk_off(0.05F) > 0.0F);
        CHECK(walk_off(0.3F) < 0.0F);
    }

    TEST_CASE("Coyote time does not allow a double jump") {
        auto const world = flat_world();
        auto body = settled(world);
        body.step(world, {.jump_pressed = true, .jump_held = true}, dt);
        float const v1 = body.velocity().y;
        body.step(world, {.jump_pressed = true, .jump_held = true}, dt);
        CHECK(body.velocity().y < v1);
    }

    TEST_CASE("A jump pressed shortly before landing fires on landing") {
        auto const world = flat_world();
        CharacterBody body{{0.0F, 0.3F, 0.0F}, radius, height};
        bool jumped = false;
        for (int i = 0; i < 60 && !jumped; ++i) {
            body.step(world, {.jump_pressed = i == 2, .jump_held = true}, dt);
            jumped = body.velocity().y > 0.5F;
        }
        CHECK(jumped);

        CharacterBody early{{0.0F, 3.0F, 0.0F}, radius, height};
        early.step(world, {.jump_pressed = true, .jump_held = true}, dt);
        run(early, world, {}, 120);
        CHECK(early.grounded());
        CHECK(early.velocity().y == doctest::Approx(0.0F));
    }

    TEST_CASE("Ground acceleration and deceleration follow the parameters") {
        auto const world = flat_world();
        MovementParams params;
        params.ground_accel = 30.0F;
        params.ground_decel = 60.0F;
        auto body = settled(world, params);

        CharacterInput const walk{.desired_velocity = {5.0F, 0.0F, 0.0F}};
        run(body, world, walk, 6);
        CHECK(body.velocity().x == doctest::Approx(30.0F * 6.0F * dt));
        run(body, world, walk, 60);
        CHECK(body.velocity().x == doctest::Approx(5.0F));

        run(body, world, {}, 3);
        CHECK(body.velocity().x == doctest::Approx(5.0F - 60.0F * 3.0F * dt));
        run(body, world, {}, 30);
        CHECK(body.velocity().x == doctest::Approx(0.0F));
    }

    TEST_CASE("Air control uses air_accel") {
        auto const world = flat_world();
        MovementParams params;
        params.air_accel = 10.0F;
        CharacterBody body{{0.0F, 10.0F, 0.0F}, radius, height, params};
        run(body, world, {.desired_velocity = {5.0F, 0.0F, 0.0F}}, 6);
        CHECK_FALSE(body.grounded());
        CHECK(body.velocity().x == doctest::Approx(10.0F * 6.0F * dt));
    }

    TEST_CASE("Slopes steeper than the limit are not walkable") {
        constexpr float pi = 3.14159265F;
        auto make = [](float degrees) {
            float const a = degrees * pi / 180.0F;
            FakeWorld w;
            w.planes.push_back(Plane{.normal = {std::sin(a), std::cos(a), 0.0F}});
            return w;
        };

        auto const gentle = make(30.0F);
        CharacterBody on_gentle{{0.0F, 0.2F, 0.0F}, radius, height, MovementParams{.max_slope_degrees = 45.0F}};
        run(on_gentle, gentle, {}, 60);
        CHECK(on_gentle.grounded());
        CHECK(on_gentle.ground_normal().y == doctest::Approx(std::cos(30.0F * pi / 180.0F)));

        auto const steep = make(60.0F);
        CharacterBody on_steep{{0.0F, 0.2F, 0.0F}, radius, height, MovementParams{.max_slope_degrees = 45.0F}};
        run(on_steep, steep, {}, 60);
        CHECK_FALSE(on_steep.grounded());
    }

    TEST_CASE("Walking off a tiny step stays grounded; a big drop does not") {
        FakeWorld world;
        world.planes.push_back(Plane{.x_max = 1.0F});
        world.planes.push_back(Plane{.point = {0.0F, -0.15F, 0.0F}});

        CharacterBody body{{0.0F, 0.0F, 0.0F}, radius, height};
        run(body, world, {}, 10);
        CharacterInput const walk{.desired_velocity = {3.0F, 0.0F, 0.0F}};
        for (int i = 0; i < 90; ++i) {
            body.step(world, walk, dt);
            CHECK(body.grounded());
        }
        CHECK(body.position().x > 1.5F);
        CHECK(body.position().y == doctest::Approx(-0.15F).epsilon(0.1));

        FakeWorld cliff;
        cliff.planes.push_back(Plane{.x_max = 1.0F});
        cliff.planes.push_back(Plane{.point = {0.0F, -3.0F, 0.0F}});
        CharacterBody faller{{0.0F, 0.0F, 0.0F}, radius, height};
        run(faller, cliff, {}, 10);
        bool airborne = false;
        for (int i = 0; i < 90; ++i) {
            faller.step(cliff, walk, dt);
            airborne = airborne || !faller.grounded();
        }
        CHECK(airborne);
    }

    TEST_CASE("Character steps up a low ledge") {
        FakeWorld world;
        world.planes.push_back(Plane{});
        world.planes.push_back(Plane{.point = {0.0F, 0.2F, 0.0F}, .x_min = 1.6F});
        world.planes.push_back(Plane{.normal = {-1.0F, 0.0F, 0.0F}, .point = {2.0F, 0.0F, 0.0F}, .y_max = 0.19F});

        CharacterBody body{{0.0F, 0.0F, 0.0F}, radius, height};
        run(body, world, {}, 10);
        run(body, world, {.desired_velocity = {3.0F, 0.0F, 0.0F}}, 120);
        CHECK(body.grounded());
        CHECK(body.position().y == doctest::Approx(0.2F).epsilon(0.1));
    }

    TEST_CASE("interpolated_position blends previous and current position") {
        auto const world = flat_world();
        CharacterBody body{{0.0F, 0.0F, 0.0F}, radius, height};
        run(body, world, {}, 10);
        body.step(world, {.desired_velocity = {6.0F, 0.0F, 0.0F}}, dt);
        float const from = body.previous_position().x;
        float const to = body.position().x;
        REQUIRE(to > from);
        CHECK(body.interpolated_position(0.0F).x == doctest::Approx(from));
        CHECK(body.interpolated_position(0.5F).x == doctest::Approx((from + to) * 0.5F));
        CHECK(body.interpolated_position(1.0F).x == doctest::Approx(to));
    }

    TEST_CASE("FixedStepper step counts and alpha") {
        FixedStepper stepper{1.0F / 60.0F, 5};
        int steps = 0;
        auto count = [&](float) { ++steps; };

        float alpha = stepper.advance(1.0F / 120.0F, count);
        CHECK(steps == 0);
        CHECK(alpha == doctest::Approx(0.5F));

        alpha = stepper.advance(1.0F / 120.0F, count);
        CHECK(steps == 1);
        CHECK(alpha < 0.01F);

        steps = 0;
        alpha = stepper.advance(3.5F / 60.0F, count);
        CHECK(steps == 3);
        CHECK(alpha == doctest::Approx(0.5F));

        steps = 0;
        alpha = stepper.advance(2.0F, count);
        CHECK(steps == 5);
        CHECK(alpha < 1.0F);
        steps = 0;
        stepper.advance(0.0F, count);
        CHECK(steps == 0);
    }

    TEST_CASE("CharacterBody walks on a real PhysicsWorld floor and ignores its own collider") {
        BS::priority_thread_pool pool{2};
        entt::registry registry;
        PhysicsWorldSettings const settings{};
        PhysicsWorld world{settings, pool, registry};

        auto const floor = registry.create();
        auto const floor_transform =
                registry.emplace<Components::Transform>(floor, Components::Transform{.position = {0.0F, -0.5F, 0.0F}});
        auto const floor_body = registry.emplace<Components::RigidBody>(
                floor, Components::RigidBody{.half_extents = {50.0F, 0.5F, 50.0F}, .is_static = true});
        world.add_body(registry, floor, floor_transform, floor_body);

        auto const self = registry.create();
        auto const self_transform =
                registry.emplace<Components::Transform>(self, Components::Transform{.position = {0.0F, 0.9F, 0.0F}});
        auto const self_body = registry.emplace<Components::RigidBody>(
                self, Components::RigidBody{.mass = 1.0F, .is_static = true});
        world.add_body(registry, self, self_transform, self_body);
        world.step(registry, dt);

        CharacterBody body{{0.0F, 1.0F, 0.0F}, radius, height, {}, self};
        for (int i = 0; i < 90; ++i) {
            body.step(world, {.desired_velocity = {2.0F, 0.0F, 0.0F}}, dt);
        }
        CHECK(body.grounded());
        CHECK(std::abs(body.position().y) < 0.02F);
        CHECK(body.position().x > 1.0F);
    }

    TEST_CASE("CharacterBody walks up a staircase of 0.25 m rises on a real PhysicsWorld") {
        BS::priority_thread_pool pool{2};
        entt::registry registry;
        PhysicsWorldSettings const settings{};
        PhysicsWorld world{settings, pool, registry};

        auto add_box = [&](glm::vec3 const &position, glm::vec3 const &half_extents) {
            auto const entity = registry.create();
            auto const transform =
                    registry.emplace<Components::Transform>(entity, Components::Transform{.position = position});
            auto const body = registry.emplace<Components::RigidBody>(
                    entity, Components::RigidBody{.half_extents = half_extents, .is_static = true});
            world.add_body(registry, entity, transform, body);
        };
        add_box({0.0F, -0.5F, 0.0F}, {80.0F, 0.5F, 80.0F});
        for (int i = 0; i < 4; ++i) {
            auto const rise = 0.25F * static_cast<float>(i + 1);
            add_box({6.0F + 1.5F * static_cast<float>(i), rise * 0.5F, 0.0F}, {0.75F, rise * 0.5F, 3.0F});
        }
        world.step(registry, dt);

        CharacterBody body{{3.0F, 0.0F, 0.0F}, radius, height};
        for (int i = 0; i < 120; ++i) {
            body.step(world, {.desired_velocity = {4.0F, 0.0F, 0.0F}}, dt);
        }
        CHECK(body.grounded());
        CHECK(body.position().y == doctest::Approx(1.0F).epsilon(0.05));
        CHECK(body.position().x > 10.0F);
    }
}
