#include <doctest/doctest.h>

#include "assets/load_model.hxx"
#include "physics/character_body.hxx"
#include "physics/mesh_collider.hxx"
#include "physics/physics_world.hxx"
#include "scene/components.hxx"

#include <BS_thread_pool.hpp>
#include <entt/entt.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>
#include <vector>

namespace {
    // A flat grid of `cells` x `cells` squares of side `cell` (two triangles each) in the y = 0 plane.
    auto grid(std::uint32_t cells, float cell, std::vector<glm::vec3> &positions, std::vector<std::uint32_t> &indices)
            -> void {
        for (std::uint32_t z = 0; z <= cells; ++z) {
            for (std::uint32_t x = 0; x <= cells; ++x) {
                positions.emplace_back(static_cast<float>(x) * cell, 0.0F, static_cast<float>(z) * cell);
            }
        }

        for (std::uint32_t z = 0; z < cells; ++z) {
            for (std::uint32_t x = 0; x < cells; ++x) {
                auto const a = z * (cells + 1) + x;
                auto const b = a + 1;
                auto const c = a + cells + 1;
                auto const d = c + 1;
                for (auto const index: {a, c, b, b, c, d}) {
                    indices.push_back(index);
                }
            }
        }
    }

    struct WorldFixture {
        BS::priority_thread_pool pool{2};
        entt::registry registry;
        PhysicsWorldSettings settings{};
        PhysicsWorld world{settings, pool, registry};
    };
}

TEST_SUITE("unit") {
    TEST_CASE("MeshCollider: a character stands on a triangle-mesh floor and a ray hits it") {
        std::vector<glm::vec3> positions;
        std::vector<std::uint32_t> indices;
        grid(10, 2.0F, positions, indices);
        auto const mesh = MeshCollider::build(std::move(positions), std::move(indices));
        REQUIRE(mesh != nullptr);
        CHECK(mesh->triangle_count() == 200);

        WorldFixture fixture;
        auto const floor = fixture.registry.create();
        fixture.world.add_static_mesh(floor, Components::Transform{.position = {-10.0F, 0.0F, -10.0F}}, mesh);

        auto const hit = fixture.world.raycast({0.0F, 5.0F, 0.0F}, {0.0F, -1.0F, 0.0F}, 20.0F);
        REQUIRE(hit.has_value());
        CHECK(hit->distance == doctest::Approx(5.0F));
        CHECK(hit->entity == floor);

        CharacterBody body{{0.0F, 1.0F, 0.0F}, 0.3F, 1.8F};

        for (int i = 0; i < 120; ++i) {
            body.step(fixture.world, {}, 1.0F / 60.0F);
        }

        CHECK(body.grounded());
        CHECK(body.position().y == doctest::Approx(0.0F).epsilon(0.05));

        // Walking off the edge of a 20 m floor falls.
        for (int i = 0; i < 600; ++i) {
            body.step(fixture.world, {.desired_velocity = {4.0F, 0.0F, 0.0F}}, 1.0F / 60.0F);
        }

        CHECK(body.position().y < -1.0F);
    }

    TEST_CASE("MeshCollider: a mesh over a million triangles is split into parts and still collides everywhere") {
        std::vector<glm::vec3> positions;
        std::vector<std::uint32_t> indices;
        grid(800, 0.5F, positions, indices);
        auto const mesh = MeshCollider::build(std::move(positions), std::move(indices));

        // 800 x 800 cells x 2 = 1.28M triangles, beyond what one Bullet part can number.
        CHECK(mesh->triangle_count() == 1'280'000);

        WorldFixture fixture;
        auto const floor = fixture.registry.create();
        fixture.world.add_static_mesh(floor, Components::Transform{}, mesh);

        for (auto const x: {1.0F, 100.0F, 399.0F}) {
            for (auto const z: {1.0F, 200.0F, 399.0F}) {
                auto const hit = fixture.world.raycast({x, 3.0F, z}, {0.0F, -1.0F, 0.0F}, 10.0F);
                REQUIRE(hit.has_value());
                CHECK(hit->distance == doctest::Approx(3.0F));
            }
        }
    }

    TEST_CASE("collect_model_triangles: applies node transforms and drops primitives under min_extent") {
        ModelCpuData model;
        auto &big = model.meshes.emplace_back().primitives.emplace_back();
        big.vertices = {{.position = {0, 0, 0}}, {.position = {1, 0, 0}}, {.position = {0, 0, 1}}};
        big.indices = {0, 1, 2};
        big.bounds = std::pair{glm::vec3{0, 0, 0}, glm::vec3{1, 0, 1}};

        auto &small = model.meshes.emplace_back().primitives.emplace_back();
        small.vertices = {{.position = {0, 0, 0}}, {.position = {0.01F, 0, 0}}, {.position = {0, 0, 0.01F}}};
        small.indices = {0, 1, 2};
        small.bounds = std::pair{glm::vec3{0, 0, 0}, glm::vec3{0.01F, 0, 0.01F}};

        model.nodes.resize(3);
        model.nodes[0].children = {1, 2};
        model.nodes[0].local_transform = glm::translate(glm::mat4{1.0F}, {10.0F, 0.0F, 0.0F});
        model.nodes[1].mesh_index = 0;
        model.nodes[2].mesh_index = 1;
        model.scene_roots = {0};

        std::vector<glm::vec3> positions;
        std::vector<std::uint32_t> indices;
        collect_model_triangles(model, 0.5F, positions, indices);

        REQUIRE(indices.size() == 3);
        REQUIRE(positions.size() == 3);
        CHECK(positions[0].x == doctest::Approx(10.0F));
        CHECK(positions[1].x == doctest::Approx(11.0F));

        positions.clear();
        indices.clear();
        collect_model_triangles(model, 0.0F, positions, indices);
        CHECK(indices.size() == 6);
    }
}
