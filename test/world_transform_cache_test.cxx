#include <doctest/doctest.h>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <random>
#include <vector>

#include <entt/entt.hpp>

#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {

    auto random_transform(std::mt19937 &engine) -> Components::Transform {
        std::uniform_real_distribution<float> unit{-1.0F, 1.0F};
        return Components::Transform{
                .position = glm::vec3{unit(engine), unit(engine), unit(engine)} * 5.0F,
                .rotation = glm::normalize(glm::quat{1.0F + unit(engine), unit(engine), unit(engine), unit(engine)}),
                .scale = glm::vec3{1.0F + 0.2F * unit(engine)},
        };
    }

    auto matches(glm::mat4 const &a, glm::mat4 const &b) -> bool {
        for (auto column = 0; column < 4; ++column) {
            for (auto row = 0; row < 4; ++row) {
                if (std::abs(a[column][row] - b[column][row]) > 1e-4F * (1.0F + std::abs(b[column][row]))) {
                    return false;
                }
            }
        }
        return true;
    }

    // Every entity with a Transform, asked through the cache and through get_world_transform.
    auto all_match(entt::registry &registry, systems::WorldTransformCache &cache) -> bool {
        cache.begin_frame();
        auto ok = true;
        for (auto const [entity, transform]: registry.view<Components::Transform>().each()) {
            ok = ok && matches(cache.get(registry, entity, transform),
                               systems::get_world_transform(registry, entity, transform));
        }
        return ok;
    }

}

TEST_SUITE("scene") {
    TEST_CASE("WorldTransformCache: entities without a parent get their own matrix") {
        auto engine = std::mt19937{1};
        entt::registry registry;
        for (auto i = 0; i < 50; ++i) {
            registry.emplace<Components::Transform>(registry.create(), random_transform(engine));
        }

        systems::WorldTransformCache cache;
        CHECK(all_match(registry, cache));
    }

    TEST_CASE("WorldTransformCache: a chain of parents matches the walk up it") {
        auto engine = std::mt19937{2};
        entt::registry registry;
        auto parent = entt::entity{entt::null};
        for (auto level = 0; level < 40; ++level) {
            auto const entity = registry.create();
            registry.emplace<Components::Transform>(entity, random_transform(engine));
            if (parent != entt::null) {
                registry.emplace<Components::Parent>(entity, Components::Parent{.entity = parent});
            }
            parent = entity;
        }

        systems::WorldTransformCache cache;
        CHECK(all_match(registry, cache));
    }

    TEST_CASE("WorldTransformCache: a random forest, with children on every level") {
        auto engine = std::mt19937{3};
        entt::registry registry;
        auto entities = std::vector<entt::entity>{};
        for (auto i = 0; i < 400; ++i) {
            auto const entity = registry.create();
            registry.emplace<Components::Transform>(entity, random_transform(engine));
            if (!entities.empty() && engine() % 4 != 0) {
                registry.emplace<Components::Parent>(
                        entity, Components::Parent{.entity = entities[engine() % entities.size()]});
            }
            entities.push_back(entity);
        }

        systems::WorldTransformCache cache;
        CHECK(all_match(registry, cache));
    }

    TEST_CASE("WorldTransformCache: a parent without a transform is passed through") {
        auto engine = std::mt19937{4};
        entt::registry registry;

        auto const grandparent = registry.create();
        registry.emplace<Components::Transform>(grandparent, random_transform(engine));

        auto const bare = registry.create();
        registry.emplace<Components::Parent>(bare, Components::Parent{.entity = grandparent});

        auto const child = registry.create();
        registry.emplace<Components::Transform>(child, random_transform(engine));
        registry.emplace<Components::Parent>(child, Components::Parent{.entity = bare});

        systems::WorldTransformCache cache;
        CHECK(all_match(registry, cache));
    }

    TEST_CASE("WorldTransformCache: a parent that no longer exists is ignored") {
        auto engine = std::mt19937{5};
        entt::registry registry;

        auto const gone = registry.create();
        auto const child = registry.create();
        registry.emplace<Components::Transform>(child, random_transform(engine));
        registry.emplace<Components::Parent>(child, Components::Parent{.entity = gone});
        registry.destroy(gone);

        systems::WorldTransformCache cache;
        CHECK(all_match(registry, cache));
    }

    TEST_CASE("WorldTransformCache: a cycle of parents ends instead of looping") {
        auto engine = std::mt19937{6};
        entt::registry registry;

        auto const a = registry.create();
        auto const b = registry.create();
        registry.emplace<Components::Transform>(a, random_transform(engine));
        registry.emplace<Components::Transform>(b, random_transform(engine));
        registry.emplace<Components::Parent>(a, Components::Parent{.entity = b});
        registry.emplace<Components::Parent>(b, Components::Parent{.entity = a});

        systems::WorldTransformCache cache;
        cache.begin_frame();
        auto const world = cache.get(registry, a, registry.get<Components::Transform>(a));
        CHECK(std::isfinite(world[3][0]));
    }

    TEST_CASE("WorldTransformCache: a new frame sees transforms that changed") {
        auto engine = std::mt19937{7};
        entt::registry registry;

        auto const parent = registry.create();
        registry.emplace<Components::Transform>(parent, random_transform(engine));
        auto const child = registry.create();
        registry.emplace<Components::Transform>(child, random_transform(engine));
        registry.emplace<Components::Parent>(child, Components::Parent{.entity = parent});

        systems::WorldTransformCache cache;
        CHECK(all_match(registry, cache));

        registry.get<Components::Transform>(parent).position += glm::vec3{3.0F, 0.0F, -2.0F};
        CHECK(all_match(registry, cache));
    }

    TEST_CASE("Transform::matrix is translate * rotate * scale") {
        auto engine = std::mt19937{8};
        for (auto i = 0; i < 200; ++i) {
            auto const transform = random_transform(engine);
            auto const composed = glm::translate(glm::mat4{1.0F}, transform.position) *
                                  glm::mat4_cast(transform.rotation) * glm::scale(glm::mat4{1.0F}, transform.scale);
            CHECK(matches(transform.matrix(), composed));
        }
    }

    TEST_CASE("Transform::matrix keeps a non-uniform scale along the rotated axes") {
        auto const transform = Components::Transform{
                .position = glm::vec3{1.0F, 2.0F, 3.0F},
                .rotation = glm::angleAxis(0.7F, glm::normalize(glm::vec3{1.0F, 1.0F, 0.0F})),
                .scale = glm::vec3{2.0F, 0.5F, 3.0F},
        };
        auto const composed = glm::translate(glm::mat4{1.0F}, transform.position) *
                              glm::mat4_cast(transform.rotation) * glm::scale(glm::mat4{1.0F}, transform.scale);
        CHECK(matches(transform.matrix(), composed));
    }
}
