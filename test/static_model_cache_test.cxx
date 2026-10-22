#include <doctest/doctest.h>

#include <entt/entt.hpp>

#include <algorithm>
#include <vector>

#include "rendering/static_model_cache.hxx"
#include "scene/components.hxx"

namespace {
    constexpr ModelHandle cube{.index = 1, .generation = 1};
    constexpr ModelHandle sphere{.index = 2, .generation = 1};

    auto const any_model = [](ModelHandle) { return true; };

    auto add_model(entt::registry &registry, ModelHandle model, glm::vec3 position, bool is_static)
            -> entt::entity {
        auto const entity = registry.create();
        registry.emplace<Components::Transform>(entity, Components::Transform{.position = position});
        registry.emplace<Components::Model>(entity, Components::Model{.model = model});
        if (is_static) {
            registry.emplace<Components::Static>(entity);
        }
        return entity;
    }
}

TEST_SUITE("unit") {
    TEST_CASE("StaticModelCache: static models are batched by model, dynamic ones are not") {
        entt::registry registry;
        auto const first = add_model(registry, cube, {1.0F, 0.0F, 0.0F}, true);
        auto const second = add_model(registry, cube, {2.0F, 0.0F, 0.0F}, true);
        auto const other = add_model(registry, sphere, {3.0F, 0.0F, 0.0F}, true);
        auto const moving = add_model(registry, cube, {4.0F, 0.0F, 0.0F}, false);

        StaticModelCache cache;
        cache.update(registry, 1, {}, any_model);

        REQUIRE(cache.batches().size() == 2);
        CHECK(cache.entity_count() == 3);
        CHECK(cache.contains(first));
        CHECK(cache.contains(second));
        CHECK(cache.contains(other));
        CHECK_FALSE(cache.contains(moving));

        auto const batch = std::ranges::find(cache.batches(), cube, &StaticModelCache::Batch::model);
        REQUIRE(batch != cache.batches().end());
        REQUIRE(batch->transforms.size() == 2);

        auto positions = std::vector{batch->transforms[0][3].x, batch->transforms[1][3].x};
        std::ranges::sort(positions);
        CHECK(positions == std::vector{1.0F, 2.0F});
    }

    TEST_CASE("StaticModelCache: a child of a static parent is static, with the parent's transform") {
        entt::registry registry;
        auto const parent = registry.create();
        registry.emplace<Components::Transform>(parent, Components::Transform{.position = {10.0F, 0.0F, 0.0F}});
        registry.emplace<Components::Static>(parent);

        auto const child = add_model(registry, cube, {1.0F, 0.0F, 0.0F}, false);
        registry.emplace<Components::Parent>(child, Components::Parent{.entity = parent});

        StaticModelCache cache;
        cache.update(registry, 1, {}, any_model);

        REQUIRE(cache.batches().size() == 1);
        CHECK(cache.contains(child));
        CHECK(cache.batches().front().transforms.front()[3].x == 11.0F);
    }

    TEST_CASE("StaticModelCache: the bake is kept until the revision changes, then gets a new batch revision") {
        entt::registry registry;
        auto const entity = add_model(registry, cube, {1.0F, 0.0F, 0.0F}, true);

        StaticModelCache cache;
        cache.update(registry, 1, {}, any_model);
        auto const revision = cache.batches().front().revision;
        CHECK(revision != 0);

        registry.get<Components::Transform>(entity).position.x = 5.0F;
        cache.update(registry, 1, {}, any_model);
        CHECK(cache.batches().front().revision == revision);
        CHECK(cache.batches().front().transforms.front()[3].x == 1.0F);

        cache.update(registry, 2, {}, any_model);
        CHECK(cache.batches().front().revision != revision);
        CHECK(cache.batches().front().transforms.front()[3].x == 5.0F);
    }

    TEST_CASE("StaticModelCache: excluded, outlined, slot-overridden and missing-model entities stay out") {
        entt::registry registry;
        auto const selected = add_model(registry, cube, {}, true);
        auto const outlined = add_model(registry, cube, {}, true);
        registry.emplace<Components::Outlined>(outlined);
        auto const slotted = add_model(registry, cube, {}, true);
        registry.emplace<Components::MaterialOverride>(
                slotted, Components::MaterialOverride{.slots = {MaterialSlotOverride{}}});
        auto const missing = add_model(registry, sphere, {}, true);
        auto const kept = add_model(registry, cube, {}, true);

        std::vector<entt::entity> const excluded{selected};
        StaticModelCache cache;
        cache.update(registry, 1, excluded, [](ModelHandle model) { return model == cube; });

        CHECK(cache.entity_count() == 1);
        CHECK(cache.contains(kept));
        CHECK_FALSE(cache.contains(selected));
        CHECK_FALSE(cache.contains(outlined));
        CHECK_FALSE(cache.contains(slotted));
        CHECK_FALSE(cache.contains(missing));
    }

    TEST_CASE("StaticModelCache: changing the excluded entities rebakes, clear forgets everything") {
        entt::registry registry;
        auto const entity = add_model(registry, cube, {}, true);

        StaticModelCache cache;
        cache.update(registry, 1, {}, any_model);
        CHECK(cache.contains(entity));

        std::vector<entt::entity> const excluded{entity};
        cache.update(registry, 1, excluded, any_model);
        CHECK_FALSE(cache.contains(entity));
        CHECK(cache.batches().empty());

        cache.update(registry, 1, {}, any_model);
        CHECK(cache.contains(entity));
        cache.clear();
        CHECK_FALSE(cache.contains(entity));
        CHECK(cache.entity_count() == 0);
    }
}
