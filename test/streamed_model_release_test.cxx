#include <doctest/doctest.h>

#include <entt/entt.hpp>

#include <algorithm>
#include <vector>

#include "scene/components.hxx"

namespace {

    struct ReleaseRecorder {
        std::vector<ModelHandle> released;

        auto on_model_destroyed(entt::registry &registry, entt::entity entity) -> void {
            if (auto const model = Components::model_released_by_model_destroy(registry, entity)) {
                released.push_back(*model);
            }
        }

        auto on_tag_destroyed(entt::registry &registry, entt::entity entity) -> void {
            if (auto const model = Components::model_released_by_tag_destroy(registry, entity)) {
                released.push_back(*model);
            }
        }

        auto connect(entt::registry &registry) -> void {
            registry.on_destroy<Components::Model>().connect<&ReleaseRecorder::on_model_destroyed>(*this);
            registry.on_destroy<Components::StreamedModelTag>().connect<&ReleaseRecorder::on_tag_destroyed>(*this);
        }
    };

    constexpr ModelHandle owned_model{.index = 2, .generation = 1};
    constexpr ModelHandle shared_model{.index = 3, .generation = 1};

    auto make_owned(entt::registry &registry, ModelHandle model = owned_model) -> entt::entity {
        auto const entity = registry.create();
        registry.emplace<Components::Model>(entity, Components::Model{.model = model});
        registry.emplace<Components::StreamedModelTag>(entity);
        return entity;
    }

    auto make_shared(entt::registry &registry) -> entt::entity {
        auto const entity = registry.create();
        registry.emplace<Components::Model>(entity, Components::Model{.model = shared_model});
        return entity;
    }

    auto make_registry(bool tag_pool_first) -> entt::registry {
        entt::registry registry;

        if (tag_pool_first) {
            static_cast<void>(registry.storage<Components::StreamedModelTag>());
            static_cast<void>(registry.storage<Components::Model>());
        } else {
            static_cast<void>(registry.storage<Components::Model>());
            static_cast<void>(registry.storage<Components::StreamedModelTag>());
        }

        return registry;
    }

}

TEST_SUITE("StreamedModelTag release") {
    TEST_CASE("destroying an owning entity releases its model once") {
        for (bool const tag_pool_first: {true, false}) {
            CAPTURE(tag_pool_first);

            auto registry = make_registry(tag_pool_first);
            ReleaseRecorder recorder;
            recorder.connect(registry);

            registry.destroy(make_owned(registry));

            CHECK(recorder.released == std::vector<ModelHandle>{owned_model});
        }
    }

    TEST_CASE("an entity without the tag releases nothing") {
        for (bool const tag_pool_first: {true, false}) {
            CAPTURE(tag_pool_first);

            auto registry = make_registry(tag_pool_first);
            ReleaseRecorder recorder;
            recorder.connect(registry);

            registry.destroy(make_shared(registry));

            CHECK(recorder.released.empty());
        }
    }

    TEST_CASE("removing the tag, then the model, releases once") {
        auto registry = make_registry(false);
        ReleaseRecorder recorder;
        recorder.connect(registry);

        auto const entity = make_owned(registry);

        registry.remove<Components::StreamedModelTag>(entity);
        CHECK(recorder.released == std::vector<ModelHandle>{owned_model});

        registry.remove<Components::Model>(entity);
        CHECK(recorder.released == std::vector<ModelHandle>{owned_model});
    }

    TEST_CASE("removing the model, then the tag, releases once") {
        auto registry = make_registry(false);
        ReleaseRecorder recorder;
        recorder.connect(registry);

        auto const entity = make_owned(registry);

        registry.remove<Components::Model>(entity);
        CHECK(recorder.released == std::vector<ModelHandle>{owned_model});

        registry.remove<Components::StreamedModelTag>(entity);
        CHECK(recorder.released == std::vector<ModelHandle>{owned_model});
    }

    TEST_CASE("clearing the registry releases each owning entity once") {
        for (bool const tag_pool_first: {true, false}) {
            CAPTURE(tag_pool_first);

            auto registry = make_registry(tag_pool_first);
            ReleaseRecorder recorder;
            recorder.connect(registry);

            static_cast<void>(make_owned(registry, ModelHandle{.index = 4, .generation = 1}));
            static_cast<void>(make_shared(registry));
            static_cast<void>(make_owned(registry, ModelHandle{.index = 5, .generation = 1}));

            registry.clear();

            CHECK(recorder.released.size() == 2);
            CHECK(std::ranges::count(recorder.released, ModelHandle{.index = 4, .generation = 1}) == 1);
            CHECK(std::ranges::count(recorder.released, ModelHandle{.index = 5, .generation = 1}) == 1);
        }
    }

    TEST_CASE("clearing only the tags, as ~Scene does, releases while the models remain") {
        auto registry = make_registry(false);
        ReleaseRecorder recorder;
        recorder.connect(registry);

        static_cast<void>(make_owned(registry));
        static_cast<void>(make_shared(registry));

        registry.clear<Components::StreamedModelTag>();

        CHECK(recorder.released == std::vector<ModelHandle>{owned_model});
    }

    TEST_CASE("replacing the tag is not a removal") {
        auto registry = make_registry(false);
        ReleaseRecorder recorder;
        recorder.connect(registry);

        auto const entity = make_owned(registry);

        registry.emplace_or_replace<Components::StreamedModelTag>(entity);
        registry.patch<Components::Model>(entity, [](Components::Model &model) { model.model = shared_model; });

        CHECK(recorder.released.empty());
    }
}
