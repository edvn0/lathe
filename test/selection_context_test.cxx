#include <doctest/doctest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

#include "scene/selection_context.hxx"

namespace {

    [[nodiscard]] auto entity(std::uint32_t id) -> entt::entity { return entt::entity{id}; }

}

TEST_CASE("select replaces the selection and makes the entity primary") {
    SelectionContext selection;

    selection.add(entity(1));
    selection.add(entity(2));
    selection.select(entity(3));

    auto const snapshot = selection.snapshot();
    CHECK(snapshot.entities == std::vector{entity(3)});
    CHECK(selection.primary() == entity(3));

    selection.select(entt::null);
    CHECK(selection.empty());
    CHECK(selection.primary() == entt::entity{entt::null});
}

TEST_CASE("The primary falls back to the latest remaining entity") {
    SelectionContext selection;

    selection.add(entity(1));
    selection.add(entity(2));
    selection.add(entity(3));
    CHECK(selection.primary() == entity(3));

    selection.remove(entity(3));
    CHECK(selection.primary() == entity(2));

    selection.toggle(entity(1));
    CHECK(selection.primary() == entity(2));
    CHECK(selection.size() == 1);
}

TEST_CASE("modify publishes a batch as one version") {
    SelectionContext selection;
    auto const before = selection.version();

    selection.modify([](SelectionContext::Transaction &transaction) {
        transaction.set(entity(1), true);
        transaction.set(entity(2), true);
        transaction.set(entity(3), true);
        transaction.set_primary(entity(1));
    });

    CHECK(selection.version() == before + 1);
    CHECK(selection.size() == 3);
    CHECK(selection.primary() == entity(1));

    selection.modify([](SelectionContext::Transaction &transaction) { transaction.set(entity(2), true); });
    CHECK(selection.version() == before + 2);

    selection.modify([](SelectionContext::Transaction &transaction) { transaction.set(entity(2), true); });
    CHECK(selection.version() == before + 2);
}

TEST_CASE("assign keeps order and takes an explicit primary") {
    SelectionContext selection;

    std::array const entities{entity(4), entity(5), entity(6)};
    selection.assign(entities, entity(5));

    CHECK(selection.snapshot().entities == std::vector<entt::entity>{entities.begin(), entities.end()});
    CHECK(selection.primary() == entity(5));

    selection.assign(entities);
    CHECK(selection.primary() == entity(6));
}

TEST_CASE("retain_if drops rejected entities") {
    SelectionContext selection;
    std::array const entities{entity(1), entity(2), entity(3), entity(4)};
    selection.assign(entities);

    selection.retain_if([](entt::entity e) { return (entt::to_integral(e) % 2) == 0; });

    CHECK(selection.snapshot().entities == std::vector{entity(2), entity(4)});
    CHECK(selection.primary() == entity(4));
}

TEST_CASE("Readers never observe a half-applied batch") {
    SelectionContext selection;
    std::array const first{entity(1), entity(2), entity(3)};
    std::array const second{entity(7), entity(8), entity(9)};
    std::atomic<bool> stop{false};
    std::atomic<bool> torn{false};

    std::thread reader{[&] {
        while (!stop.load()) {
            auto const snapshot = selection.snapshot();
            bool const is_first = snapshot.entities == std::vector<entt::entity>{first.begin(), first.end()};
            bool const is_second = snapshot.entities == std::vector<entt::entity>{second.begin(), second.end()};
            if (!snapshot.entities.empty() && !is_first && !is_second) {
                torn.store(true);
            }
        }
    }};

    for (int i = 0; i < 2000; ++i) {
        selection.assign(i % 2 == 0 ? std::span<entt::entity const>{first} : std::span<entt::entity const>{second});
    }

    stop.store(true);
    reader.join();

    CHECK_FALSE(torn.load());
}
