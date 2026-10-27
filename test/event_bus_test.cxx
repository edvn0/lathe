#include <doctest/doctest.h>

#include <vector>

#include "scene/input_events.hxx"

TEST_CASE("EventBus: priority order, consumption, unsubscription and metrics") {
    auto bus = EventBus{};
    auto order = std::vector<int>{};

    auto late = bus.subscribe<KeyPressedEvent>([&](auto const &) { order.push_back(2); return false; }, 10);
    auto early = bus.subscribe<KeyPressedEvent>([&](auto const &) { order.push_back(1); return false; }, 0);

    CHECK_FALSE(bus.publish(KeyPressedEvent{.key = 1}));
    CHECK(order == std::vector{1, 2});

    {
        auto eater = bus.subscribe<KeyPressedEvent>([&](auto const &) { order.push_back(0); return true; }, -1);
        order.clear();
        CHECK(bus.publish(KeyPressedEvent{.key = 1}));
        CHECK(order == std::vector{0});
    }

    order.clear();
    bus.publish(KeyPressedEvent{.key = 1});
    CHECK(order == std::vector{1, 2});

    auto const metrics = bus.metrics();
    REQUIRE(metrics.size() == 1);
    CHECK(metrics[0].name == "KeyPressed");
    CHECK(metrics[0].published == 3);
    CHECK(metrics[0].consumed == 1);
    CHECK(metrics[0].subscribers == 2);
}
