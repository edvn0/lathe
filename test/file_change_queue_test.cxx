#include <doctest/doctest.h>

#include "assets/file_change_queue.hxx"

TEST_SUITE("unit") {
    TEST_CASE("a changed path is handed out once it has been quiet for the debounce time") {
        using namespace std::chrono_literals;

        auto queue = FileChangeQueue{100ms};
        auto const start = FileChangeQueue::Clock::now();

        queue.push("a.lua", start);

        CHECK(queue.drain(start + 50ms).empty());

        queue.push("a.lua", start + 80ms);

        CHECK(queue.drain(start + 150ms).empty());

        auto const ready = queue.drain(start + 200ms);

        REQUIRE(ready.size() == 1);
        CHECK(ready[0] == "a.lua");
        CHECK(queue.drain(start + 400ms).empty());
    }

    TEST_CASE("paths are debounced independently") {
        using namespace std::chrono_literals;

        auto queue = FileChangeQueue{100ms};
        auto const start = FileChangeQueue::Clock::now();

        queue.push("old.lua", start);
        queue.push("new.lua", start + 90ms);

        auto const ready = queue.drain(start + 110ms);

        REQUIRE(ready.size() == 1);
        CHECK(ready[0] == "old.lua");
    }
}
