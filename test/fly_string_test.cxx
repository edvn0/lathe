#include <doctest/doctest.h>

#include "core/fly_string.hxx"

#include <atomic>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <BS_thread_pool.hpp>

TEST_SUITE("unit") {
    TEST_CASE("FlyString: default constructed value is empty") {
        auto const value = FlyString{};

        CHECK(value.empty());
        CHECK(value.view().empty());
        CHECK(value.view() == "");
        CHECK(std::string_view{value.c_str()} == "");
    }

    TEST_CASE("FlyString: stores its value") {
        auto const value = FlyString{"hello world"};

        CHECK_FALSE(value.empty());
        CHECK(value.view() == "hello world");
        CHECK(std::string_view{value.c_str()} == "hello world");
    }

    TEST_CASE("FlyString: identical strings compare equal") {
        auto const a = FlyString{"hello"};
        auto const b = FlyString{"hello"};

        CHECK(a == b);
    }

    TEST_CASE("FlyString: different strings compare unequal") {
        auto const a = FlyString{"hello"};
        auto const b = FlyString{"world"};

        CHECK_FALSE(a == b);
    }

    TEST_CASE("FlyString: identical strings share interned storage") {
        auto const a = FlyString{"hello"};
        auto const b = FlyString{"hello"};

        CHECK(a.c_str() == b.c_str());
    }

    TEST_CASE("FlyString: does not depend on source lifetime") {
        auto value = FlyString{};

        {
            auto source = std::string{"temporary string"};
            value = FlyString{source};

            CHECK(value.view() == source);
        }

        CHECK(value.view() == "temporary string");
    }

    TEST_CASE("FlyString: copy preserves interned identity") {
        auto const original = FlyString{"hello"};
        auto const copy = original;

        CHECK(copy == original);
        CHECK(copy.c_str() == original.c_str());
    }

    TEST_CASE("FlyString: move preserves interned identity") {
        auto original = FlyString{"hello"};
        auto const *storage = original.c_str();

        auto const moved = std::move(original);

        CHECK(moved.view() == "hello");
        CHECK(moved.c_str() == storage);
    }

    TEST_CASE("FlyString: embedded null characters are preserved") {
        constexpr char data[] = {'a', '\0', 'b', 'c'};

        auto const value = FlyString{std::string_view{data, sizeof(data)}};

        REQUIRE(value.view().size() == sizeof(data));

        CHECK(value.view()[0] == 'a');
        CHECK(value.view()[1] == '\0');
        CHECK(value.view()[2] == 'b');
        CHECK(value.view()[3] == 'c');
    }

    TEST_CASE("FlyString: default and explicitly empty values have current semantics") {
        auto const default_value = FlyString{};
        auto const interned_empty = FlyString{""};

        CHECK(default_value.empty());
        CHECK(interned_empty.empty());

        CHECK(default_value.view() == interned_empty.view());

        // This documents the current implementation.
        //
        // Remove/change this test if FlyString{""} is changed to use
        // nullptr as the canonical representation for an empty string.
        CHECK_FALSE(default_value == interned_empty);
    }
}

TEST_SUITE("smoke") {
    TEST_CASE("FlyString: interned storage survives pool growth") {
        auto const original = FlyString{"persistent value"};
        auto const *storage = original.c_str();

        constexpr auto count = 100'000;

        for (auto i = 0; i < count; ++i) {
            [[maybe_unused]] auto const ignored = FlyString{"unique_string_" + std::to_string(i)};
        }

        CHECK(original.view() == "persistent value");
        CHECK(original.c_str() == storage);
    }
    TEST_CASE("FlyString: concurrent interning is stable") {
        constexpr auto thread_count = 16;
        constexpr auto iterations = 100'000;

        BS::thread_pool pool{thread_count};

        std::vector<char const *> shared_pointers(thread_count);
        std::atomic_bool failed = false;

        auto futures = pool.submit_loop(
                std::size_t{0}, std::size_t{thread_count},
                [&](std::size_t thread_index) {
                    char const *expected_shared = nullptr;

                    for (auto i = 0; i < iterations; ++i) {
                        auto const shared = FlyString{"concurrently interned string"};

                        auto const unique =
                                FlyString{"thread_" + std::to_string(thread_index) + "_value_" + std::to_string(i)};

                        if (expected_shared == nullptr) {
                            expected_shared = shared.c_str();
                        } else if (shared.c_str() != expected_shared) {
                            failed.store(true, std::memory_order_relaxed);
                        }

                        if (unique.empty()) {
                            failed.store(true, std::memory_order_relaxed);
                        }
                    }

                    shared_pointers[thread_index] = expected_shared;
                },
                thread_count);

        futures.wait();

        CHECK_FALSE(failed.load(std::memory_order_relaxed));

        REQUIRE_FALSE(shared_pointers.empty());

        for (auto const *pointer: shared_pointers) {
            CHECK(pointer == shared_pointers.front());
        }
    }

    TEST_CASE("FlyString: large number of duplicate strings are deduplicated") {
        constexpr auto unique_count = 100'000;
        constexpr auto repetitions = 100;

        std::vector<FlyString> strings;
        strings.reserve(unique_count * repetitions);

        for (auto repetition = 0; repetition < repetitions; ++repetition) {
            for (auto i = 0; i < unique_count; ++i) {
                strings.emplace_back("entity_" + std::to_string(i));
            }
        }

        REQUIRE(strings.size() == unique_count * repetitions);

        for (auto i = 0; i < unique_count; ++i) {
            auto const &first = strings[i];

            for (auto repetition = 1; repetition < repetitions; ++repetition) {
                auto const &other = strings[repetition * unique_count + i];

                CHECK(first == other);
                CHECK(first.c_str() == other.c_str());
            }
        }
    }
}
