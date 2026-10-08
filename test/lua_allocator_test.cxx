#include <doctest/doctest.h>

#include <cstddef>

#include "scripting/lua_allocator.hxx"

namespace {
    constexpr std::size_t string_type_tag = 4;

    [[nodiscard]] auto allocate(LuaMemoryBudget &budget, std::size_t size) -> void * {
        return lua_budget_alloc(&budget, nullptr, string_type_tag, size);
    }

    auto release(LuaMemoryBudget &budget, void *block, std::size_t size) -> void {
        static_cast<void>(lua_budget_alloc(&budget, block, size, 0));
    }
}

TEST_SUITE("unit") {
    TEST_CASE("LuaMemoryBudget: a new block counts its size, not the type tag") {
        LuaMemoryBudget budget{};

        void *block = allocate(budget, 100);
        REQUIRE(block != nullptr);
        CHECK(budget.used_bytes == 100);

        release(budget, block, 100);
        CHECK(budget.used_bytes == 0);
    }

    TEST_CASE("LuaMemoryBudget: growth past the limit fails and leaves the count alone") {
        LuaMemoryBudget budget{.limit_bytes = 128};

        void *block = allocate(budget, 100);
        REQUIRE(block != nullptr);

        CHECK(lua_budget_alloc(&budget, block, 100, 200) == nullptr);
        CHECK(budget.limit_hit);
        CHECK(budget.used_bytes == 100);

        CHECK(allocate(budget, 64) == nullptr);
        CHECK(budget.used_bytes == 100);

        release(budget, block, 100);
    }

    TEST_CASE("LuaMemoryBudget: shrinking succeeds and lowers the count; freeing subtracts the old size") {
        LuaMemoryBudget budget{.limit_bytes = 128};

        void *block = allocate(budget, 100);
        REQUIRE(block != nullptr);

        budget.limit_bytes = 10;
        void *shrunk = lua_budget_alloc(&budget, block, 100, 50);
        REQUIRE(shrunk != nullptr);
        CHECK(budget.used_bytes == 50);
        CHECK_FALSE(budget.limit_hit);

        release(budget, shrunk, 50);
        CHECK(budget.used_bytes == 0);
    }

    TEST_CASE("LuaMemoryBudget: the error-handler headroom applies only while the handler runs") {
        LuaMemoryBudget budget{.limit_bytes = 100, .error_handler_headroom_bytes = 100};

        CHECK(allocate(budget, 150) == nullptr);
        CHECK(budget.limit_hit);

        budget.in_error_handler = true;
        void *block = allocate(budget, 150);
        REQUIRE(block != nullptr);
        CHECK(budget.used_bytes == 150);

        budget.in_error_handler = false;
        CHECK(allocate(budget, 1) == nullptr);

        release(budget, block, 150);
    }

    TEST_CASE("LuaMemoryBudget: peak_bytes tracks the maximum and reset_run_flags restarts it") {
        LuaMemoryBudget budget{};

        void *first = allocate(budget, 300);
        void *second = allocate(budget, 200);
        REQUIRE(first != nullptr);
        REQUIRE(second != nullptr);
        CHECK(budget.peak_bytes == 500);

        release(budget, first, 300);
        CHECK(budget.used_bytes == 200);
        CHECK(budget.peak_bytes == 500);

        budget.limit_hit = true;
        budget.in_error_handler = true;
        budget.reset_run_flags();
        CHECK(budget.peak_bytes == 200);
        CHECK_FALSE(budget.limit_hit);
        CHECK_FALSE(budget.in_error_handler);

        release(budget, second, 200);
    }
}
