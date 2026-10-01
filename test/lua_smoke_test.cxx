#include <doctest/doctest.h>

#include <cstddef>
#include <cstdlib>

#include <sol/sol.hpp>

// Temporary (plan step 0b): checks that Lua compiled as C and sol2 build and run under -fno-exceptions/-fno-rtti.
// Superseded by the ScriptEngine tests.
namespace {
    auto smoke_alloc(void * /*user_data*/, void *block, std::size_t /*old_size*/, std::size_t new_size) noexcept
            -> void * {
        if (new_size == 0) {
            std::free(block);
            return nullptr;
        }
        return std::realloc(block, new_size);
    }
} // namespace

TEST_SUITE("unit") {
    TEST_CASE("Lua smoke: sol2 runs a chunk on a custom-allocator state") {
        lua_State *state = lua_newstate(&smoke_alloc, nullptr);
        REQUIRE(state != nullptr);

        {
            sol::state_view lua{state};
            auto const result = lua.safe_script("return 1 + 1", &sol::script_pass_on_error);
            REQUIRE(result.valid());
            CHECK(result.get<int>() == 2);
        }

        lua_close(state);
    }
}
