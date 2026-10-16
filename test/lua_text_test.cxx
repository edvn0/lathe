#include <doctest/doctest.h>

#include <cmath>
#include <optional>
#include <string>

#include "scene/components.hxx"
#include "scripting/lua_runtime.hxx"
#include "scripting/lua_text.hxx"

namespace {
    auto text = Components::Text{};

    auto parse(lua_State *state) -> int {
        if (auto const problem = read_text_table(state, 1, text)) {
            lua_pushlstring(state, problem->data(), problem->size());
        } else {
            lua_pushnil(state);
        }

        return 1;
    }

    // Runs `table_source` through read_text_table; the problem, or nullopt when it was accepted.
    auto apply(LuaRuntime &runtime, std::string const &table_source) -> std::optional<std::string> {
        auto *const state = runtime.state();
        auto const top = lua_gettop(state);

        lua_pushcfunction(state, &parse);
        lua_setglobal(state, "parse");

        auto const chunk = "return parse(" + table_source + ")";
        REQUIRE(luaL_loadstring(state, chunk.c_str()) == LUA_OK);
        REQUIRE(lua_pcall(state, 0, 1, 0) == LUA_OK);

        auto result = lua_isnil(state, -1) ? std::nullopt : std::optional<std::string>{lua_tostring(state, -1)};

        lua_settop(state, top);
        CHECK(lua_gettop(state) == top);

        return result;
    }

    auto make_runtime() -> LuaRuntime {
        auto runtime = LuaRuntime::create();
        REQUIRE(runtime.has_value());
        return std::move(*runtime);
    }
} // namespace

TEST_SUITE("unit") {
    TEST_CASE("a text table sets the fields it names and keeps the rest") {
        auto runtime = make_runtime();
        text = Components::Text{};

        CHECK_FALSE(apply(runtime, R"({text = "NEON\nBAR", font = 2, size = 1.5, color = {1, 0.25, 0.5}, emissive = 8,
                                       centred = false, visible = false})")
                            .has_value());
        CHECK(text.text == "NEON\nBAR");
        CHECK(text.font == 2);
        CHECK(text.size == doctest::Approx(1.5F));
        CHECK(text.colour.y == doctest::Approx(0.25F));
        CHECK(text.colour.w == doctest::Approx(1.0F));
        CHECK(text.emissive == doctest::Approx(8.0F));
        CHECK_FALSE(text.centred);
        CHECK_FALSE(text.visible);

        // A second table changes only what it names.
        CHECK_FALSE(apply(runtime, "{emissive = 3, colour = {0, 1, 0, 0.5}}").has_value());
        CHECK(text.text == "NEON\nBAR");
        CHECK(text.emissive == doctest::Approx(3.0F));
        CHECK(text.colour.y == doctest::Approx(1.0F));
        CHECK(text.colour.w == doctest::Approx(0.5F));
        CHECK(text.size == doctest::Approx(1.5F));

        CHECK_FALSE(apply(runtime, "{centered = true}").has_value());
        CHECK(text.centred);
        CHECK_FALSE(apply(runtime, "{}").has_value());
    }

    TEST_CASE("a text table with a bad field is refused whole") {
        auto runtime = make_runtime();
        text = Components::Text{.text = "keep", .size = 2.0F};

        auto const refused = [&](std::string const &table, std::string const &mentions) {
            auto const problem = apply(runtime, table);
            CHECK(problem.has_value());
            if (problem) {
                CHECK(problem->find(mentions) != std::string::npos);
            }
            // Even the valid fields before the bad one did not apply.
            CHECK(text.text == "keep");
            CHECK(text.size == doctest::Approx(2.0F));
            CHECK(text.emissive == doctest::Approx(1.0F));
        };

        refused("{emissive = 5, bogus = 1}", "bogus");
        refused("{emissive = 5, size = 0}", "size");
        refused("{emissive = 5, size = -1}", "size");
        refused("{emissive = 5, size = 1e9}", "size");
        refused("{emissive = 5, size = 0/0}", "size");
        refused("{emissive = 5, size = 'big'}", "size");
        refused("{size = 1, emissive = -1}", "emissive");
        refused("{size = 1, emissive = 1e6}", "emissive");
        refused("{size = 1, emissive = math.huge}", "emissive");
        refused("{size = 1, text = 5}", "text");
        refused("{size = 1, font = -1}", "font");
        refused("{size = 1, font = 1.5}", "font");
        refused("{size = 1, color = {1, 2}}", "color");
        refused("{size = 1, color = {1, 2, 3, 4, 5}}", "color");
        refused("{size = 1, color = {1, 'x', 3}}", "color");
        refused("{size = 1, color = {-1, 0, 0}}", "color");
        refused("{size = 1, color = {1, 0, 0, 2}}", "color");
        refused("{size = 1, color = 'red'}", "color");
        refused("{size = 1, centred = 1}", "centred");
        refused("{size = 1, visible = 'yes'}", "visible");
        refused("{[1] = 'positional'}", "strings");
    }

    TEST_CASE("text stops at the length the scene format holds") {
        auto runtime = make_runtime();
        text = Components::Text{};

        CHECK_FALSE(apply(runtime, "{text = string.rep('x', 64 * 1024)}").has_value());
        CHECK(text.text.size() == 64U * 1024U);
        CHECK(apply(runtime, "{text = string.rep('x', 64 * 1024 + 1)}").has_value());
        CHECK(text.text.size() == 64U * 1024U);
        CHECK(apply(runtime, "42").has_value());
    }
}
