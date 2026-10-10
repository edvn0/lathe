#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

#include "scene/components.hxx"
#include "scripting/lua_particles.hxx"
#include "scripting/lua_runtime.hxx"

namespace {
    auto emitter = Components::ParticleEmitter{};

    auto parse(lua_State *state) -> int {
        if (auto const problem = read_particle_table(state, 1, emitter)) {
            lua_pushlstring(state, problem->data(), problem->size());
        } else {
            lua_pushnil(state);
        }

        return 1;
    }

    // Runs `table_source` through read_particle_table; the problem, or nullopt when it was accepted.
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

        return result;
    }

    auto make_runtime() -> LuaRuntime {
        auto runtime = LuaRuntime::create();

        REQUIRE(runtime.has_value());

        return std::move(*runtime);
    }
}

TEST_SUITE("unit") {
    TEST_CASE("a particle table sets the fields it names and keeps the rest") {
        auto runtime = make_runtime();
        emitter = Components::ParticleEmitter{};

        CHECK_FALSE(apply(runtime, R"({count = 2000, rate = 50.5, lifetime = 1.5, gravity = {0, -3, 1}, shape = "sphere",
                                       shape_size = 2, speed = 4, emitting = false, colour_end = {1, 0.5, 0.25}})")
                            .has_value());

        CHECK(emitter.count == 2000);
        CHECK(emitter.rate == doctest::Approx(50.5F));
        CHECK(emitter.lifetime == doctest::Approx(1.5F));
        CHECK(emitter.gravity.y == doctest::Approx(-3.0F));
        CHECK(emitter.gravity.z == doctest::Approx(1.0F));
        CHECK(emitter.shape == Components::ParticleShape::sphere);
        CHECK_FALSE(emitter.emitting);
        // A colour without alpha keeps the alpha it had.
        CHECK(emitter.colour_end.y == doctest::Approx(0.5F));
        CHECK(emitter.colour_end.w == doctest::Approx(Components::ParticleEmitter{}.colour_end.w));

        CHECK_FALSE(apply(runtime, "{rate = 1}").has_value());
        CHECK(emitter.rate == doctest::Approx(1.0F));
        CHECK(emitter.count == 2000);
        CHECK(emitter.shape == Components::ParticleShape::sphere);
    }

    TEST_CASE("a material key is accepted and left to the caller") {
        auto runtime = make_runtime();
        emitter = Components::ParticleEmitter{};

        CHECK_FALSE(apply(runtime, "{material = {}}").has_value());
    }

    TEST_CASE("bad particle tables are rejected with a reason and leave the emitter untouched") {
        auto runtime = make_runtime();
        emitter = Components::ParticleEmitter{};
        auto const before = emitter;

        struct Case {
            char const *table;
            char const *mentions;
        };

        constexpr auto cases = std::array{
                Case{"{bogus = 1}", "bogus"},
                Case{"{count = 0}", "count"},
                Case{"{count = -3}", "count"},
                Case{"{count = 2000000}", "count"},
                Case{"{count = 10.5}", "count"},
                Case{"{count = '10'}", "count"},
                Case{"{rate = 0/0}", "rate"},
                Case{"{rate = 1/0}", "rate"},
                Case{"{rate = -1}", "rate"},
                Case{"{lifetime = 0}", "lifetime"},
                Case{"{speed_variance = 2}", "speed_variance"},
                Case{"{cone_degrees = 181}", "cone_degrees"},
                Case{"{shape = 'blob'}", "shape"},
                Case{"{shape = 3}", "shape"},
                Case{"{emitting = 1}", "emitting"},
                Case{"{gravity = {0, 1}}", "gravity"},
                Case{"{gravity = {0, 1, 2, 3}}", "gravity"},
                Case{"{gravity = {0, 'x', 2}}", "gravity"},
                Case{"{gravity = 5}", "gravity"},
                Case{"{colour_start = {1, 1}}", "colour_start"},
                Case{"{colour_start = {1, 1, 1, 2}}", "colour_start"},
                Case{"{colour_end = {-1, 1, 1}}", "colour_end"},
                Case{"{[1] = 5}", "strings"},
                // One good field in front of a bad one does not get applied.
                Case{"{rate = 7, lifetime = -1}", "lifetime"},
        };

        for (auto const &test: cases) {
            CAPTURE(test.table);
            auto const problem = apply(runtime, test.table);

            REQUIRE(problem.has_value());
            CHECK(problem->find(test.mentions) != std::string::npos);
            CHECK(emitter.count == before.count);
            CHECK(emitter.rate == before.rate);
            CHECK(emitter.lifetime == before.lifetime);
        }
    }

    TEST_CASE("something that is not a table is rejected") {
        auto runtime = make_runtime();

        CHECK(apply(runtime, "5").has_value());
        CHECK(apply(runtime, "'count'").has_value());
    }
}
