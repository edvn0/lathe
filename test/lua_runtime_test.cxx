#include <doctest/doctest.h>

#include <array>
#include <string>
#include <string_view>
#include <unordered_map>

#include "scripting/lua_runtime.hxx"

namespace {
    auto make_runtime(LuaRuntime::Settings settings = {}) -> LuaRuntime {
        auto runtime = LuaRuntime::create(settings);

        REQUIRE(runtime.has_value());

        return std::move(*runtime);
    }

    auto load(LuaRuntime &runtime, std::string_view source) -> bool {
        auto const result = runtime.run_main("test.lua", source);

        if (!result) {
            MESSAGE(result.error());
        }

        return result.has_value();
    }

    auto call_number(LuaRuntime &runtime, char const *name) -> double {
        auto const result = runtime.call(name, {}, 1);

        REQUIRE(result.ok);

        auto const value = lua_tonumber(runtime.state(), -1);

        runtime.pop(1);

        return value;
    }
} // namespace

TEST_CASE("the entry script's callbacks are called with their arguments and keep state between calls") {
    auto runtime = make_runtime();

    REQUIRE(load(runtime, R"(
        local total = 0
        return {
            on_update = function(dt) total = total + dt end,
            total = function() return total end,
        }
    )"));

    std::array const step{0.5};

    CHECK(runtime.call("on_update", step).ok);
    CHECK(runtime.call("on_update", step).ok);
    CHECK(call_number(runtime, "total") == doctest::Approx(1.0));
}

TEST_CASE("a missing callback is a no-op, but asking it for results is an error") {
    auto runtime = make_runtime();

    REQUIRE(load(runtime, "return {}"));

    CHECK_FALSE(runtime.has_callback("on_update"));
    CHECK(runtime.call("on_update").ok);
    CHECK_FALSE(runtime.call("camera", {}, 1).ok);
}

TEST_CASE("the entry script must return a table") {
    auto runtime = make_runtime();

    CHECK_FALSE(runtime.run_main("test.lua", "return 3").has_value());
    CHECK_FALSE(runtime.run_main("test.lua", "this is not lua").has_value());
}

TEST_CASE("callback_flag reads boolean fields") {
    auto runtime = make_runtime();

    REQUIRE(load(runtime, "return { wants_cursor = true, other = 1 }"));

    CHECK(runtime.callback_flag("wants_cursor"));
    CHECK_FALSE(runtime.callback_flag("other"));
    CHECK_FALSE(runtime.callback_flag("absent"));
}

TEST_CASE("the sandbox leaves out file, process and code-loading access") {
    auto runtime = make_runtime();

    REQUIRE(load(runtime, R"(
        return { probe = function()
            local forbidden = { io, os, package, debug, dofile, loadfile, load, collectgarbage }
            local count = 0
            for _, value in pairs(forbidden) do count = count + 1 end
            return count
        end }
    )"));

    CHECK(call_number(runtime, "probe") == 0.0);
}

TEST_CASE("require resolves dotted names through the source loader and caches the result") {
    auto runtime = make_runtime();

    std::unordered_map<std::string, std::string> files{
            {"util/maths", "loads = (loads or 0) + 1; return { double = function(x) return x * 2 end }"},
    };

    runtime.set_source_loader([&](std::string const &path) -> std::optional<std::string> {
        auto const it = files.find(path);

        return it != files.end() ? std::optional{it->second} : std::nullopt;
    });

    REQUIRE(load(runtime, R"(
        local maths = require("util.maths")
        local again = require("util.maths")
        return { run = function() return maths.double(21) + (maths == again and 0 or 1000) + loads end }
    )"));

    CHECK(call_number(runtime, "run") == doctest::Approx(43.0));
}

TEST_CASE("require rejects unknown and malformed module names") {
    auto runtime = make_runtime();

    runtime.set_source_loader([](std::string const &) -> std::optional<std::string> { return std::nullopt; });

    CHECK_FALSE(load(runtime, "require('nothing.here'); return {}"));
    CHECK_FALSE(load(runtime, "require('../etc/passwd'); return {}"));
    CHECK_FALSE(load(runtime, "require('a..b'); return {}"));
}

TEST_CASE("native modules are required by name") {
    auto runtime = make_runtime();

    runtime.register_native_module("native.answer", [](lua_State *state) -> int {
        lua_newtable(state);
        lua_pushinteger(state, 42);
        lua_setfield(state, -2, "value");

        return 1;
    });

    REQUIRE(load(runtime, R"(
        local answer = require("native.answer")
        return { value = function() return answer.value end }
    )"));

    CHECK(call_number(runtime, "value") == 42.0);
}

TEST_CASE("a callback that errors is reported once and disabled until reset") {
    auto runtime = make_runtime();

    REQUIRE(load(runtime, R"(
        return { on_update = function() error("boom") end, fine = function() return 1 end }
    )"));

    auto const failed = runtime.call("on_update");

    CHECK_FALSE(failed.ok);
    CHECK(failed.error.find("boom") != std::string::npos);
    CHECK(runtime.last_error().find("on_update") != std::string::npos);

    // Disabled: later calls are skipped rather than failing every frame.
    CHECK_FALSE(runtime.has_callback("on_update"));
    CHECK(runtime.call("on_update").ok);

    // Other callbacks are unaffected.
    CHECK(call_number(runtime, "fine") == 1.0);

    runtime.reset_errors();

    CHECK(runtime.has_callback("on_update"));
    CHECK(runtime.last_error().empty());
}

TEST_CASE("a runaway callback is stopped at its time budget") {
    auto runtime = make_runtime({.call_budget = std::chrono::milliseconds{20}});

    REQUIRE(load(runtime, "return { spin = function() while true do end end, fine = function() return 7 end }"));

    auto const result = runtime.call("spin");

    CHECK_FALSE(result.ok);
    CHECK(result.error.find("time budget") != std::string::npos);

    // The state is usable afterwards, with a fresh budget.
    CHECK(call_number(runtime, "fine") == 7.0);
}

TEST_CASE("allocation past the memory cap fails the call instead of the process") {
    auto runtime = make_runtime({.memory_limit_bytes = 8ULL * 1024 * 1024, .call_budget = std::chrono::seconds{5}});

    REQUIRE(load(runtime, R"(
        return { hog = function()
            local items = {}
            for i = 1, 100000000 do items[i] = i end
        end, fine = function() return 5 end }
    )"));

    auto const result = runtime.call("hog");

    CHECK_FALSE(result.ok);
    CHECK(result.error.find("memory") != std::string::npos);

    runtime.reset_errors();

    CHECK(call_number(runtime, "fine") == 5.0);
}

TEST_CASE("lua_field_vec3 reads named and positional components") {
    auto runtime = make_runtime();

    REQUIRE(load(runtime, "return { camera = function() return { eye = {1, 2, 3}, target = {x = 4, y = 5, z = 6} } end }"));

    REQUIRE(runtime.call("camera", {}, 1).ok);

    auto *const state = runtime.state();
    auto const table = lua_gettop(state);

    auto const eye = lua_field_vec3(state, table, "eye", {});
    auto const target = lua_field_vec3(state, table, "target", {});
    auto const missing = lua_field_vec3(state, table, "up", {0.0F, 1.0F, 0.0F});

    CHECK(eye.x == 1.0F);
    CHECK(eye.z == 3.0F);
    CHECK(target.y == 5.0F);
    CHECK(missing.y == 1.0F);

    runtime.pop(1);
}
