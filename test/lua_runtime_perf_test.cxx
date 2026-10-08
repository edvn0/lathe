#include <doctest/doctest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <format>
#include <print>
#include <string>

#include "chess_lua_harness.hxx"
#include "scripting/lua_runtime.hxx"

namespace {
    using Clock = std::chrono::steady_clock;

    auto seconds_since(Clock::time_point start) -> double {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }

    auto make_runtime(LuaRuntime::Settings settings = {.call_budget = std::chrono::seconds{20}}) -> LuaRuntime {
        auto runtime = LuaRuntime::create(settings);

        REQUIRE(runtime.has_value());

        return std::move(*runtime);
    }

    auto load(LuaRuntime &runtime, std::string_view source) -> void {
        auto const result = runtime.run_main("perf.lua", source);

        if (!result) {
            INFO(result.error());
        }

        REQUIRE(result.has_value());
    }

    auto report(std::string_view name, std::string const &measurement) -> void {
        std::println("[lua perf] {:<34} {}", name, measurement);
    }

    auto nop(lua_State * ) -> int { return 0; }

    auto nop_three_numbers(lua_State *state) -> int {
        luaL_checknumber(state, 1);
        luaL_checknumber(state, 2);
        luaL_checknumber(state, 3);

        return 0;
    }

    auto bench_module(lua_State *state) -> int {
        lua_createtable(state, 0, 2);
        lua_pushcfunction(state, &nop);
        lua_setfield(state, -2, "nop");
        lua_pushcfunction(state, &nop_three_numbers);
        lua_setfield(state, -2, "nop3");

        return 1;
    }
}

TEST_CASE("lua perf: calling an empty callback from the host") {
    auto runtime = make_runtime();

    load(runtime, "return { on_update = function(dt) end }");

    constexpr int calls = 200'000;
    std::array const arguments{0.016};

    auto const start = Clock::now();

    for (int index = 0; index < calls; ++index) {
        REQUIRE(runtime.call("on_update", arguments).ok);
    }

    auto const elapsed = seconds_since(start);
    auto const per_call_ns = elapsed / calls * 1e9;

    report("host -> lua callback",
           std::format("{:.0f} ns per call, {:.0f}k calls/s", per_call_ns, calls / elapsed / 1e3));

    CHECK(per_call_ns < 100'000.0);
}

TEST_CASE("lua perf: interpreter throughput") {
    auto runtime = make_runtime();

    load(runtime, R"(
        return { run = function(n)
            local sum = 0
            for i = 1, n do sum = sum + (i % 7) * 3 end
            return sum
        end }
    )");

    constexpr double iterations = 5'000'000;
    std::array const arguments{iterations};

    auto const start = Clock::now();

    REQUIRE(runtime.call("run", arguments, 1).ok);
    runtime.pop(1);

    auto const elapsed = seconds_since(start);

    report("arithmetic loop", std::format("{:.1f} M iterations/s", iterations / elapsed / 1e6));

    CHECK(iterations / elapsed > 200'000.0);
}

TEST_CASE("lua perf: native function call cost") {
    auto runtime = make_runtime();

    runtime.register_native_module("bench", &bench_module);

    load(runtime, R"(
        local bench = require("bench")
        return {
            nop = function(n) local f = bench.nop for i = 1, n do f() end end,
            nop3 = function(n) local f = bench.nop3 for i = 1, n do f(1.0, 2.0, 3.0) end end,
        }
    )");

    constexpr double calls = 2'000'000;
    std::array const arguments{calls};

    auto measure = [&](char const *name) {
        auto const start = Clock::now();

        REQUIRE(runtime.call(name, arguments).ok);

        return seconds_since(start) / calls * 1e9;
    };

    auto const plain = measure("nop");
    auto const with_arguments = measure("nop3");

    report("lua -> native, no arguments", std::format("{:.0f} ns per call", plain));
    report("lua -> native, 3 numbers", std::format("{:.0f} ns per call ({:.0f} calls per ms)", with_arguments,
                                                    1e6 / with_arguments));

    CHECK(with_arguments < 20'000.0);
}

TEST_CASE("lua perf: allocation churn and the collector") {
    auto runtime = make_runtime();

    load(runtime, R"(
        return { churn = function(n)
            local last
            for i = 1, n do last = { x = i, y = i * 2, name = "item" .. i } end
            return last.x
        end }
    )");

    constexpr double allocations = 1'000'000;
    std::array const arguments{allocations};

    auto const start = Clock::now();

    REQUIRE(runtime.call("churn", arguments, 1).ok);
    runtime.pop(1);

    auto const elapsed = seconds_since(start);

    report("table + string churn",
           std::format("{:.2f} M allocations/s, {:.1f} MiB in use after", allocations / elapsed / 1e6,
                       static_cast<double>(runtime.memory_in_use()) / (1024.0 * 1024.0)));

    CHECK(runtime.memory_in_use() < 64ULL * 1024 * 1024);
}

TEST_CASE("lua perf: how much fits under the memory cap") {
    constexpr std::size_t cap = 32ULL * 1024 * 1024;

    auto runtime = make_runtime({.memory_limit_bytes = cap, .call_budget = std::chrono::seconds{20}});

    load(runtime, R"(
        return {
            fill = function()
                local items = {}
                local ok = pcall(function() for i = 1, 100000000 do items[i] = i end end)
                return #items, ok
            end,
            alive = function() return 1 end,
        }
    )");

    REQUIRE(runtime.call("fill", {}, 2).ok);

    auto const count = lua_tointeger(runtime.state(), -2);
    auto const finished = lua_toboolean(runtime.state(), -1) != 0;

    runtime.pop(2);

    report("array of numbers under 32 MiB",
           std::format("{} elements ({:.1f} bytes each)", count, static_cast<double>(cap) / static_cast<double>(count)));

    CHECK_FALSE(finished);
    CHECK(count > 1'000'000);
    CHECK(runtime.call("alive", {}, 1).ok);

    runtime.pop(1);
}

TEST_CASE("lua perf: how closely the time budget is kept") {
    constexpr auto budget = std::chrono::milliseconds{10};

    auto runtime = make_runtime({.call_budget = budget});

    load(runtime, "return { spin = function() while true do end end }");

    auto const start = Clock::now();
    auto const result = runtime.call("spin");
    auto const elapsed_ms = seconds_since(start) * 1e3;

    REQUIRE_FALSE(result.ok);

    report("10 ms budget, runaway loop", std::format("stopped after {:.2f} ms ({:+.2f} ms)", elapsed_ms,
                                                      elapsed_ms - static_cast<double>(budget.count())));

    CHECK(elapsed_ms >= static_cast<double>(budget.count()));
    CHECK(elapsed_ms < 100.0);
}

TEST_CASE("lua perf: the native chess engine from Lua") {
    auto runtime = make_runtime();

    runtime.register_native_module("native.chess", &luaopen_lathe_chess);

    load(runtime, R"(
        local chess = require("native.chess")
        local engine = chess.new()
        return {
            moves = function(n)
                local total = 0
                for i = 1, n do total = total + #engine:moves() end
                return total
            end,
            play = function(n)
                -- Shuffle knights out and back: four plies per round, always legal.
                for i = 1, n do
                    engine:move(6, 21)
                    engine:move(62, 45)
                    engine:move(21, 6)
                    engine:move(45, 62)
                    if engine:state() ~= "playing" then engine:reset() end
                end
            end,
        }
    )");

    {
        constexpr double calls = 20'000;
        std::array const arguments{calls};
        auto const start = Clock::now();

        REQUIRE(runtime.call("moves", arguments, 1).ok);

        auto const total = lua_tointeger(runtime.state(), -1);

        runtime.pop(1);

        auto const elapsed = seconds_since(start);

        report("engine:moves() (20 moves each)", std::format("{:.0f}k calls/s", calls / elapsed / 1e3));

        CHECK(total == 20 * static_cast<lua_Integer>(calls));
        CHECK(calls / elapsed > 1'000.0);
    }

    {
        constexpr double rounds = 5'000;
        std::array const arguments{rounds};
        auto const start = Clock::now();

        REQUIRE(runtime.call("play", arguments).ok);

        auto const elapsed = seconds_since(start);

        report("engine:move() with legality check", std::format("{:.0f}k moves/s", rounds * 4 / elapsed / 1e3));

        CHECK(rounds * 4 / elapsed > 1'000.0);
    }
}

TEST_CASE("lua perf: a frame of the chess script") {
    chess_lua_test::Harness harness;

    harness.start_playing();

    constexpr int frames = 3'000;

    auto const start = Clock::now();

    for (int index = 0; index < frames; ++index) {
        harness.frame();
    }

    auto const per_frame_us = seconds_since(start) / frames * 1e6;

    report("chess on_update + on_ui (stub API)",
           std::format("{:.1f} us per frame ({:.2f}% of a 60 Hz frame)", per_frame_us, per_frame_us / 16'667.0 * 100.0));

    CHECK(per_frame_us < 2'000.0);
}

TEST_CASE("lua perf: callbacks per frame the budget allows") {
    auto runtime = make_runtime({.call_budget = std::chrono::milliseconds{20}});

    runtime.register_native_module("bench", &bench_module);

    load(runtime, R"(
        local set = require("bench").nop3
        return { update = function(dt, count)
            for i = 1, count do set(i * 0.5 + dt, i * 0.25, i * 0.125) end
        end }
    )");

    for (double const count: {1'000.0, 10'000.0, 100'000.0}) {
        std::array const arguments{0.016, count};

        auto const start = Clock::now();
        auto const result = runtime.call("update", arguments);
        auto const elapsed_ms = seconds_since(start) * 1e3;

        report(std::format("{:.0f} updates in one frame", count),
               result.ok ? std::format("{:.2f} ms ({:.1f}% of 16.7 ms)", elapsed_ms, elapsed_ms / 16.667 * 100.0)
                         : std::format("stopped at the 20 ms budget ({:.1f} ms)", elapsed_ms));

        runtime.reset_errors();
    }

    std::array const arguments{0.016, 1'000.0};
    auto const start = Clock::now();

    REQUIRE(runtime.call("update", arguments).ok);

    CHECK(seconds_since(start) * 1e3 < 5.0);
}
