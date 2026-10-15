#include <doctest/doctest.h>

#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "scripting/lua_runtime.hxx"

namespace {
    auto read_file(std::string const &relative) -> std::string {
        std::ifstream file{std::string{TEST_ASSETS_DIR} + "/" + relative, std::ios::binary};

        REQUIRE(file.good());

        std::ostringstream contents;

        contents << file.rdbuf();

        return std::move(contents).str();
    }

    struct CameraView {
        float eye_x = 0.0F;
        float ortho_height = 0.0F;
        float up_z = 0.0F;
    };

    // Runs assets/scripts/snake/main.lua against test/data/snake_stub.lua.
    struct SnakeHarness {
        // Wall-clock call budgets are for runaway scripts, not for this test (see chess_lua_harness.hxx).
        LuaRuntime runtime = [] {
            auto created = LuaRuntime::create(LuaRuntime::Settings{.call_budget = std::chrono::seconds{5}});

            REQUIRE(created.has_value());

            return std::move(*created);
        }();

        SnakeHarness() {
            runtime.set_source_loader([](std::string const &path) -> std::optional<std::string> {
                return read_file("assets/scripts/" + path + ".lua");
            });

            REQUIRE(runtime.run_main("snake_stub.lua", read_file("test/data/snake_stub.lua") + "\nreturn {}")
                            .has_value());
            REQUIRE(runtime.run_main("assets/scripts/snake/main.lua", read_file("assets/scripts/snake/main.lua"))
                            .has_value());
        }

        auto call(char const *name, std::vector<double> const &arguments = {}) -> void {
            auto const result = runtime.call(name, arguments);

            INFO(name << ": " << result.error);
            REQUIRE(result.ok);
        }

        auto run(std::string const &code) -> void { REQUIRE(luaL_dostring(runtime.state(), code.c_str()) == LUA_OK); }

        auto string_result(std::string const &expression) -> std::string {
            run("RESULT = tostring(" + expression + ")");
            lua_getglobal(runtime.state(), "RESULT");

            std::string value = lua_tostring(runtime.state(), -1);

            lua_pop(runtime.state(), 1);

            return value;
        }

        auto frame(double dt = 0.05) -> void {
            run("TEXTS = {}");
            call("on_ui");
            call("on_update", {dt});
        }

        auto press(std::string const &label) -> void {
            run("PRESSED = [[" + label + "]]");
            frame(0.001);
            run("PRESSED = nil");
        }

        auto key(char const *name) -> void { call("on_key", {static_cast<double>(string_result_int(name))}); }

        auto string_result_int(char const *name) -> int {
            return std::stoi(string_result(std::string{"key."} + name));
        }

        auto camera(double aspect = 1.777) -> CameraView {
            auto const result = runtime.call("camera", std::vector<double>{aspect}, 1);

            REQUIRE(result.ok);

            auto *const state = runtime.state();
            auto const table = lua_gettop(state);
            CameraView view;

            view.eye_x = lua_field_vec3(state, table, "eye", {}).x;
            view.up_z = lua_field_vec3(state, table, "up", {}).z;
            view.ortho_height = static_cast<float>(lua_field_number(state, table, "ortho_height", 0.0));
            runtime.pop(1);

            return view;
        }

        auto window() -> std::string { return string_result("LAST_WINDOW"); }

        auto board_problem() -> std::string { return string_result("CHECK_BOARD()"); }
    };
}

TEST_CASE("snake: every board is built up front and only the camera moves between them") {
    SnakeHarness snake;

    snake.call("on_populate");
    snake.frame();

    CHECK(snake.string_result("COUNT('_floor')") == "3");
    CHECK(snake.window() == "##snake_menu");
    CHECK(snake.board_problem().empty());

    // The camera looks straight down, orthographically, at the first board.
    auto const first = snake.camera();
    CHECK(first.ortho_height > 0.0F);
    CHECK(first.up_z == doctest::Approx(1.0F));
    CHECK(first.eye_x == doctest::Approx(6.0F).epsilon(0.01));

    auto const entities_before = snake.string_result("#ENTITIES");

    snake.press("  Large  ");

    for (int index = 0; index < 120; ++index) {
        snake.frame();
    }

    // Large is the third board, 100 units per board along, 32 cells wide: its centre is x = 216.
    auto const large = snake.camera();
    CHECK(large.eye_x == doctest::Approx(216.0F).epsilon(0.01));
    CHECK(large.ortho_height > first.ortho_height);
    CHECK(snake.string_result("#ENTITIES") == entities_before);
    CHECK(snake.board_problem().empty());
}

TEST_CASE("snake: a round runs to game over and can be played again") {
    SnakeHarness snake;

    snake.call("on_populate");
    snake.frame();

    snake.press("Start");
    snake.frame();
    CHECK(snake.window() == "##snake_hud");
    CHECK(snake.string_result("HAS_TEXT('Score 0')") == "true");

    // Steer in a square for a while: every frame the pieces on the board have to be a valid snake.
    char const *const turns[] = {"UP", "LEFT", "DOWN", "RIGHT"};

    for (int step = 0; step < 240; ++step) {
        if (step % 6 == 0) {
            snake.key(turns[(step / 6) % 4]);
        }

        snake.frame();

        INFO("frame " << step);
        auto const problem = snake.board_problem();
        REQUIRE(problem.empty());
    }

    // Left alone it reaches a wall.
    for (int index = 0; index < 4000 && snake.window() != "##snake_over"; ++index) {
        snake.frame();
    }

    CHECK(snake.window() == "##snake_over");
    CHECK(snake.board_problem().empty());

    snake.press("Play again");
    snake.frame();

    CHECK(snake.window() == "##snake_hud");
    CHECK(snake.string_result("HAS_TEXT('Score 0')") == "true");
    CHECK(snake.board_problem().empty());
}

TEST_CASE("snake: the menu's choices change the round that starts") {
    SnakeHarness snake;

    snake.call("on_populate");
    snake.frame();

    snake.press("  Fast  ");
    snake.press("  Frenzy  ");
    snake.press("  Medium  ");

    // Medium is 20 wide, so the snake that waits in the menu sits on the second board (x from 100).
    CHECK(snake.board_problem().empty());
    CHECK(snake.string_result("math.floor(({VISIBLE()})[1][1][1] / 100 + 0.5)") == "1");

    snake.press("Start");

    for (int index = 0; index < 20; ++index) {
        snake.frame();
    }

    CHECK(snake.board_problem().empty());
}

TEST_CASE("snake: the food is an avocado once it has downloaded, and spheres before that") {
    SnakeHarness snake;

    snake.call("on_populate");
    snake.frame();

    // The download is asked for when the game starts; until it finishes the menu says so and the food is a sphere.
    CHECK(snake.string_result("#HTTP_PENDING") == "1");
    CHECK(snake.string_result("HAS_TEXT('Fetching')") == "true");
    CHECK(snake.string_result("FOOD_KIND()") == "sphere");

    snake.run("FINISH_DOWNLOADS(true)");
    snake.frame();
    CHECK(snake.string_result("HAS_TEXT('Fetching')") == "false");

    // The next round (here, choosing a map) uses the avocado.
    snake.press("  Medium  ");
    CHECK(snake.string_result("FOOD_KIND()") == "avocado");
    CHECK(snake.board_problem().empty());

    snake.press("Start");

    for (int index = 0; index < 60; ++index) {
        snake.frame();
        REQUIRE(snake.board_problem().empty());
    }
}

TEST_CASE("snake: a failed download leaves the game playable with spheres and says so") {
    SnakeHarness snake;

    snake.call("on_populate");
    snake.frame();

    snake.run("FINISH_DOWNLOADS(false)");
    snake.frame();

    CHECK(snake.string_result("HAS_TEXT('unavailable')") == "true");

    snake.press("Start");

    for (int index = 0; index < 60; ++index) {
        snake.frame();
        REQUIRE(snake.board_problem().empty());
    }

    CHECK(snake.string_result("FOOD_KIND()") == "sphere");
}

TEST_CASE("snake: the rules hold, including under random play") {
    auto created = LuaRuntime::create(LuaRuntime::Settings{.call_budget = std::chrono::seconds{20}});

    REQUIRE(created.has_value());

    created->set_source_loader([](std::string const &path) -> std::optional<std::string> {
        return read_file("assets/scripts/" + path + ".lua");
    });

    auto const result = created->run_main("snake_rules.lua", read_file("test/data/snake_rules.lua"));

    INFO(result.error());
    CHECK(result.has_value());
}
