#include <doctest/doctest.h>

#include <array>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#include "chess_lua.hxx"
#include "scripting/lua_runtime.hxx"

// Runs the real assets/scripts/chess/main.lua against stub engine APIs (scene, assets, ui, ...) and the real native
// chess module, and plays it through its callbacks: no renderer, no window.

namespace {
    auto read_asset(std::string const &relative) -> std::string {
        std::ifstream file{std::string{TEST_ASSETS_DIR} + "/" + relative, std::ios::binary};

        REQUIRE(file.good());

        std::ostringstream contents;

        contents << file.rdbuf();

        return std::move(contents).str();
    }

    // What the engine's bindings would provide, recording what the script asks of them.
    constexpr char const *stub_api = R"(
        key = { ENTER = 257, SPACE = 32, ESCAPE = 256, BACKSPACE = 259, UP = 265, DOWN = 264, LEFT = 263, RIGHT = 262 }
        for code = 65, 90 do key[string.char(code)] = code end
        mouse = { LEFT = 0, RIGHT = 1, MIDDLE = 2 }

        WINDOWS = {}
        TEXTS = {}
        PRESSED = nil
        PICK = nil
        QUIT = false
        PENDING = 0

        local function entity(name)
            local e = { name = name, position = { 0, 0, 0 }, outlined = false }
            function e:valid() return true end
            function e:destroy() end
            function e:set_position(x, y, z) self.position = { x, y, z } end
            function e:set_euler(x, y, z) self.yaw = y end
            function e:set_scale() end
            function e:set_model(model) self.model = model end
            function e:set_material() end
            function e:set_outlined(on) self.outlined = on end
            return e
        end

        ENTITIES = {}
        scene = {
            spawn = function(name) local e = entity(name); ENTITIES[name] = e; return e end,
            find = function(name) return ENTITIES[name] end,
            clear = function() ENTITIES = {} end,
        }

        assets = {
            load_model = function(path) return { path = path } end,
            cube = function() return { path = "cube" } end,
            material = function() return {} end,
            release_material = function() end,
            pending = function() return PENDING end,
        }

        camera = { pick_plane = function() if PICK then return PICK.x, PICK.z end end }

        game = { player_mode = true, time = function() return 0 end, quit = function() QUIT = true end }

        ui = {
            window = function(id, options, body) table.insert(WINDOWS, id); LAST_WINDOW = id; body() end,
            text = function(s) table.insert(TEXTS, s) end,
            text_centred = function(s) table.insert(TEXTS, s) end,
            text_disabled = function() end,
            button = function(label) return PRESSED == label end,
            small_button = function(label) return PRESSED == label end,
            progress = function() end,
            spinner = function() end,
            dummy = function() end,
            same_line = function() end,
            separator = function() end,
            display_size = function() return 1920, 1080 end,
        }
    )";

    struct Harness {
        LuaRuntime runtime = [] {
            auto created = LuaRuntime::create();

            REQUIRE(created.has_value());

            return std::move(*created);
        }();

        Harness() {
            runtime.register_native_module("native.chess", &luaopen_lathe_chess);
            runtime.set_source_loader([](std::string const &path) -> std::optional<std::string> {
                return read_asset("assets/scripts/" + path + ".lua");
            });

            // The stub chunk is not a game: it returns an empty callback table, then the real script replaces it.
            REQUIRE(runtime.run_main("stub.lua", std::string{stub_api} + "return {}").has_value());
            REQUIRE(runtime.run_main("assets/scripts/chess/main.lua", read_asset("assets/scripts/chess/main.lua"))
                            .has_value());
        }

        auto call(char const *name, std::initializer_list<double> arguments = {}) -> void {
            std::vector<double> const values{arguments};
            auto const result = runtime.call(name, values);

            INFO(name << ": " << result.error);
            REQUIRE(result.ok);
        }

        auto run(std::string const &code) -> void {
            REQUIRE(luaL_dostring(runtime.state(), code.c_str()) == LUA_OK);
        }

        auto string_global(char const *name) -> std::string {
            lua_getglobal(runtime.state(), name);

            std::string value = lua_isstring(runtime.state(), -1) ? lua_tostring(runtime.state(), -1) : "";

            lua_pop(runtime.state(), 1);

            return value;
        }

        auto bool_global(char const *name) -> bool {
            lua_getglobal(runtime.state(), name);

            auto const value = lua_toboolean(runtime.state(), -1) != 0;

            lua_pop(runtime.state(), 1);

            return value;
        }

        auto frame(float dt = 0.016F) -> void {
            call("on_update", {static_cast<double>(dt)});
            call("on_ui");
        }

        // Leaves the loading screen and presses Play.
        auto start_playing() -> void {
            call("on_populate");
            call("on_bind");

            for (int index = 0; index < 40; ++index) {
                frame();
            }

            REQUIRE(string_global("LAST_WINDOW") == "##menu");

            run("PRESSED = 'Play'");
            frame();
            run("PRESSED = nil");
            frame();

            REQUIRE(string_global("LAST_WINDOW") == "##hud");
        }

        // A click on a square, through the cursor and mouse-button callbacks.
        auto click(int file, int rank) -> void {
            auto const x = (static_cast<double>(file) - 3.5) * 1.08;
            auto const z = (static_cast<double>(rank) - 3.5) * 1.08;

            run("PICK = { x = " + std::to_string(x) + ", z = " + std::to_string(z) + " }");
            call("on_cursor", {0.0, 0.0, 1.0});
            call("on_mouse_button", {0.0});
            frame();
        }

        auto move(int from_file, int from_rank, int to_file, int to_rank) -> void {
            click(from_file, from_rank);
            click(to_file, to_rank);

            // Let the camera swing finish settling between moves.
            for (int index = 0; index < 5; ++index) {
                frame(0.1F);
            }
        }
    };
} // namespace

TEST_CASE("the chess script loads and idles on its loading screen until the models are in") {
    Harness harness;

    harness.run("PENDING = 5");
    harness.call("on_populate");
    harness.call("on_bind");

    for (int index = 0; index < 60; ++index) {
        harness.frame();
    }

    CHECK(harness.string_global("LAST_WINDOW") == "##loading");

    harness.run("PENDING = 0");

    for (int index = 0; index < 40; ++index) {
        harness.frame();
    }

    CHECK(harness.string_global("LAST_WINDOW") == "##menu");
}

TEST_CASE("Play starts a game; scholar's mate through mouse clicks ends on the game-over screen") {
    Harness harness;

    harness.start_playing();

    harness.move(4, 1, 4, 3); // e4
    harness.move(4, 6, 4, 4); // e5
    harness.move(5, 0, 2, 3); // Bc4
    harness.move(1, 7, 2, 5); // Nc6
    harness.move(3, 0, 7, 4); // Qh5
    harness.move(6, 7, 5, 5); // Nf6

    CHECK(harness.string_global("LAST_WINDOW") == "##hud");

    harness.move(7, 4, 5, 6); // Qxf7#

    // The result is shown after a short delay.
    for (int index = 0; index < 30; ++index) {
        harness.frame(0.1F);
    }

    CHECK(harness.string_global("LAST_WINDOW") == "##game_over");

    harness.run("TEXTS = {}");
    harness.frame();

    // Checkmate, and black is the side to move, so white won.
    harness.run("RESULT = table.concat(TEXTS, '|')");

    auto const texts = harness.string_global("RESULT");

    CHECK(texts.find("Checkmate") != std::string::npos);
    CHECK(texts.find("White wins") != std::string::npos);

    harness.run("PRESSED = 'Play again'");
    harness.frame();
    harness.run("PRESSED = nil");
    harness.frame();
    harness.frame();

    CHECK(harness.string_global("LAST_WINDOW") == "##hud");

    harness.run("TEXTS = {}");
    harness.frame();
    harness.run("RESULT = table.concat(TEXTS, '|')");

    CHECK(harness.string_global("RESULT").find("White to move") != std::string::npos);
}

TEST_CASE("Escape pauses and resumes, and Quit asks the host to exit") {
    Harness harness;

    harness.start_playing();

    harness.call("on_key", {256.0, 0.0});
    harness.frame();

    CHECK(harness.string_global("LAST_WINDOW") == "##pause");

    harness.call("on_key", {256.0, 0.0});
    harness.frame();

    CHECK(harness.string_global("LAST_WINDOW") == "##hud");

    harness.call("on_key", {256.0, 0.0});
    harness.run("PRESSED = 'Quit'");
    harness.frame();

    CHECK(harness.bool_global("QUIT"));
}

TEST_CASE("an illegal click is rejected and leaves the board alone") {
    Harness harness;

    harness.start_playing();

    // Selecting a black piece on white's turn does nothing.
    harness.click(4, 6);
    harness.click(4, 4);

    harness.run("TEXTS = {}");
    harness.frame();
    harness.run("RESULT = table.concat(TEXTS, '|')");

    CHECK(harness.string_global("RESULT").find("White to move") != std::string::npos);
}
