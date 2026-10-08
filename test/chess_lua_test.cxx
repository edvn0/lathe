#include <doctest/doctest.h>

#include <string>

#include "chess_lua_harness.hxx"

using chess_lua_test::Harness;

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

    harness.move(4, 1, 4, 3);
    harness.move(4, 6, 4, 4);
    harness.move(5, 0, 2, 3);
    harness.move(1, 7, 2, 5);
    harness.move(3, 0, 7, 4);
    harness.move(6, 7, 5, 5);

    CHECK(harness.string_global("LAST_WINDOW") == "##hud");

    harness.move(7, 4, 5, 6);

    for (int index = 0; index < 30; ++index) {
        harness.frame(0.1F);
    }

    CHECK(harness.string_global("LAST_WINDOW") == "##game_over");

    harness.run("TEXTS = {}");
    harness.frame();

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

    harness.click(4, 6);
    harness.click(4, 4);

    harness.run("TEXTS = {}");
    harness.frame();
    harness.run("RESULT = table.concat(TEXTS, '|')");

    CHECK(harness.string_global("RESULT").find("White to move") != std::string::npos);
}
