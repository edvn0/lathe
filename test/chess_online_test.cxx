#include <doctest/doctest.h>

#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "chess_lua_harness.hxx"
#include "live_server.hxx"

using chess_lua_test::Harness;

namespace {
    // Frames every harness until its condition (a Lua expression) holds, so the real network gets time to answer.
    auto wait_until(std::vector<std::pair<Harness *, std::string>> const &conditions) -> bool {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};

        while (std::chrono::steady_clock::now() < deadline) {
            bool all = true;

            for (auto const &[harness, condition] : conditions) {
                harness->run("TEXTS = {}");
                harness->frame(0.05F);
                harness->run("COND = (" + condition + ") and true or false");

                all = harness->bool_global("COND") && all;
            }

            if (all) {
                return true;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds{4});
        }

        return false;
    }

    constexpr char const *helpers = R"(
        function has_text(s) for _, t in ipairs(TEXTS) do if t:find(s, 1, true) then return true end end return false end
    )";

    auto reach_menu(Harness &harness) -> void {
        harness.run(helpers);
        harness.call("on_populate");
        harness.call("on_bind");

        for (int index = 0; index < 40; ++index) {
            harness.frame();
        }

        REQUIRE(harness.string_global("LAST_WINDOW") == "##menu");
    }

    auto press(Harness &harness, std::string const &label) -> void {
        harness.run("PRESSED = '" + label + "'");
        harness.frame();
        harness.run("PRESSED = nil");
    }

    // Menu -> Play online -> Connect, until the lobby is showing.
    auto connect(Harness &harness, std::string const &url) -> void {
        press(harness, "Play online");

        harness.run("INPUT['##server'] = '" + url + "'");
        press(harness, "Connect");

        REQUIRE(wait_until({{&harness, "LAST_WINDOW == '##online' and has_text('Connected as player')"}}));
    }
}

TEST_CASE("two Lua clients play scholar's mate through a real server") {
    live_server::LiveServer live;

    Harness white;
    Harness black;

    reach_menu(white);
    reach_menu(black);

    connect(white, live.url());
    connect(black, live.url());

    press(white, "Create room");

    REQUIRE(wait_until({{&white, "has_text('Waiting for an opponent')"}}));

    black.run("INPUT['##room'] = '1'");
    press(black, "Join room");

    // The game starts for both once the second player is seated.
    REQUIRE(wait_until({
            {&white, "LAST_WINDOW == '##hud' and has_text('Your move') and has_text('You are white')"},
            {&black, "LAST_WINDOW == '##hud' and has_text(\"Opponent's move\") and has_text('You are black')"},
    }));

    auto const turn = [&](Harness &mover, Harness &other, int from_file, int from_rank, int to_file, int to_rank) {
        mover.move(from_file, from_rank, to_file, to_rank);

        REQUIRE(wait_until({
                {&other, "LAST_WINDOW == '##hud' and has_text('Your move')"},
                {&mover, "LAST_WINDOW == '##hud' and has_text(\"Opponent's move\")"},
        }));
    };

    turn(white, black, 4, 1, 4, 3);
    turn(black, white, 4, 6, 4, 4);
    turn(white, black, 5, 0, 2, 3);
    turn(black, white, 1, 7, 2, 5);
    turn(white, black, 3, 0, 7, 4);
    turn(black, white, 6, 7, 5, 5);

    // Black's board mirrors white's e-pawn on e4, as the server's history arrived.
    black.run(R"(
        local board = require("chess.board")
        local x, _, z = board.position(board.square(4, 3))
        PAWN_ON_E4 = false
        for name, entity in pairs(ENTITIES) do
            if name:find("piece_") and math.abs(entity.position[1] - x) < 1e-6 and math.abs(entity.position[3] - z) < 1e-6 then
                PAWN_ON_E4 = true
            end
        end
    )");
    CHECK(black.bool_global("PAWN_ON_E4"));

    white.move(7, 4, 5, 6);

    REQUIRE(wait_until({
            {&white, "LAST_WINDOW == '##game_over' and has_text('You win')"},
            {&black, "LAST_WINDOW == '##game_over' and has_text('You lose')"},
    }));
}

TEST_CASE("a Lua client cannot move out of turn") {
    live_server::LiveServer live;

    Harness white;
    Harness black;

    reach_menu(white);
    reach_menu(black);

    connect(white, live.url());
    connect(black, live.url());

    press(white, "Create room");
    REQUIRE(wait_until({{&white, "has_text('Waiting for an opponent')"}}));

    black.run("INPUT['##room'] = '1'");
    press(black, "Join room");

    REQUIRE(wait_until({{&black, "LAST_WINDOW == '##hud' and has_text('You are black')"}}));

    // Black clicks one of its own pieces on white's turn: nothing is selected.
    black.click(4, 6);

    black.run("TEXTS = {}");
    black.frame();
    black.run("COND = has_text('Waiting for your opponent')");

    CHECK(black.bool_global("COND"));
}

TEST_CASE("the lobby reports a server that is not there and offers to try again") {
    Harness harness;

    reach_menu(harness);

    press(harness, "Play online");
    harness.run("INPUT['##server'] = 'ws://127.0.0.1:" + std::to_string(live_server::free_port()) + "'");
    press(harness, "Connect");

    REQUIRE(wait_until({{&harness, "LAST_WINDOW == '##online' and has_text('Could not connect')"}}));
}

TEST_CASE("a Lua client that receives the game start and the first move in one frame shows the move") {
    live_server::LiveServer live;

    Harness white;
    Harness black;

    reach_menu(white);
    reach_menu(black);

    connect(white, live.url());
    connect(black, live.url());

    press(white, "Create room");
    REQUIRE(wait_until({{&white, "has_text('Waiting for an opponent')"}}));

    black.run("INPUT['##room'] = '1'");
    press(black, "Join room");

    // Black is not framed while white starts and moves, so both states are waiting for its next frame.
    REQUIRE(wait_until({{&white, "LAST_WINDOW == '##hud' and has_text('Your move')"}}));

    white.move(4, 1, 4, 3);

    REQUIRE(wait_until({{&white, "has_text(\"Opponent's move\")"}}));

    std::this_thread::sleep_for(std::chrono::milliseconds{200});

    REQUIRE(wait_until({{&black, "LAST_WINDOW == '##hud' and has_text('Your move')"}}));
}

TEST_CASE("a Lua client whose connection drops resumes the game") {
    live_server::LiveServer live;

    Harness white;
    Harness black;

    reach_menu(white);
    reach_menu(black);

    connect(white, live.url());
    connect(black, live.url());

    press(white, "Create room");
    REQUIRE(wait_until({{&white, "has_text('Waiting for an opponent')"}}));

    black.run("INPUT['##room'] = '1'");
    press(black, "Join room");

    REQUIRE(wait_until({{&white, "LAST_WINDOW == '##hud' and has_text('Your move')"}}));

    white.move(4, 1, 4, 3);

    REQUIRE(wait_until({{&black, "LAST_WINDOW == '##hud' and has_text('Your move')"}}));

    // The network drops for everyone; both scripts reconnect on their own and take their seats back.
    live.restart_transport();

    REQUIRE(wait_until({
            {&black, "LAST_WINDOW == '##hud' and has_text('Your move') and not has_text('reconnecting')"},
            {&white, "LAST_WINDOW == '##hud' and has_text(\"Opponent's move\") and not has_text('reconnecting')"},
    }));

    // And the game goes on where it left off.
    black.move(4, 6, 4, 4);

    REQUIRE(wait_until({{&white, "LAST_WINDOW == '##hud' and has_text('Your move')"}}));
}
