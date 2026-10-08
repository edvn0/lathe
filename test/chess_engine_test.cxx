#include <doctest/doctest.h>

#include "chess_engine.hxx"

namespace {
    using chess::GameState;
    using chess::Square;

    auto play(chess::ChessEngine &engine, Square from, Square to) -> chess::MoveResult {
        auto const result = engine.update(from, to);

        REQUIRE(result.moved);

        return result;
    }
}

TEST_CASE("a fresh game is in progress with white to move") {
    chess::ChessEngine engine;

    CHECK(engine.game_state() == GameState::playing);
    CHECK(engine.side_to_move() == chess::Side::white);
    CHECK(engine.legal_moves().size() == 20);
}

TEST_CASE("scholar's mate ends the game in checkmate") {
    chess::ChessEngine engine;

    play(engine, Square::e2, Square::e4);
    play(engine, Square::e7, Square::e5);
    play(engine, Square::f1, Square::c4);
    play(engine, Square::b8, Square::c6);
    play(engine, Square::d1, Square::h5);
    play(engine, Square::g8, Square::f6);

    auto const mate = play(engine, Square::h5, Square::f7);

    CHECK(mate.game_state == GameState::checkmate);
    CHECK(engine.game_state() == GameState::checkmate);
    CHECK(engine.side_to_move() == chess::Side::black);
}

TEST_CASE("the third occurrence of a position is a draw") {
    chess::ChessEngine engine;

    for (int round = 0; round < 2; ++round) {
        play(engine, Square::g1, Square::f3);
        play(engine, Square::g8, Square::f6);
        play(engine, Square::f3, Square::g1);

        auto const last = play(engine, Square::f6, Square::g8);

        CHECK(last.game_state == (round == 0 ? GameState::playing : GameState::draw_repetition));
    }

    CHECK(engine.game_state() == GameState::draw_repetition);
}

TEST_CASE("reset forgets earlier positions") {
    chess::ChessEngine engine;

    for (int round = 0; round < 2; ++round) {
        play(engine, Square::g1, Square::f3);
        play(engine, Square::g8, Square::f6);
        play(engine, Square::f3, Square::g1);
        play(engine, Square::f6, Square::g8);
    }

    REQUIRE(engine.game_state() == GameState::draw_repetition);

    engine.reset();

    CHECK(engine.game_state() == GameState::playing);

    play(engine, Square::g1, Square::f3);
    play(engine, Square::g8, Square::f6);
    play(engine, Square::f3, Square::g1);

    CHECK(play(engine, Square::f6, Square::g8).game_state == GameState::playing);
}
