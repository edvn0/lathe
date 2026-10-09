#include <doctest/doctest.h>

#include <chrono>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chess_server_game.hxx"
#include "gameserver/game_server.hxx"

namespace {
    using gameserver::ConnectionId;

    struct Harness {
        std::vector<std::pair<ConnectionId, JsonValue>> sent;

        std::size_t bad_messages = 0;

        gameserver::GameServer server;

        explicit Harness(std::size_t threads = 2)
            : server{[this](ConnectionId connection, std::string text) {
                         // Runs on worker threads, where doctest assertions are not safe; ~Harness checks the count.
                         if (auto parsed = parse_json(text)) {
                             sent.emplace_back(connection, std::move(*parsed));
                         } else {
                             ++bad_messages;
                         }
                     },
                     std::chrono::seconds{60}, threads} {
            server.register_game("chess", ChessServerGame::create);

            server.on_connect(1);
            server.on_connect(2);
        }

        ~Harness() {
            server.wait_idle();

            CHECK(bad_messages == 0);
        }

        Harness(Harness const &) = delete;
        auto operator=(Harness const &) -> Harness & = delete;
        Harness(Harness &&) = delete;
        auto operator=(Harness &&) -> Harness & = delete;

        auto last_to(ConnectionId connection) const -> JsonValue const & {
            for (auto it = sent.rbegin(); it != sent.rend(); ++it) {
                if (it->first == connection) {
                    return it->second;
                }
            }

            FAIL("no message sent to connection");

            return sent.front().second;
        }

        // The token from the welcome sent to `connection`'s first message.
        auto welcome_token(ConnectionId connection) const -> std::string {
            for (auto const &[to, message] : sent) {
                if (to == connection && message["type"].as_string() == "welcome") {
                    return message["token"].as_string();
                }
            }

            FAIL("no welcome sent to connection");

            return {};
        }

        // Delivers a client message and waits for any game job it started, so tests can assert synchronously.
        auto message(ConnectionId connection, std::string_view text) -> void {
            server.on_message(connection, text);
            server.wait_idle();
        }

        auto start_chess() -> void {
            message(1, R"({"type":"create_room","game":"chess"})");
            message(2, R"({"type":"join_room","room":1})");
        }

        auto move(ConnectionId connection, std::string const &from, std::string const &to) -> void {
            message(connection,
                              R"({"type":"action","action":{"from":")" + from + R"(","to":")" + to + R"("}})");
        }
    };
}

TEST_CASE("creating a room joins it and waits for the second player") {
    Harness harness;

    harness.message(1, R"({"type":"create_room","game":"chess"})");

    auto const &joined = harness.last_to(1);

    CHECK(joined["type"].as_string() == "joined");
    CHECK(joined["seat"].as_number() == 0);
    // Two welcomes (one per connection) and the join.
    CHECK(harness.sent.size() == 3);
}

TEST_CASE("the room starts when it fills and each player gets a state") {
    Harness harness;

    harness.start_chess();

    auto const &white = harness.last_to(1);
    auto const &black = harness.last_to(2);

    CHECK(white["type"].as_string() == "state");
    CHECK(white["state"]["your_side"].as_string() == "white");
    CHECK(white["state"]["legal_moves"].items().size() == 20);

    CHECK(black["state"]["your_side"].as_string() == "black");
    CHECK(black["state"]["legal_moves"].items().empty());
}

TEST_CASE("an unknown game, a bad message and an early action are rejected") {
    Harness harness;

    harness.message(1, R"({"type":"create_room","game":"nope"})");
    CHECK(harness.last_to(1)["code"].as_string() == "unknown_game");

    harness.message(1, "not json");
    CHECK(harness.last_to(1)["code"].as_string() == "bad_json");

    harness.message(1, R"({"type":"action","action":{}})");
    CHECK(harness.last_to(1)["code"].as_string() == "not_in_room");
}

TEST_CASE("moves are validated and broadcast") {
    Harness harness;

    harness.start_chess();

    harness.move(2, "e7", "e5");
    CHECK(harness.last_to(2)["code"].as_string() == "not_your_turn");

    harness.move(1, "e2", "e5");
    CHECK(harness.last_to(1)["code"].as_string() == "illegal_move");

    harness.move(1, "e2", "e4");

    auto const &black = harness.last_to(2);

    CHECK(black["type"].as_string() == "state");
    CHECK(black["state"]["side_to_move"].as_string() == "black");
    CHECK(black["state"]["board"].items()[4].as_string() == "....P...");
    CHECK(black["state"]["history"].items().size() == 1);
    CHECK(black["state"]["history"].items()[0].as_string() == "e2e4");
}

TEST_CASE("checkmate ends the game and closes the room") {
    Harness harness;

    harness.start_chess();

    harness.move(1, "e2", "e4");
    harness.move(2, "e7", "e5");
    harness.move(1, "f1", "c4");
    harness.move(2, "b8", "c6");
    harness.move(1, "d1", "h5");
    harness.move(2, "g8", "f6");
    harness.move(1, "h5", "f7");

    auto const &over = harness.last_to(2);

    CHECK(over["type"].as_string() == "game_over");
    CHECK(over["reason"].as_string() == "checkmate");
    CHECK(over["winner"].as_number() == harness.last_to(1)["winner"].as_number());

    // The room is gone, so both players can start a new one.
    harness.message(1, R"({"type":"create_room","game":"chess"})");
    CHECK(harness.last_to(1)["type"].as_string() == "joined");
}

TEST_CASE("every connection is welcomed with a player id and a token") {
    Harness harness;

    auto const &welcome = harness.last_to(1);

    CHECK(welcome["type"].as_string() == "welcome");
    CHECK(welcome["player"].as_number() == 1);
    CHECK(welcome["token"].as_string().size() == 32);
    CHECK(welcome["token"].as_string() != harness.last_to(2)["token"].as_string());
}

TEST_CASE("leaving a room is acknowledged and closes a started room for the opponent") {
    Harness harness;

    harness.start_chess();

    harness.message(2, R"({"type":"leave_room"})");

    CHECK(harness.last_to(2)["type"].as_string() == "left");

    auto const &closed = harness.last_to(1);

    CHECK(closed["type"].as_string() == "room_closed");
    CHECK(closed["reason"].as_string() == "player_left");

    harness.message(2, R"({"type":"leave_room"})");
    CHECK(harness.last_to(2)["code"].as_string() == "not_in_room");
}

TEST_CASE("a dropped player keeps their seat and resumes on a new connection") {
    Harness harness;

    harness.start_chess();

    harness.server.on_disconnect(2);

    auto const &notice = harness.last_to(1);

    CHECK(notice["type"].as_string() == "opponent_disconnected");

    harness.move(1, "e2", "e4");

    harness.server.on_connect(3);

    auto const resume_token = harness.welcome_token(2);

    harness.message(3, R"({"type":"resume","token":")" + resume_token + R"("})");

    // Welcome, joined, then the current state, all addressed to the new connection.
    auto const &state = harness.last_to(3);

    CHECK(state["type"].as_string() == "state");
    CHECK(state["state"]["your_side"].as_string() == "black");
    CHECK(state["state"]["side_to_move"].as_string() == "black");
    CHECK(harness.last_to(1)["type"].as_string() == "opponent_reconnected");

    harness.move(3, "e7", "e5");

    CHECK(harness.last_to(1)["state"]["side_to_move"].as_string() == "white");
}

TEST_CASE("resume rejects an unknown token and a session that is still connected") {
    Harness harness;

    harness.server.on_connect(3);

    harness.message(3, R"({"type":"resume","token":"nope"})");
    CHECK(harness.last_to(3)["code"].as_string() == "bad_token");

    harness.message(3, R"({"type":"resume","token":")" + harness.welcome_token(1) + R"("})");
    CHECK(harness.last_to(3)["code"].as_string() == "session_in_use");
}

TEST_CASE("a seat that is not reclaimed in time closes the room") {
    Harness harness;

    harness.start_chess();

    auto const dropped_at = gameserver::Clock::now();

    harness.server.on_disconnect(2, dropped_at);

    harness.server.on_tick(dropped_at + std::chrono::seconds{59});
    CHECK(harness.last_to(1)["type"].as_string() == "opponent_disconnected");

    harness.server.on_tick(dropped_at + std::chrono::seconds{61});

    auto const &closed = harness.last_to(1);

    CHECK(closed["type"].as_string() == "room_closed");
    CHECK(closed["reason"].as_string() == "player_timeout");

    // The expired session is gone for good.
    harness.server.on_connect(3);
    harness.message(3, R"({"type":"resume","token":")" + harness.welcome_token(2) + R"("})");
    CHECK(harness.last_to(3)["code"].as_string() == "bad_token");
}

TEST_CASE("a player who drops before the game starts is forgotten") {
    Harness harness;

    harness.message(1, R"({"type":"create_room","game":"chess"})");
    harness.server.on_disconnect(1);

    harness.message(2, R"({"type":"join_room","room":1})");
    CHECK(harness.last_to(2)["code"].as_string() == "no_such_room");
}

namespace {
    // Takes a while to answer, to show that rooms do not wait on each other.
    class SlowGame final : public gameserver::IServerGame {
    public:
        [[nodiscard]] auto player_count() const -> std::size_t override { return 1; }

        auto start(std::span<gameserver::PlayerId const> /*players*/) -> void override {}

        auto apply_action(gameserver::PlayerId /*player*/, JsonValue const &action)
                -> std::expected<void, gameserver::ActionError> override {
            std::this_thread::sleep_for(std::chrono::milliseconds{150});

            counter_ = static_cast<int>(action["n"].as_number());

            return {};
        }

        auto write_state(gameserver::PlayerId /*player*/, JsonWriter &writer) const -> void override {
            writer.value("n", counter_);
        }

        [[nodiscard]] auto outcome() const -> std::optional<gameserver::GameOutcome> override { return std::nullopt; }

    private:
        int counter_ = 0;
    };
}

TEST_CASE("slow games in different rooms run in parallel") {
    constexpr int rooms = 4;

    Harness harness{rooms};

    harness.server.register_game("slow", [] { return std::make_unique<SlowGame>(); });

    for (int index = 0; index < rooms; ++index) {
        auto const connection = static_cast<ConnectionId>(10 + index);

        harness.server.on_connect(connection);
        harness.message(connection, R"({"type":"create_room","game":"slow"})");
    }

    auto const begin = gameserver::Clock::now();

    for (int index = 0; index < rooms; ++index) {
        harness.server.on_message(static_cast<ConnectionId>(10 + index), R"({"type":"action","action":{"n":1}})");
    }

    harness.server.wait_idle();

    auto const elapsed = gameserver::Clock::now() - begin;

    // Serial execution would take 600 ms.
    CHECK(elapsed < std::chrono::milliseconds{400});

    for (int index = 0; index < rooms; ++index) {
        CHECK(harness.last_to(static_cast<ConnectionId>(10 + index))["state"]["n"].as_number() == 1);
    }
}

TEST_CASE("actions in one room run in the order they arrived") {
    Harness harness{4};

    harness.server.register_game("slow", [] { return std::make_unique<SlowGame>(); });

    harness.server.on_connect(10);
    harness.message(10, R"({"type":"create_room","game":"slow"})");

    for (int n = 1; n <= 3; ++n) {
        harness.server.on_message(10, R"({"type":"action","action":{"n":)" + std::to_string(n) + "}}");
    }

    harness.server.wait_idle();

    std::vector<double> seen;

    for (auto const &[to, message] : harness.sent) {
        if (to == 10 && message["type"].as_string() == "state") {
            seen.push_back(message["state"]["n"].as_number());
        }
    }

    // The start state, then one per action, strictly in order.
    CHECK(seen == std::vector<double>{0, 1, 2, 3});
}
