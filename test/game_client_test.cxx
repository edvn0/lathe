#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "chess_server_game.hxx"
#include "core/json.hxx"
#include "gameclient/game_client.hxx"
#include "gameclient/ix_websocket_client_transport.hxx"
#include "gameserver/game_server.hxx"
#include "live_server.hxx"

namespace {
    using gameclient::ClientEvent;
    using gameclient::GameClient;

    // Collects drained events and waits for a message of a given type.
    struct Inbox {
        GameClient &client;

        std::vector<JsonValue> messages;
        std::vector<ClientEvent::Kind> kinds;

        auto pump() -> void {
            for (auto &event : client.drain()) {
                kinds.push_back(event.kind);

                if (event.kind == ClientEvent::Kind::message) {
                    auto parsed = parse_json(event.text);

                    REQUIRE(parsed.has_value());

                    messages.push_back(std::move(*parsed));
                }
            }
        }

        // Waits for a message of `type` after the ones already consumed, and returns it.
        auto wait_for(std::string_view type) -> JsonValue {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};

            while (std::chrono::steady_clock::now() < deadline) {
                pump();

                for (std::size_t index = consumed; index < messages.size(); ++index) {
                    if (messages[index]["type"].as_string() == type) {
                        consumed = index + 1;

                        return messages[index];
                    }
                }

                std::this_thread::sleep_for(std::chrono::milliseconds{2});
            }

            FAIL("timed out waiting for " << type);

            return {};
        }

        std::size_t consumed = 0;
    };

    // A transport that talks to a GameServer in the same process.
    class Network {
    public:
        Network() : server_{[this](gameserver::ConnectionId connection, std::string text) { deliver(connection, std::move(text)); }, std::chrono::seconds{60}, 2} {
            server_.register_game("chess", ChessServerGame::create);
        }

        auto factory() -> gameclient::TransportFactory {
            return [this] { return std::make_unique<Link>(*this); };
        }

        // Cuts the current connection of the most recently dialled client, as a network failure would.
        auto drop_last() -> void {
            std::function<void(std::string)> on_close;
            gameserver::ConnectionId connection = 0;

            {
                std::scoped_lock const lock{mutex_};

                connection = next_connection_ - 1;
                on_close = links_.at(connection)->events.on_close;
                links_.erase(connection);
            }

            server_.on_disconnect(connection);
            on_close("dropped");
        }

        auto idle() -> void { server_.wait_idle(); }

    private:
        class Link final : public gameclient::IClientTransport {
        public:
            explicit Link(Network &network) : network_(network) {}

            ~Link() override { close(); }

            auto connect(std::string /*url*/, gameclient::ClientTransportEvents new_events) -> void override {
                events = std::move(new_events);

                {
                    std::scoped_lock const lock{network_.mutex_};

                    id_ = network_.next_connection_++;
                    network_.links_[id_] = this;
                }

                events.on_open();
                network_.server_.on_connect(id_);
            }

            auto send(std::string text) -> bool override {
                network_.server_.on_message(id_, text);

                return true;
            }

            auto close() -> void override {
                std::scoped_lock const lock{network_.mutex_};

                if (auto const link = network_.links_.find(id_); link != network_.links_.end() && link->second == this) {
                    network_.links_.erase(link);
                }
            }

            gameclient::ClientTransportEvents events;

        private:
            Network &network_;

            gameserver::ConnectionId id_ = 0;
        };

        auto deliver(gameserver::ConnectionId connection, std::string text) -> void {
            std::function<void(std::string)> on_message;

            {
                std::scoped_lock const lock{mutex_};

                auto const link = links_.find(connection);

                if (link == links_.end()) {
                    return;
                }

                on_message = link->second->events.on_message;
            }

            on_message(std::move(text));
        }

        std::mutex mutex_;
        std::map<gameserver::ConnectionId, Link *> links_;
        gameserver::ConnectionId next_connection_ = 1;

        gameserver::GameServer server_;
    };

    auto create_room(GameClient &client) -> void {
        REQUIRE(client.send(R"({"type":"create_room","game":"chess"})"));
    }

    auto play(GameClient &client, std::string const &from, std::string const &to) -> void {
        REQUIRE(client.send(R"({"type":"action","action":{"from":")" + from + R"(","to":")" + to + R"("}})"));
    }
}

TEST_CASE("a client connects, is welcomed and keeps the session token") {
    Network network;

    GameClient client{network.factory()};
    Inbox inbox{client};

    CHECK_FALSE(client.send("{}"));

    client.connect("ws://ignored");

    auto const welcome = inbox.wait_for("welcome");

    CHECK(client.is_open());
    CHECK(client.token() == welcome["token"].as_string());
    CHECK(inbox.kinds.front() == ClientEvent::Kind::opened);
}

TEST_CASE("two clients play moves through the server") {
    Network network;

    GameClient white{network.factory()};
    GameClient black{network.factory()};
    Inbox white_inbox{white};
    Inbox black_inbox{black};

    white.connect("x");
    white_inbox.wait_for("welcome");
    black.connect("x");
    black_inbox.wait_for("welcome");

    create_room(white);

    auto const room = white_inbox.wait_for("joined")["room"].as_number();

    REQUIRE(black.send(R"({"type":"join_room","room":)" + std::to_string(static_cast<int>(room)) + "}"));

    white_inbox.wait_for("state");
    black_inbox.wait_for("state");

    play(white, "e2", "e4");

    auto const state = black_inbox.wait_for("state");

    CHECK(state["state"]["history"].items().size() == 1);
    CHECK(state["state"]["history"].items()[0].as_string() == "e2e4");
    CHECK(state["state"]["your_side"].as_string() == "black");
}

TEST_CASE("a dropped client reconnects and resumes its seat") {
    Network network;

    GameClient white{network.factory()};
    GameClient black{network.factory()};
    Inbox white_inbox{white};
    Inbox black_inbox{black};

    white.connect("x");
    white_inbox.wait_for("welcome");

    create_room(white);

    auto const room = white_inbox.wait_for("joined")["room"].as_number();

    black.connect("x");
    black_inbox.wait_for("welcome");

    REQUIRE(black.send(R"({"type":"join_room","room":)" + std::to_string(static_cast<int>(room)) + "}"));

    white_inbox.wait_for("state");
    black_inbox.wait_for("state");

    play(white, "e2", "e4");
    black_inbox.wait_for("state");

    network.drop_last();

    black_inbox.pump();
    CHECK(black_inbox.kinds.back() == ClientEvent::Kind::closed);
    CHECK_FALSE(black.is_open());
    CHECK_FALSE(black.send("{}"));

    white_inbox.wait_for("opponent_disconnected");

    black.reconnect();
    black_inbox.wait_for("welcome");

    // The fresh connection's welcome must not replace the saved session.
    REQUIRE(black.resume());

    auto const joined = black_inbox.wait_for("joined");

    CHECK(joined["seat"].as_number() == 1);

    auto const state = black_inbox.wait_for("state");

    CHECK(state["state"]["history"].items().size() == 1);
    CHECK(state["state"]["side_to_move"].as_string() == "black");

    white_inbox.wait_for("opponent_reconnected");
}

TEST_CASE("resume without a session does nothing") {
    Network network;

    GameClient client{network.factory()};

    CHECK_FALSE(client.resume());

    client.reconnect();
    CHECK_FALSE(client.is_open());
}

TEST_CASE("a client plays a game against the server over a real WebSocket") {
    live_server::LiveServer live;

    {
        GameClient white{&gameclient::make_ix_websocket_client_transport};
        GameClient black{&gameclient::make_ix_websocket_client_transport};
        Inbox white_inbox{white};
        Inbox black_inbox{black};

        auto const url = live.url();

        white.connect(url);
        white_inbox.wait_for("welcome");
        black.connect(url);
        black_inbox.wait_for("welcome");

        create_room(white);

        auto const room = white_inbox.wait_for("joined")["room"].as_number();

        REQUIRE(black.send(R"({"type":"join_room","room":)" + std::to_string(static_cast<int>(room)) + "}"));

        white_inbox.wait_for("state");
        black_inbox.wait_for("state");

        play(white, "e2", "e4");

        auto const state = black_inbox.wait_for("state");

        CHECK(state["state"]["history"].items()[0].as_string() == "e2e4");

        // Cut black's connection; white is told, and black can come back.
        black.close();

        white_inbox.wait_for("opponent_disconnected");

        black.reconnect();
        black_inbox.wait_for("welcome");

        REQUIRE(black.resume());
        black_inbox.wait_for("joined");
        black_inbox.wait_for("state");
        white_inbox.wait_for("opponent_reconnected");
    }
}

TEST_CASE("a connection to nothing reports closed") {
    GameClient client{&gameclient::make_ix_websocket_client_transport};
    Inbox inbox{client};

    client.connect("ws://127.0.0.1:" + std::to_string(live_server::free_port()));

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};

    while (std::chrono::steady_clock::now() < deadline && (inbox.pump(), inbox.kinds.empty())) {
        std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }

    REQUIRE_FALSE(inbox.kinds.empty());
    CHECK(inbox.kinds.back() == ClientEvent::Kind::closed);
    CHECK_FALSE(client.is_open());
}
