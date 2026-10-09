#pragma once

#include <doctest/doctest.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include "chess_server_game.hxx"
#include "gameserver/game_server.hxx"
#include "gameserver/ix_websocket_transport.hxx"

namespace live_server {
    inline auto free_port() -> std::uint16_t {
        auto const descriptor = ::socket(AF_INET, SOCK_STREAM, 0);

        REQUIRE(descriptor >= 0);

        sockaddr_in address{};

        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        REQUIRE(::bind(descriptor, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0);

        socklen_t length = sizeof(address);

        REQUIRE(::getsockname(descriptor, reinterpret_cast<sockaddr *>(&address), &length) == 0);

        ::close(descriptor);

        return ntohs(address.sin_port);
    }

    // A real lathe-server (WebSocket transport and chess) on a free loopback port, for the lifetime of the object.
    struct LiveServer {
        std::uint16_t port = free_port();

        std::mutex mutex;
        std::unique_ptr<gameserver::IServerTransport> transport;

        gameserver::GameServer server{[this](gameserver::ConnectionId connection, std::string text) {
                                          std::scoped_lock const lock{mutex};

                                          if (transport) {
                                              transport->send(connection, std::move(text));
                                          }
                                      },
                                      std::chrono::seconds{60}, 2};

        LiveServer() {
            server.register_game("chess", ChessServerGame::create);

            start_transport();
        }

        ~LiveServer() { stop_transport(); }

        LiveServer(LiveServer const &) = delete;
        auto operator=(LiveServer const &) -> LiveServer & = delete;
        LiveServer(LiveServer &&) = delete;
        auto operator=(LiveServer &&) -> LiveServer & = delete;

        // Drops every connection and listens again on the same port. Players keep their seats in the GameServer, so
        // this looks to the clients like a network outage they can resume from.
        auto restart_transport() -> void {
            stop_transport();
            start_transport();
        }

        [[nodiscard]] auto url() const -> std::string { return "ws://127.0.0.1:" + std::to_string(port); }

    private:
        // Stopped outside the lock: stopping joins the I/O threads, which may be waiting on it to send.
        auto stop_transport() -> void {
            std::unique_ptr<gameserver::IServerTransport> old;

            {
                std::scoped_lock const lock{mutex};

                old = std::move(transport);
            }

            if (old) {
                old->stop();
            }
        }

        auto start_transport() -> void {
            auto fresh = gameserver::make_ix_websocket_transport("127.0.0.1", port);

            auto const started = fresh->start(gameserver::TransportEvents{
                    .on_connect = [this](gameserver::ConnectionId connection) { server.on_connect(connection); },
                    .on_disconnect = [this](gameserver::ConnectionId connection) { server.on_disconnect(connection); },
                    .on_message = [this](gameserver::ConnectionId connection,
                                         std::string_view text) { server.on_message(connection, text); },
            });

            REQUIRE(started.has_value());

            std::scoped_lock const lock{mutex};

            transport = std::move(fresh);
        }
    };
}
