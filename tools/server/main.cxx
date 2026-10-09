#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <format>
#include <string>
#include <thread>

#include "chess_server_game.hxx"
#include "core/command_line.hxx"
#include "core/logger.hxx"
#include "gameserver/game_server.hxx"
#include "gameserver/ix_websocket_transport.hxx"

namespace {
    std::atomic<bool> stop_requested{false};

    auto request_stop(int /*signal*/) -> void { stop_requested = true; }
}

auto main(int argc, char **argv) -> int {
    std::string host = "127.0.0.1";
    std::uint16_t port = 9002;
    std::uint32_t grace_seconds = 60;
    std::size_t threads = 0;

    CommandLine cli{"lathe-server", "Headless multiplayer game server speaking JSON over WebSocket."};

    auto network = cli.group("Network");
    network.value("--host", "ADDR", "Address to listen on (127.0.0.1; use 0.0.0.0 for all interfaces)", host);
    network.value("--port", "N", "Port to listen on (9002)", port, {.min = 1, .max = 65535});
    network.value("--threads", "N", "Game worker threads, 0 for one per hardware thread (0)", threads);
    network.value("--reconnect-grace", "SECONDS", "How long a dropped player keeps their seat (60)", grace_seconds);

    auto const outcome = cli.parse(argc, argv);

    if (!outcome) {
        std::fprintf(stderr, "lathe-server: %s\nTry --help.\n", outcome.error().c_str());
        return 2;
    }

    if (*outcome == CommandLine::Outcome::help) {
        std::fputs(cli.help_text().c_str(), stdout);
        return 0;
    }

    auto transport = gameserver::make_ix_websocket_transport(host, port);

    gameserver::GameServer server{[&transport](gameserver::ConnectionId connection, std::string text) {
                                      transport->send(connection, std::move(text));
                                  },
                                  std::chrono::seconds{grace_seconds}, threads};

    server.register_game("chess", ChessServerGame::create);

    auto const started = transport->start(gameserver::TransportEvents{
            .on_connect = [&server](gameserver::ConnectionId connection) { server.on_connect(connection); },
            .on_disconnect = [&server](gameserver::ConnectionId connection) { server.on_disconnect(connection); },
            .on_message = [&server](gameserver::ConnectionId connection,
                                    std::string_view text) { server.on_message(connection, text); },
    });

    if (!started) {
        std::fprintf(stderr, "lathe-server: cannot listen on %s:%u: %s\n", host.c_str(), static_cast<unsigned>(port),
                     started.error().c_str());
        return 1;
    }

    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);

    logger::info(std::format("lathe-server listening on ws://{}:{}", host, port));

    while (!stop_requested) {
        server.on_tick();

        std::this_thread::sleep_for(std::chrono::milliseconds{250});
    }

    // Stop the transport first so no event reaches the server while it is being torn down.
    transport->stop();

    return 0;
}
