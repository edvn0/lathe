#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <string_view>

#include "gameserver/game_server.hxx"

namespace gameserver {

    // Callbacks a transport raises. They may be invoked from the transport's own I/O threads (GameServer is
    // thread-safe), and a transport must raise on_connect before any on_message for that connection and on_disconnect
    // exactly once after the last one.
    struct TransportEvents {
        std::function<void(ConnectionId)> on_connect;
        std::function<void(ConnectionId)> on_disconnect;
        std::function<void(ConnectionId, std::string_view)> on_message;
    };

    // A server-side transport carrying JSON text messages. Implementations hide their library behind this interface
    // so the networking stack can be swapped without touching GameServer or any game.
    class IServerTransport {
    public:
        virtual ~IServerTransport() = default;

        // Starts listening and returns once the socket is bound; I/O continues on the transport's own threads.
        [[nodiscard]] virtual auto start(TransportEvents events) -> std::expected<void, std::string> = 0;

        // Stops accepting, drops every connection and joins the transport's threads. Safe to call twice.
        virtual auto stop() -> void = 0;

        // Queues `text` for `connection`; silently dropped if the connection is already gone.
        virtual auto send(ConnectionId connection, std::string text) -> void = 0;

        [[nodiscard]] virtual auto port() const -> std::uint16_t = 0;
    };

}
