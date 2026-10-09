#pragma once

#include <functional>
#include <string>

namespace gameclient {

    // Callbacks a client transport raises, from its own I/O thread. A transport raises on_open at most once and
    // on_close exactly once per connect() (including when the connection never opened).
    struct ClientTransportEvents {
        std::function<void()> on_open;
        std::function<void(std::string)> on_message;
        std::function<void(std::string)> on_close;
    };

    // One WebSocket-style connection carrying JSON text. Implementations hide their library behind this interface so
    // the networking stack can be swapped without touching GameClient or any game.
    class IClientTransport {
    public:
        virtual ~IClientTransport() = default;

        // Starts connecting and returns immediately; the outcome arrives through `events`. Called once per instance.
        virtual auto connect(std::string url, ClientTransportEvents events) -> void = 0;

        // False if the connection is not open.
        virtual auto send(std::string text) -> bool = 0;

        // Closes the connection and joins the transport's threads. Safe to call twice and from the destructor.
        virtual auto close() -> void = 0;
    };

}
