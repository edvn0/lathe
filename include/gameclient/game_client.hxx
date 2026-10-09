#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gameclient/client_transport.hxx"

namespace gameclient {

    using TransportFactory = std::function<std::unique_ptr<IClientTransport>()>;

    struct ClientEvent {
        enum class Kind : std::uint8_t {
            opened,
            message,
            closed,
        };

        Kind kind = Kind::message;

        // The JSON text for a message, the reason for closed.
        std::string text;
    };

    // The client side of lathe-server's protocol (see gameserver/game_server.hxx), game-agnostic. The transport runs on
    // its own threads and the game polls drain() once per frame, so no game code runs off the main thread.
    //
    // The session token from the server's first welcome is kept across reconnects: connect() starts a fresh session,
    // reconnect() dials the same server again, and resume() then takes the old seat back.
    class GameClient {
    public:
        explicit GameClient(TransportFactory factory);
        ~GameClient();

        GameClient(GameClient const &) = delete;
        auto operator=(GameClient const &) -> GameClient & = delete;
        GameClient(GameClient &&) = delete;
        auto operator=(GameClient &&) -> GameClient & = delete;

        // Starts a new session. Drops any current connection and forgets the old session token.
        auto connect(std::string url) -> void;

        // Dials the last URL again, keeping the session token. No-op before the first connect().
        auto reconnect() -> void;

        // Sends `{"type":"resume","token":...}`; false if there is no token or the connection is not open.
        auto resume() -> bool;

        // False if the connection is not open.
        auto send(std::string text) -> bool;

        // Forgets the session token, e.g. after the server rejects it as expired.
        auto forget_session() -> void;

        auto close() -> void;

        // Everything that happened since the last call, in order.
        [[nodiscard]] auto drain() -> std::vector<ClientEvent>;

        [[nodiscard]] auto is_open() const -> bool;
        [[nodiscard]] auto token() const -> std::string;

    private:
        auto dial() -> void;

        TransportFactory factory_;

        mutable std::mutex mutex_;

        // Shared so send() can use it without holding the lock while a dial replaces it.
        std::shared_ptr<IClientTransport> transport_;
        std::vector<ClientEvent> events_;

        std::string url_;
        std::string token_;

        // Bumped per dial so events from a replaced transport are ignored.
        std::uint64_t generation_ = 0;
        bool open_ = false;
    };

}
