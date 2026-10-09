#include "gameserver/ix_websocket_transport.hxx"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <ixwebsocket/IXConnectionState.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXWebSocketMessage.h>
#include <ixwebsocket/IXWebSocketServer.h>

namespace gameserver {

    namespace {
        struct ConnectionTag final : ix::ConnectionState {
            ConnectionId id = 0;
        };

        class IxWebSocketTransport final : public IServerTransport {
        public:
            IxWebSocketTransport(std::string host, std::uint16_t port)
                : server_(port, host), port_(port) {}

            ~IxWebSocketTransport() override { stop(); }

            IxWebSocketTransport(IxWebSocketTransport const &) = delete;
            auto operator=(IxWebSocketTransport const &) -> IxWebSocketTransport & = delete;
            IxWebSocketTransport(IxWebSocketTransport &&) = delete;
            auto operator=(IxWebSocketTransport &&) -> IxWebSocketTransport & = delete;

            auto start(TransportEvents events) -> std::expected<void, std::string> override {
                events_ = std::move(events);

                server_.setConnectionStateFactory([this]() -> std::shared_ptr<ix::ConnectionState> {
                    auto tag = std::make_shared<ConnectionTag>();

                    tag->id = next_connection_++;

                    return tag;
                });

                server_.setOnClientMessageCallback(
                        [this](std::shared_ptr<ix::ConnectionState> const &state, ix::WebSocket &socket,
                               ix::WebSocketMessagePtr const &message) {
                            handle(static_cast<ConnectionTag const &>(*state).id, socket, *message);
                        });

                auto const [bound, error] = server_.listen();

                if (!bound) {
                    return std::unexpected(error);
                }

                server_.start();

                running_ = true;

                return {};
            }

            auto stop() -> void override {
                if (!running_.exchange(false)) {
                    return;
                }

                server_.stop();
            }

            auto send(ConnectionId connection, std::string text) -> void override {
                std::scoped_lock const lock{mutex_};

                auto const socket = sockets_.find(connection);

                if (socket != sockets_.end()) {
                    socket->second->sendText(text);
                }
            }

            [[nodiscard]] auto port() const -> std::uint16_t override { return port_; }

        private:
            auto handle(ConnectionId connection, ix::WebSocket &socket, ix::WebSocketMessage const &message) -> void {
                switch (message.type) {
                    case ix::WebSocketMessageType::Open: {
                        {
                            std::scoped_lock const lock{mutex_};

                            sockets_[connection] = &socket;
                        }

                        events_.on_connect(connection);
                        break;
                    }
                    case ix::WebSocketMessageType::Message:
                        if (!message.binary) {
                            events_.on_message(connection, message.str);
                        }
                        break;
                    case ix::WebSocketMessageType::Close: {
                        bool known = false;

                        {
                            std::scoped_lock const lock{mutex_};

                            known = sockets_.erase(connection) != 0;
                        }

                        // A connection that never finished its handshake was never announced.
                        if (known) {
                            events_.on_disconnect(connection);
                        }
                        break;
                    }
                    case ix::WebSocketMessageType::Error:
                    case ix::WebSocketMessageType::Ping:
                    case ix::WebSocketMessageType::Pong:
                    case ix::WebSocketMessageType::Fragment:
                        break;
                }
            }

            ix::WebSocketServer server_;

            std::uint16_t port_;

            TransportEvents events_;

            std::mutex mutex_;
            std::unordered_map<ConnectionId, ix::WebSocket *> sockets_;

            std::atomic<ConnectionId> next_connection_{1};
            std::atomic<bool> running_{false};
        };
    }

    auto make_ix_websocket_transport(std::string host, std::uint16_t port) -> std::unique_ptr<IServerTransport> {
        return std::make_unique<IxWebSocketTransport>(std::move(host), port);
    }

}
