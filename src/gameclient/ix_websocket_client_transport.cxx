#include "gameclient/ix_websocket_client_transport.hxx"

#include <atomic>
#include <memory>
#include <string>
#include <utility>

#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXWebSocketMessage.h>

namespace gameclient {

    namespace {
        class IxWebSocketClientTransport final : public IClientTransport {
        public:
            IxWebSocketClientTransport() = default;

            ~IxWebSocketClientTransport() override { close(); }

            IxWebSocketClientTransport(IxWebSocketClientTransport const &) = delete;
            auto operator=(IxWebSocketClientTransport const &) -> IxWebSocketClientTransport & = delete;
            IxWebSocketClientTransport(IxWebSocketClientTransport &&) = delete;
            auto operator=(IxWebSocketClientTransport &&) -> IxWebSocketClientTransport & = delete;

            auto connect(std::string url, ClientTransportEvents events) -> void override {
                events_ = std::move(events);

                socket_.setUrl(url);

                // GameClient decides when to redial, so a lost connection reports closed and stays closed.
                socket_.disableAutomaticReconnection();
                socket_.disablePerMessageDeflate();

                socket_.setOnMessageCallback([this](ix::WebSocketMessagePtr const &message) { handle(*message); });

                started_ = true;

                socket_.start();
            }

            auto send(std::string text) -> bool override {
                return socket_.getReadyState() == ix::ReadyState::Open && socket_.sendText(text).success;
            }

            auto close() -> void override {
                if (!started_.exchange(false)) {
                    return;
                }

                socket_.stop();
            }

        private:
            auto handle(ix::WebSocketMessage const &message) -> void {
                switch (message.type) {
                    case ix::WebSocketMessageType::Open:
                        events_.on_open();
                        break;
                    case ix::WebSocketMessageType::Message:
                        if (!message.binary) {
                            events_.on_message(message.str);
                        }
                        break;
                    case ix::WebSocketMessageType::Close:
                        report_closed(message.closeInfo.reason);
                        break;
                    case ix::WebSocketMessageType::Error:
                        report_closed(message.errorInfo.reason);
                        break;
                    case ix::WebSocketMessageType::Ping:
                    case ix::WebSocketMessageType::Pong:
                    case ix::WebSocketMessageType::Fragment:
                        break;
                }
            }

            // A failed connection can raise Error and then Close; the owner hears about it once.
            auto report_closed(std::string reason) -> void {
                if (!closed_.exchange(true)) {
                    events_.on_close(std::move(reason));
                }
            }

            ix::WebSocket socket_;

            ClientTransportEvents events_;

            std::atomic<bool> started_{false};
            std::atomic<bool> closed_{false};
        };
    }

    auto make_ix_websocket_client_transport() -> std::unique_ptr<IClientTransport> {
        return std::make_unique<IxWebSocketClientTransport>();
    }

}
