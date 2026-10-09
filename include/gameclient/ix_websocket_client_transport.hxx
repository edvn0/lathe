#pragma once

#include <memory>

#include "gameclient/client_transport.hxx"

namespace gameclient {

    // WebSocket client on IXWebSocket (plain ws:// only). The library stays out of this header; swap in another
    // transport by implementing IClientTransport.
    [[nodiscard]] auto make_ix_websocket_client_transport() -> std::unique_ptr<IClientTransport>;

}
