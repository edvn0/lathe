#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "gameserver/transport.hxx"

namespace gameserver {

    // WebSocket transport on IXWebSocket. The library stays out of this header; swap in another transport (Beast,
    // uWebSockets, ...) by implementing IServerTransport. The port must be non-zero.
    [[nodiscard]] auto make_ix_websocket_transport(std::string host, std::uint16_t port)
            -> std::unique_ptr<IServerTransport>;

}
