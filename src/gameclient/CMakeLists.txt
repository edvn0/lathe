# ------------------------------------------------------------------------------
# engine_gameclient: the client side of lathe-server's protocol (GameClient) and the IClientTransport interface.
# Depends on engine_core only.
# ------------------------------------------------------------------------------

add_library(
    engine_gameclient
    STATIC
        game_client.cxx
)

target_include_directories(
    engine_gameclient
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_gameclient
    PUBLIC
        engine_deps
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_gameclient
    REUSE_FROM
        engine_core
)

# ------------------------------------------------------------------------------
# engine_gameclient_ix: IClientTransport on IXWebSocket. Kept apart so nothing else sees the library; a Beast client
# would be a sibling of this target.
# ------------------------------------------------------------------------------

add_library(
    engine_gameclient_ix
    STATIC
        ix_websocket_client_transport.cxx
)

target_link_libraries(
    engine_gameclient_ix
    PUBLIC
        engine_gameclient
    PRIVATE
        ixwebsocket
        engine_options
)
