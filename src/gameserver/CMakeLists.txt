# ------------------------------------------------------------------------------
# engine_gameserver: transport-agnostic room/session logic and the IServerGame plugin interface. Depends on
# engine_core only (no rendering, scene or GPU types), so a headless server can link it.
# ------------------------------------------------------------------------------

add_library(
    engine_gameserver
    STATIC
        game_server.cxx
)

target_include_directories(
    engine_gameserver
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_gameserver
    PUBLIC
        engine_deps
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_gameserver
    REUSE_FROM
        engine_core
)

# ------------------------------------------------------------------------------
# engine_gameserver_ix: IServerTransport on IXWebSocket. Kept apart so engine_gameserver (and its tests) never see the
# library; a Beast or uWebSockets transport would be a sibling of this target.
# ------------------------------------------------------------------------------

add_library(
    engine_gameserver_ix
    STATIC
        ix_websocket_transport.cxx
)

target_link_libraries(
    engine_gameserver_ix
    PUBLIC
        engine_gameserver
    PRIVATE
        ixwebsocket
        engine_options
)
