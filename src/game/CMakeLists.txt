# ------------------------------------------------------------------------------
# engine_game: the IGame interface a game implements and the GameHost it is handed (include/game/game.hxx). Header
# only. Sits above rendering, scene and terrain so a game, the Lua game and the editor can all depend on it without
# depending on each other.
# ------------------------------------------------------------------------------

add_library(engine_game INTERFACE)

target_include_directories(
    engine_game
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_game
    INTERFACE
        engine_deps
        engine_rendering
        engine_terrain
        engine_scene
        engine_core
)
