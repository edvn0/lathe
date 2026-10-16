# ------------------------------------------------------------------------------
# engine_game: the IGame interface a game implements and the GameHost it is handed (include/game/game.hxx), plus scene
# content shared by sample games and the benchmark runner (the shadow city). Sits above rendering, scene and terrain so
# a game, the Lua game and the editor can all depend on it without depending on each other.
# ------------------------------------------------------------------------------

add_library(
    engine_game
    STATIC
        shadow_city.cxx
)

target_include_directories(
    engine_game
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_game
    PUBLIC
        engine_deps
        engine_rendering
        engine_terrain
        engine_scene
        engine_assets
        engine_gpu
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_game
    REUSE_FROM
        engine_core
)
