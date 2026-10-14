# ------------------------------------------------------------------------------
# engine_luagame: LuaGame, the IGame that runs a Lua script, and the Lua API it gives scripts (scene, entity, assets,
# physics, animation, ui, ...), plus per-entity behaviours. Sits above engine_scripting (the sandboxed runtime) and
# engine_game (IGame); knows nothing of the editor in engine_app.
# ------------------------------------------------------------------------------

add_library(
    engine_luagame
    STATIC
        lua_animation.cxx
        lua_behaviours.cxx
        lua_game.cxx
        lua_game_api.cxx
)

target_include_directories(
    engine_luagame
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_luagame
    PUBLIC
        engine_game
        engine_scripting
        engine_deps
        engine_rendering
        engine_scene
        engine_physics
        engine_animation
        engine_assets
        engine_gpu
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_luagame
    REUSE_FROM
        engine_core
)
