# ------------------------------------------------------------------------------
# engine_runtime: Runtime, everything a running game needs (renderer, scenes, the IGame lifecycle and input, terrain,
# particles, scene loading) and the declarations of run_engine, the shared startup and frame loop behind both executables (its definition,
# engine_main.cxx, is compiled into each executable because it needs the executable's game registry). The editor
# (engine_app) is a Runtime with panels on top; the player is a bare one. Knows nothing of either.
# ------------------------------------------------------------------------------

add_library(
    engine_runtime
    STATIC
        presentation_arguments.cxx
        runtime.cxx
        vulkan_bootstrap.cxx
)

target_include_directories(
    engine_runtime
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_runtime
    PUBLIC
        engine_game
        engine_scripting
        engine_deps
        engine_serialisation
        engine_rendering
        engine_terrain
        engine_scene
        engine_physics
        engine_assets
        engine_gpu
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_runtime
    REUSE_FROM
        engine_core
)
