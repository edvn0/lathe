# ------------------------------------------------------------------------------
# engine_app: Application (scene lifecycle, editor/play state, overlays).
# ------------------------------------------------------------------------------

add_library(
    engine_app
    STATIC
        application.cxx
        benchmark.cxx
        benchmark_analysis.cxx
        benchmark_compare.cxx
        benchmark_driver.cxx
        benchmark_environment.cxx
        benchmark_scenarios.cxx
        lua_game.cxx
        lua_game_api.cxx
        scene_files.cxx
)

# Recorded in benchmark results (app/benchmark_environment.hxx): the build type, and where to read the source
# revision from.
target_compile_definitions(
    engine_app
    PRIVATE
        LATHE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        LATHE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_include_directories(
    engine_app
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_app
    PUBLIC
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
    engine_app
    REUSE_FROM
        engine_core
)

# Enables the in-app Memory panel; set together with the global new/delete
# override so the panel always has data.
if(lathe_track_memory)
  target_compile_definitions(
      engine_app
      PRIVATE
          LATHE_TRACK_MEMORY=1
  )
else()
  target_compile_definitions(
      engine_app
      PRIVATE
          LATHE_TRACK_MEMORY=0
  )
endif()
