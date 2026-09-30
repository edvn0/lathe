# ------------------------------------------------------------------------------
# engine_app: Application (scene lifecycle, editor/play state, overlays).
# ------------------------------------------------------------------------------

add_library(
    engine_app
    STATIC
        application.cxx
        benchmark.cxx
)

target_include_directories(
    engine_app
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_app
    PUBLIC
        engine_deps
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
