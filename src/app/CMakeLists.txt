# ------------------------------------------------------------------------------
# engine_app: Application, the editor -- a Runtime (engine_runtime) with panels, gizmos, scene files and packaging on
# top. The game interface is engine_game, the Lua game is engine_luagame and the benchmark runner is engine_benchmark;
# none of them is part of the editor.
# ------------------------------------------------------------------------------

add_library(
    engine_app
    STATIC
        add_menu.cxx
        application.cxx
        commands.cxx
        cvar_panel.cxx
        editor_commands.cxx
        frame_graph_details.cxx
        frame_graph_editor.cxx
        game_packager.cxx
        gpu_timings_view.cxx
        load_progress.cxx
        cache_menu.cxx
        main_menu.cxx
        package_ui.cxx
        project_files.cxx
        project_settings.cxx
        scene_files.cxx
        scene_inspector.cxx
        stage_registry.cxx
        surface_pick.cxx
        text_placement.cxx
)

target_include_directories(
    engine_app
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_app
    PUBLIC
        engine_runtime
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
