# ------------------------------------------------------------------------------
# engine_app -- Application (owns Scene/PhysicsWorld lifecycle, editor/play
# state) and the renderer_application_policy.cxx impl (its .hxx stays in
# engine_rendering -- this is the one deliberate header/impl module split in
# the codebase, needed because RendererApplicationPolicy's .cxx needs the
# complete Application type).
# ------------------------------------------------------------------------------

add_library(
    engine_app
    STATIC
        application.cxx
        renderer_application_policy.cxx
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

# MINGW_VULKAN_TRACK_MEMORY gates the in-app Memory panel (application.cxx,
# memory_tracking_ui.hxx), the only consumer of this define. It's set from
# the same mingw_vulkan_track_memory variable the root CMakeLists.txt uses to
# decide whether global_new_delete.cxx is compiled into the executable, so
# the two stay in sync: no override without a MemoryTracker actually
# collecting data.
if(mingw_vulkan_track_memory)
  target_compile_definitions(
      engine_app
      PRIVATE
          MINGW_VULKAN_TRACK_MEMORY=1
  )
else()
  target_compile_definitions(
      engine_app
      PRIVATE
          MINGW_VULKAN_TRACK_MEMORY=0
  )
endif()
