# ------------------------------------------------------------------------------
# engine_scene -- low-level ECS data (components, editor camera, input
# events). Not to be confused with Scene/Entity, which live in engine_rendering
# -- see MODULARIZATION_HANDOFF.md's "Scene/Entity reclassification" note.
# ------------------------------------------------------------------------------

add_library(
    engine_scene
    STATIC
        editor_camera.cxx
)

target_include_directories(
    engine_scene
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_scene
    PUBLIC
        engine_deps
        engine_physics
        engine_assets
        engine_gpu
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_scene
    REUSE_FROM
        engine_core
)
