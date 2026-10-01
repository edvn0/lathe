# ------------------------------------------------------------------------------
# engine_scene: components, editor camera, input events. Scene/Entity live in
# engine_rendering.
# ------------------------------------------------------------------------------

add_library(
    engine_scene
    STATIC
        camera_path.cxx
        editor_camera.cxx
        hierarchy_model.cxx
        selection_context.cxx
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
